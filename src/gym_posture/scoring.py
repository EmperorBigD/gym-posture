"""Score squat form (front view) per frame and per rep, 0-100%.

Checks, inspired by open-source MediaPipe squat analysers (e.g. LearnOpenCV's
"AI Fitness Trainer" and Pose-Trainer), adapted from side view to a front camera:

- Depth:     hips reach knee height at the bottom (front-view estimate).
- Knees:     knees stay on or outside the hip-ankle line (no valgus / caving in).
- Balance:   pelvis stays centred over the feet (no hip shift to one side).
- Bar level: shoulders and hands stay level (bar not tilting).
- Tempo:     controlled descent and a steady drive up (no long grind).
"""

from dataclasses import dataclass, field

import numpy as np

from gym_posture.reps import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    Rep,
)

# Weight of each check in a rep's overall score (sums to 1).
WEIGHTS: dict[str, float] = {
    "Depth": 0.30,
    "Knees": 0.25,
    "Balance": 0.20,
    "Tempo": 0.15,
    "Bar level": 0.10,
}
# Checks that can be judged on every single frame (used for the live meter).
LIVE_CHECKS: tuple[str, ...] = ("Knees", "Balance", "Bar level")


def ramp(value: float | np.ndarray, best: float, worst: float) -> np.ndarray:
    """Map `value` linearly to 100 at `best` and 0 at `worst`, clipped to 0-100."""
    return np.clip(100 * (np.asarray(value) - worst) / (best - worst), 0, 100)


def _knee_offset(points: np.ndarray, hip: int, knee: int, ankle: int, outward: float) -> np.ndarray:
    """Knee distance from the hip-ankle line in leg lengths; positive = outside (good)."""
    hx, hy = points[:, hip, 0], points[:, hip, 1]
    kx, ky = points[:, knee, 0], points[:, knee, 1]
    ax, ay = points[:, ankle, 0], points[:, ankle, 1]
    along: np.ndarray = (ky - hy) / (ay - hy)
    line_x: np.ndarray = hx + along * (ax - hx)
    leg: np.ndarray = np.hypot(ax - hx, ay - hy)
    return outward * (kx - line_x) / leg


def _tilt_degrees(points: np.ndarray, left: int, right: int) -> np.ndarray:
    dy: np.ndarray = points[:, left, 1] - points[:, right, 1]
    dx: np.ndarray = points[:, left, 0] - points[:, right, 0]
    return np.abs(np.degrees(np.arctan2(dy, dx)))


@dataclass
class FrameScores:
    """Per-frame check scores (0-100) and the raw measurements behind them."""

    knee_offset: np.ndarray  # worse of the two legs, leg lengths
    hip_shift: np.ndarray  # pelvis offset from feet centre, stance widths
    bar_tilt: np.ndarray  # degrees
    checks: dict[str, np.ndarray] = field(default_factory=dict)
    live: np.ndarray = field(default_factory=lambda: np.empty(0))


def score_frames(points: np.ndarray) -> FrameScores:
    # The lifter faces the camera, so their left side is on the image's right (+x).
    left: np.ndarray = _knee_offset(points, L_HIP, L_KNEE, L_ANKLE, outward=1.0)
    right: np.ndarray = _knee_offset(points, R_HIP, R_KNEE, R_ANKLE, outward=-1.0)
    knee_offset: np.ndarray = np.minimum(left, right)

    pelvis_x: np.ndarray = (points[:, L_HIP, 0] + points[:, R_HIP, 0]) / 2
    feet_x: np.ndarray = (points[:, L_ANKLE, 0] + points[:, R_ANKLE, 0]) / 2
    stance: np.ndarray = np.abs(points[:, L_ANKLE, 0] - points[:, R_ANKLE, 0])
    hip_shift: np.ndarray = np.abs(pelvis_x - feet_x) / stance

    bar_tilt: np.ndarray = np.maximum(
        _tilt_degrees(points, L_SHOULDER, R_SHOULDER), _tilt_degrees(points, L_WRIST, R_WRIST)
    )
    checks: dict[str, np.ndarray] = {
        "Knees": ramp(knee_offset, best=0.0, worst=-0.08),
        "Balance": ramp(hip_shift, best=0.04, worst=0.20),
        "Bar level": ramp(bar_tilt, best=2.0, worst=8.0),
    }
    total_weight: float = sum(WEIGHTS[name] for name in LIVE_CHECKS)
    weighted: list[np.ndarray] = [checks[name] * WEIGHTS[name] for name in LIVE_CHECKS]
    live: np.ndarray = np.sum(weighted, axis=0) / total_weight
    return FrameScores(knee_offset, hip_shift, bar_tilt, checks, live)


@dataclass
class RepScore:
    rep: Rep
    checks: dict[str, float]
    overall: float
    depth_gap: float  # hip-to-knee height left at the bottom, in thigh lengths
    descent_s: float
    ascent_s: float


def score_rep(rep: Rep, points: np.ndarray, frames: FrameScores, fps: float) -> RepScore:
    window: slice = slice(rep.start, rep.end + 1)
    hip_y: float = float((points[rep.bottom, L_HIP, 1] + points[rep.bottom, R_HIP, 1]) / 2)
    knee_y: float = float((points[rep.bottom, L_KNEE, 1] + points[rep.bottom, R_KNEE, 1]) / 2)
    before: slice = slice(max(0, rep.start - round(fps)), rep.start + 1)
    standing_thigh: float = float(
        np.max(
            (points[before, L_KNEE, 1] + points[before, R_KNEE, 1]) / 2
            - (points[before, L_HIP, 1] + points[before, R_HIP, 1]) / 2
        )
    )
    depth_gap: float = (knee_y - hip_y) / standing_thigh
    descent_s: float = (rep.bottom - rep.start) / fps
    ascent_s: float = (rep.end - rep.bottom) / fps

    # Worst moments decide the rep, using a low percentile to ignore single-frame glitches.
    checks: dict[str, float] = {
        # From the front a parallel squat still shows a small hip-knee gap (~0.1 thigh).
        "Depth": float(ramp(depth_gap, best=0.12, worst=0.50)),
        "Knees": float(np.percentile(frames.checks["Knees"][window], 5)),
        "Balance": float(np.percentile(frames.checks["Balance"][window], 5)),
        "Tempo": float(
            (ramp(descent_s, best=1.0, worst=0.4) + ramp(ascent_s, best=2.5, worst=5.0)) / 2
        ),
        "Bar level": float(np.percentile(frames.checks["Bar level"][window], 5)),
    }
    overall: float = sum(checks[name] * weight for name, weight in WEIGHTS.items())
    return RepScore(rep, checks, overall, depth_gap, descent_s, ascent_s)


@dataclass
class SetScore:
    reps: list[RepScore]
    checks: dict[str, float]
    overall: float


def score_set(reps: list[RepScore]) -> SetScore:
    checks: dict[str, float] = {
        name: float(np.mean([rep.checks[name] for rep in reps])) for name in WEIGHTS
    }
    overall: float = float(np.mean([rep.overall for rep in reps]))
    return SetScore(reps, checks, overall)


def coach_tips(result: SetScore) -> list[str]:
    """Deterministic coaching cues for the weakest checks."""
    cues: dict[str, str] = {
        "Depth": "Sit a little deeper: aim for hip crease at or below the knee.",
        "Knees": "Push knees out over your toes, especially on the way up.",
        "Balance": "Keep weight even on both feet; hips are drifting to one side.",
        "Tempo": "Brace harder before the last reps so the drive up stays fast.",
        "Bar level": "Keep the bar level: even grip and both shoulders tight.",
    }
    tips: list[str] = []
    for name, value in sorted(result.checks.items(), key=lambda item: item[1]):
        if value < 90 and len(tips) < 2:
            tips.append(cues[name])
    weakest_rep: RepScore = min(result.reps, key=lambda rep: rep.overall)
    if weakest_rep.overall < result.overall - 8:
        index: int = result.reps.index(weakest_rep) + 1
        tips.append(f"Rep {index} broke down most: fatigue shows there first.")
    if not tips:
        tips.append("Clean set. Ready to add weight.")
    return tips
