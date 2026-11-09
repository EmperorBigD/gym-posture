"""Barbell back squat, filmed from the front.

Checks, adapted from open-source MediaPipe squat analysers (LearnOpenCV's
"AI Fitness Trainer", Pose-Trainer) from side view to a front camera:

- Depth:     hips reach knee height at the bottom (front-view estimate).
- Knees:     knees stay on or outside the hip-ankle line (no valgus / caving in).
- Balance:   pelvis stays centred over the feet (no hip shift to one side).
- Tempo:     controlled descent and a steady drive up (no long grind).
- Bar level: shoulders and hands stay level (bar not tilting).
"""

from typing import ClassVar

import numpy as np

from gym_posture.core.pose import (
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
    Backend,
)
from gym_posture.core.scoring import FrameScores, ramp
from gym_posture.core.signals import Rep, find_dips, rolling
from gym_posture.exercises.base import Bone, Exercise

# Window for the local "standing" reference (longer than one rep), so stepping back
# or the camera shifting does not skew depth.
BASELINE_SECONDS: float = 8.0


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


def _mid_y(points: np.ndarray, left: int, right: int) -> np.ndarray:
    return (points[:, left, 1] + points[:, right, 1]) / 2


class Squat(Exercise):
    title: ClassVar[str] = "BACK SQUAT"
    coach: ClassVar[str] = "AI SQUAT COACH"
    slug: ClassVar[str] = "squat"
    backend: ClassVar[Backend] = Backend.MEDIAPIPE
    weights: ClassVar[dict[str, float]] = {
        "Depth": 0.30,
        "Knees": 0.25,
        "Balance": 0.20,
        "Tempo": 0.15,
        "Bar level": 0.10,
    }
    live_checks: ClassVar[tuple[str, ...]] = ("Knees", "Balance", "Bar level")
    cues: ClassVar[dict[str, str]] = {
        "Depth": "Sit a little deeper: aim for hip crease at or below the knee.",
        "Knees": "Push knees out over your toes, especially on the way up.",
        "Balance": "Keep weight even on both feet; hips are drifting to one side.",
        "Tempo": "Brace harder before the last reps so the drive up stays fast.",
        "Bar level": "Keep the bar level: even grip and both shoulders tight.",
    }
    bones: ClassVar[tuple[Bone, ...]] = (
        Bone(L_HIP, L_KNEE, "Knees"),
        Bone(L_KNEE, L_ANKLE, "Knees"),
        Bone(R_HIP, R_KNEE, "Knees"),
        Bone(R_KNEE, R_ANKLE, "Knees"),
        Bone(L_HIP, R_HIP, "Balance"),
        Bone(L_SHOULDER, L_HIP, "Balance"),
        Bone(R_SHOULDER, R_HIP, "Balance"),
        Bone(L_SHOULDER, R_SHOULDER, "Bar level"),
        Bone(L_WRIST, R_WRIST, "Bar level"),
    )
    joints: ClassVar[tuple[int, ...]] = (
        L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    )  # fmt: skip
    progress_label: ClassVar[str] = "DEPTH"
    progress_target: ClassVar[str] = "parallel"

    def _drop(self, points: np.ndarray, fps: float, top_y: np.ndarray) -> np.ndarray:
        """How far `top_y` has dropped from its local standing height, in thigh lengths."""
        window: int = round(BASELINE_SECONDS * fps) | 1
        thigh: np.ndarray = _mid_y(points, L_KNEE, R_KNEE) - _mid_y(points, L_HIP, R_HIP)
        return (top_y - rolling(top_y, window, "min")) / rolling(thigh, window, "max")

    def progress(self, points: np.ndarray, fps: float) -> np.ndarray:
        """Hip drop in thigh lengths (0 = standing, ~1 = hips at knee height)."""
        return self._drop(points, fps, _mid_y(points, L_HIP, R_HIP))

    def find_reps(self, points: np.ndarray, fps: float) -> list[Rep]:
        # Shoulders must drop with the hips: rules out walking towards the camera.
        shoulders: np.ndarray = self._drop(points, fps, _mid_y(points, L_SHOULDER, R_SHOULDER))
        return find_dips(self.progress(points, fps), gate=shoulders, fps=fps)

    def frame_checks(self, points: np.ndarray, fps: float) -> dict[str, np.ndarray]:
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
        return {
            "Knees": ramp(knee_offset, best=0.0, worst=-0.08),
            "Balance": ramp(hip_shift, best=0.04, worst=0.20),
            "Bar level": ramp(bar_tilt, best=2.0, worst=8.0),
        }

    def rep_checks(
        self, rep: Rep, points: np.ndarray, frames: FrameScores, fps: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        window: slice = slice(rep.start, rep.end + 1)
        before: slice = slice(max(0, rep.start - round(fps)), rep.start + 1)
        hip_y: np.ndarray = _mid_y(points, L_HIP, R_HIP)
        knee_y: np.ndarray = _mid_y(points, L_KNEE, R_KNEE)
        standing_thigh: float = float(np.max(knee_y[before] - hip_y[before]))
        depth_gap: float = float(knee_y[rep.bottom] - hip_y[rep.bottom]) / standing_thigh
        descent_s: float = (rep.bottom - rep.start) / fps
        ascent_s: float = (rep.end - rep.bottom) / fps

        # Worst moments decide the rep, using a low percentile to ignore one-frame glitches.
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
        details: dict[str, float] = {
            "depth_gap_thigh": depth_gap,
            "descent_s": descent_s,
            "ascent_s": ascent_s,
        }
        return checks, details
