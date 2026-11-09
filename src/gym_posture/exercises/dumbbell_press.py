"""Incline dumbbell press, filmed from the feet looking up the bench.

From this angle the arms move mostly in the image plane, so we can judge:

- Range:     elbows bend deep at the bottom (dumbbells down at chest level).
- Lockout:   elbows straighten at the top.
- Symmetry:  both dumbbells move together (no side lagging behind the other).
- Wrists:    forearms stay vertical, wrists stacked over elbows.
- Tempo:     controlled lowering and a steady press (no long grind).

Symmetry ignores a steady height offset between the sides: from a low, slightly tilted
camera that offset can be perspective rather than form, so only lag is scored.
Elbow flare (upper-arm angle to the torso) is NOT scored: from the feet the torso is
too foreshortened to measure it reliably. A 45° side view would be needed.
"""

from typing import ClassVar

import numpy as np

from gym_posture.core.pose import (
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    Backend,
)
from gym_posture.core.scoring import FrameScores, ramp
from gym_posture.core.signals import DipConfig, Rep, find_dips, rolling
from gym_posture.exercises.base import Bone, Exercise

# Window for the local "lockout height" reference (longer than one rep).
BASELINE_SECONDS: float = 8.0
# Lying back / sitting up drops the dumbbells far below the shoulders for many seconds;
# real reps stay above shoulder level and take a few seconds.
DIPS: DipConfig = DipConfig(max_depth=1.3, max_seconds=6.0)
# Joint measurement is noisy near the bottom/top, so look this far past the rep edges.
LOCKOUT_SEARCH_SECONDS: float = 0.5

SIDES: tuple[tuple[int, int, int], ...] = (
    (L_SHOULDER, L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW, R_WRIST),
)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    padded: np.ndarray = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[: len(values)]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle ABC in degrees for (frames, 2) point arrays."""
    ba: np.ndarray = a - b
    bc: np.ndarray = c - b
    cos: np.ndarray = np.sum(ba * bc, axis=1) / (
        np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    )
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


class DumbbellPress(Exercise):
    title: ClassVar[str] = "INCLINE DB PRESS"
    coach: ClassVar[str] = "AI PRESS COACH"
    slug: ClassVar[str] = "dumbbell_press"
    backend: ClassVar[Backend] = Backend.RTMPOSE
    implements: ClassVar[int] = 2
    weights: ClassVar[dict[str, float]] = {
        "Range": 0.25,
        "Lockout": 0.25,
        "Symmetry": 0.25,
        "Wrists": 0.15,
        "Tempo": 0.10,
    }
    live_checks: ClassVar[tuple[str, ...]] = ("Symmetry", "Wrists")
    cues: ClassVar[dict[str, str]] = {
        "Range": "Lower the dumbbells all the way to chest level.",
        "Lockout": "Finish each rep: press until your elbows are straight.",
        "Symmetry": "Press both dumbbells together; one side is lagging.",
        "Wrists": "Keep wrists stacked over elbows, forearms vertical.",
        "Tempo": "Lower under control (~1-2 s), then press up with intent.",
    }
    bones: ClassVar[tuple[Bone, ...]] = (
        Bone(L_SHOULDER, L_ELBOW, "Wrists"),
        Bone(L_ELBOW, L_WRIST, "Wrists"),
        Bone(R_SHOULDER, R_ELBOW, "Wrists"),
        Bone(R_ELBOW, R_WRIST, "Wrists"),
        Bone(L_SHOULDER, R_SHOULDER, "Symmetry"),
        Bone(L_SHOULDER, L_HIP, "Symmetry"),
        Bone(R_SHOULDER, R_HIP, "Symmetry"),
        Bone(L_HIP, R_HIP, "Symmetry"),
    )
    joints: ClassVar[tuple[int, ...]] = (
        L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP,
    )  # fmt: skip
    progress_label: ClassVar[str] = "RANGE"
    progress_target: ClassVar[str] = "chest"

    def _side_progress(self, points: np.ndarray, fps: float) -> np.ndarray:
        """(frames, 2): how far each dumbbell is lowered, 0 = lockout, 1 = shoulder level."""
        window: int = round(BASELINE_SECONDS * fps) | 1
        out: np.ndarray = np.empty((len(points), 2))
        for side, (shoulder, _, wrist) in enumerate(SIDES):
            height: np.ndarray = points[:, shoulder, 1] - points[:, wrist, 1]
            # At least 1 px, so frames where the dumbbells never rise (sitting) stay finite.
            lockout: np.ndarray = np.maximum(rolling(height, window, "max"), 1.0)
            out[:, side] = 1 - height / lockout
        return out

    def progress(self, points: np.ndarray, fps: float) -> np.ndarray:
        return self._side_progress(points, fps).mean(axis=1)

    def find_reps(self, points: np.ndarray, fps: float) -> list[Rep]:
        sides: np.ndarray = self._side_progress(points, fps)
        # Both arms must come down: a one-arm adjustment is not a rep.
        return find_dips(sides.mean(axis=1), gate=sides.min(axis=1), fps=fps, config=DIPS)

    def _measures(self, points: np.ndarray) -> dict[str, np.ndarray]:
        """Raw per-frame measurements, worse side of the two."""
        elbow: list[np.ndarray] = []
        forearm: list[np.ndarray] = []
        for shoulder, elb, wrist in SIDES:
            s, e, w = points[:, shoulder], points[:, elb], points[:, wrist]
            elbow.append(_angle(s, e, w))
            # Forearm angle from vertical (wrist straight above the elbow = 0°).
            forearm.append(np.abs(np.degrees(np.arctan2(w[:, 0] - e[:, 0], e[:, 1] - w[:, 1]))))
        return {
            "elbow_min_deg": np.minimum(*elbow),  # the more bent arm
            "elbow_max_deg": np.maximum(*elbow),  # the less bent arm
            "forearm_tilt_deg": np.maximum(*forearm),
        }

    def frame_checks(self, points: np.ndarray, fps: float) -> dict[str, np.ndarray]:
        sides: np.ndarray = self._side_progress(points, fps)
        measures: dict[str, np.ndarray] = self._measures(points)
        gap: np.ndarray = sides[:, 0] - sides[:, 1]
        lag: np.ndarray = np.abs(gap - _moving_average(gap, round(BASELINE_SECONDS * fps) | 1))
        return {
            "Symmetry": ramp(lag, best=0.08, worst=0.35),
            # Filmed from below, truly vertical forearms still lean ~10-15° in the image.
            "Wrists": ramp(measures["forearm_tilt_deg"], best=15.0, worst=40.0),
        }

    def rep_checks(
        self, rep: Rep, points: np.ndarray, frames: FrameScores, fps: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        window: slice = slice(rep.start, rep.end + 1)
        measures: dict[str, np.ndarray] = self._measures(points)
        search: int = round(LOCKOUT_SEARCH_SECONDS * fps)
        bottom_window: slice = slice(max(0, rep.bottom - search), rep.bottom + search + 1)
        top_window: slice = slice(rep.bottom, min(len(points), rep.end + search + 1))
        # Range: how bent the LESS bent arm gets at the bottom (both must go deep).
        bottom_elbow_deg: float = float(np.min(measures["elbow_max_deg"][bottom_window]))
        # Lockout: how straight the MORE bent arm gets at the top (both must lock out).
        lockout_deg: float = float(np.max(measures["elbow_min_deg"][top_window]))
        descent_s: float = (rep.bottom - rep.start) / fps
        ascent_s: float = (rep.end - rep.bottom) / fps
        checks: dict[str, float] = {
            "Range": float(ramp(bottom_elbow_deg, best=80.0, worst=115.0)),
            "Lockout": float(ramp(lockout_deg, best=160.0, worst=130.0)),
            "Symmetry": float(np.percentile(frames.checks["Symmetry"][window], 5)),
            "Wrists": float(np.percentile(frames.checks["Wrists"][window], 5)),
            "Tempo": float(
                (ramp(descent_s, best=1.0, worst=0.4) + ramp(ascent_s, best=2.0, worst=4.0)) / 2
            ),
        }
        details: dict[str, float] = {
            "bottom_elbow_deg": bottom_elbow_deg,
            "lockout_elbow_deg": lockout_deg,
            "descent_s": descent_s,
            "ascent_s": ascent_s,
        }
        return checks, details
