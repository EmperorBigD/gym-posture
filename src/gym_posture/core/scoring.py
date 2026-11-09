"""Score containers and helpers shared by every exercise."""

from dataclasses import dataclass, field

import numpy as np

from gym_posture.core.signals import Rep

GOOD_FROM: float = 85.0
OK_FROM: float = 65.0


def ramp(value: float | np.ndarray, best: float, worst: float) -> np.ndarray:
    """Map `value` linearly to 100 at `best` and 0 at `worst`, clipped to 0-100."""
    return np.clip(100 * (np.asarray(value) - worst) / (best - worst), 0, 100)


def grade(score: float) -> str:
    """Text label for a 0-100 score (always shown next to its status colour)."""
    if score >= GOOD_FROM:
        return "GOOD"
    if score >= OK_FROM:
        return "OK"
    return "FIX"


@dataclass
class FrameScores:
    """Per-frame check scores (0-100) and the weighted live score."""

    checks: dict[str, np.ndarray]
    live: np.ndarray


@dataclass
class RepScore:
    rep: Rep
    checks: dict[str, float]
    overall: float
    # Raw measurements behind the scores, written to the JSON report.
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class SetScore:
    reps: list[RepScore]
    checks: dict[str, float]
    overall: float
