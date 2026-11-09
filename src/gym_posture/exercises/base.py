"""The interface every exercise implements, plus the shared scoring flow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from gym_posture.core.pose import Backend
from gym_posture.core.scoring import FrameScores, RepScore, SetScore
from gym_posture.core.signals import Rep


@dataclass(frozen=True)
class Lift:
    """What was lifted, for the titles and the strength ratio."""

    load_kg: float  # per implement (one barbell, or each dumbbell)
    bodyweight_kg: float
    implements: int = 1

    @property
    def total_kg(self) -> float:
        return self.load_kg * self.implements

    @property
    def load_text(self) -> str:
        """'120 kg' for a barbell, '2×30 kg' for a pair of dumbbells."""
        if self.implements == 1:
            return f"{self.load_kg:g} kg"
        return f"{self.implements}×{self.load_kg:g} kg"


@dataclass(frozen=True)
class Bone:
    """A skeleton line to draw, coloured by the status of one check."""

    start: int
    end: int
    check: str


class Exercise(ABC):
    """One lift: how to find its reps and how to score them."""

    title: ClassVar[str]  # shown on the video, e.g. "BACK SQUAT"
    coach: ClassVar[str]  # brand line above the title, e.g. "AI SQUAT COACH"
    slug: ClassVar[str]  # output folder name
    backend: ClassVar[Backend]  # pose model that works best for the camera angle
    implements: ClassVar[int] = 1
    # Weight of each check in a rep's overall score (sums to 1), in display order.
    weights: ClassVar[dict[str, float]]
    # Checks that make sense on every single frame (they drive the live meter).
    live_checks: ClassVar[tuple[str, ...]]
    # Coaching cue for each check, used when that check scores low.
    cues: ClassVar[dict[str, str]]
    fatigue_cue: ClassVar[str] = "Rep {rep} broke down most: fatigue shows there first."
    bones: ClassVar[tuple[Bone, ...]]
    joints: ClassVar[tuple[int, ...]]
    # The progress bar under the graph (e.g. squat depth), 1.0 = target reached.
    progress_label: ClassVar[str]
    progress_target: ClassVar[str]

    @abstractmethod
    def find_reps(self, points: np.ndarray, fps: float) -> list[Rep]: ...

    @abstractmethod
    def progress(self, points: np.ndarray, fps: float) -> np.ndarray:
        """Per-frame position in the rep: 0 = top, 1 = target depth."""

    @abstractmethod
    def frame_checks(self, points: np.ndarray, fps: float) -> dict[str, np.ndarray]:
        """Scores (0-100) per frame for the checks listed in `live_checks`."""

    @abstractmethod
    def rep_checks(
        self, rep: Rep, points: np.ndarray, frames: FrameScores, fps: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        """(score per check, raw measurements) for one rep."""

    # --- shared flow -------------------------------------------------------------
    def score_frames(self, points: np.ndarray, fps: float) -> FrameScores:
        checks: dict[str, np.ndarray] = self.frame_checks(points, fps)
        total_weight: float = sum(self.weights[name] for name in self.live_checks)
        weighted: list[np.ndarray] = [
            checks[name] * self.weights[name] for name in self.live_checks
        ]
        return FrameScores(checks=checks, live=np.sum(weighted, axis=0) / total_weight)

    def score_rep(self, rep: Rep, points: np.ndarray, frames: FrameScores, fps: float) -> RepScore:
        checks, details = self.rep_checks(rep, points, frames, fps)
        overall: float = sum(checks[name] * weight for name, weight in self.weights.items())
        return RepScore(rep=rep, checks=checks, overall=overall, details=details)

    def score_set(self, reps: list[RepScore]) -> SetScore:
        checks: dict[str, float] = {
            name: float(np.mean([rep.checks[name] for rep in reps])) for name in self.weights
        }
        overall: float = float(np.mean([rep.overall for rep in reps]))
        return SetScore(reps=reps, checks=checks, overall=overall)

    def tips(self, result: SetScore) -> list[str]:
        """Deterministic coaching cues for the weakest checks and the weakest rep."""
        tips: list[str] = []
        for name, value in sorted(result.checks.items(), key=lambda item: item[1]):
            if value < 90 and len(tips) < 2:
                tips.append(self.cues[name])
        weakest: RepScore = min(result.reps, key=lambda rep: rep.overall)
        if weakest.overall < result.overall - 8:
            tips.append(self.fatigue_cue.format(rep=result.reps.index(weakest) + 1))
        if not tips:
            tips.append("Clean set. Ready to add weight.")
        return tips
