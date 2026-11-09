import numpy as np

from gym_posture.render import clip_frames, status
from gym_posture.reps import find_reps
from gym_posture.scoring import ramp

FPS: float = 30.0


def _squat(frames: int) -> np.ndarray:
    """One smooth dip to depth 1.0 and back."""
    return np.sin(np.linspace(0, np.pi, frames))


def test_ramp_clips_and_interpolates() -> None:
    assert float(ramp(0.0, best=0.0, worst=-0.08)) == 100
    assert float(ramp(-0.04, best=0.0, worst=-0.08)) == 50
    assert float(ramp(-1.0, best=0.0, worst=-0.08)) == 0


def test_find_reps_counts_dips_and_ignores_incomplete_ones() -> None:
    flat: np.ndarray = np.zeros(30)
    depth: np.ndarray = np.concatenate([flat, _squat(60), flat, _squat(60), flat, _squat(60)[:40]])
    reps = find_reps(depth, shoulders=depth, fps=FPS)
    assert len(reps) == 2  # the last dip runs off the end of the video
    assert all(rep.start < rep.bottom < rep.end for rep in reps)


def test_find_reps_ignores_dips_without_shoulder_drop() -> None:
    depth: np.ndarray = np.concatenate([np.zeros(30), _squat(60), np.zeros(30)])
    assert find_reps(depth, shoulders=np.zeros_like(depth), fps=FPS) == []


def test_clip_frames_merges_overlapping_windows() -> None:
    keep: list[int] = clip_frames([(40, 60), (70, 90)], fps=10.0, total=200)
    assert keep == list(range(30, 101))


def test_status_thresholds() -> None:
    assert status(90)[1] == "GOOD"
    assert status(70)[1] == "OK"
    assert status(40)[1] == "FIX"
