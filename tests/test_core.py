import numpy as np

from gym_posture.core.scoring import grade, ramp
from gym_posture.core.signals import DipConfig, find_dips
from gym_posture.exercises.base import Lift
from gym_posture.render.video import clip_frames

FPS: float = 30.0


def _dip(frames: int) -> np.ndarray:
    """One smooth dip to depth 1.0 and back."""
    return np.sin(np.linspace(0, np.pi, frames))


def test_ramp_clips_and_interpolates() -> None:
    assert float(ramp(0.0, best=0.0, worst=-0.08)) == 100
    assert float(ramp(-0.04, best=0.0, worst=-0.08)) == 50
    assert float(ramp(-1.0, best=0.0, worst=-0.08)) == 0


def test_find_dips_counts_dips_and_ignores_incomplete_ones() -> None:
    flat: np.ndarray = np.zeros(30)
    signal: np.ndarray = np.concatenate([flat, _dip(60), flat, _dip(60), flat, _dip(60)[:40]])
    reps = find_dips(signal, gate=signal, fps=FPS)
    assert len(reps) == 2  # the last dip runs off the end of the video
    assert all(rep.start < rep.bottom < rep.end for rep in reps)


def test_find_dips_ignores_dips_the_gate_does_not_confirm() -> None:
    signal: np.ndarray = np.concatenate([np.zeros(30), _dip(60), np.zeros(30)])
    assert find_dips(signal, gate=np.zeros_like(signal), fps=FPS) == []


def test_clip_frames_merges_overlapping_windows() -> None:
    assert clip_frames([(40, 60), (70, 90)], fps=10.0, total=200) == list(range(30, 101))


def test_grade_thresholds() -> None:
    assert grade(90) == "GOOD"
    assert grade(70) == "OK"
    assert grade(40) == "FIX"


def test_lift_load_text_and_total() -> None:
    assert Lift(120, 75).load_text == "120 kg"
    dumbbells: Lift = Lift(30, 75, implements=2)
    assert dumbbells.load_text == "2×30 kg"
    assert dumbbells.total_kg == 60


def test_find_dips_rejects_too_deep_or_too_long_dips() -> None:
    flat: np.ndarray = np.zeros(30)
    rep: np.ndarray = _dip(60)
    setup: np.ndarray = 3 * _dip(60)  # lying back: dumbbells far below the shoulders
    slow: np.ndarray = _dip(400)  # sitting up: takes far longer than a rep
    signal: np.ndarray = np.concatenate([flat, setup, flat, rep, flat, slow, flat])
    config: DipConfig = DipConfig(max_depth=1.3, max_seconds=6.0)
    reps = find_dips(signal, gate=signal, fps=FPS, config=config)
    assert len(reps) == 1
    assert 120 < reps[0].bottom < 180
