"""Clean joint signals and find reps in them (exercise-independent)."""

from dataclasses import dataclass

import numpy as np

from gym_posture.core.pose import JOINT_COUNT, PoseTrack

SMOOTH_SECONDS: float = 0.15


@dataclass
class Rep:
    """Frame indices of one rep: start (top), bottom (deepest point) and end (top again)."""

    start: int
    bottom: int
    end: int


@dataclass(frozen=True)
class DipConfig:
    """How to cut a 0 (top) .. ~1 (bottom) signal into reps."""

    # A dip only counts as a rep if the signal reaches at least this deep.
    min_depth: float = 0.6
    # A rep starts / ends once the signal is back within this of the top...
    edge: float = 0.12
    # ...or once it is above `settle_depth` and moves less than `settle_move`
    # over `settle_seconds` (e.g. the lifter stops short of full lockout to re-rack).
    settle_depth: float = 0.3
    settle_move: float = 0.03
    settle_seconds: float = 0.4
    # Reject "dips" deeper or longer than this (getting on / off the bench, re-racking).
    max_depth: float = float("inf")
    max_seconds: float = float("inf")


DEFAULT_DIPS: DipConfig = DipConfig()


def _fill_and_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Linearly fill NaN gaps, then apply a centred moving average."""
    index: np.ndarray = np.arange(len(values))
    valid: np.ndarray = ~np.isnan(values)
    filled: np.ndarray = np.interp(index, index[valid], values[valid])
    if window <= 1:
        return filled
    kernel: np.ndarray = np.ones(window) / window
    padded: np.ndarray = np.pad(filled, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def clean_points(track: PoseTrack) -> np.ndarray:
    """Return (frames, 17, 2) x/y pixel positions with gaps filled and jitter smoothed."""
    window: int = max(1, round(SMOOTH_SECONDS * track.fps))
    out: np.ndarray = np.empty((len(track.points), JOINT_COUNT, 2))
    for joint in range(JOINT_COUNT):
        for axis in range(2):
            out[:, joint, axis] = _fill_and_smooth(track.points[:, joint, axis], window)
    return out


def rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    """Centred rolling min or max."""
    padded: np.ndarray = np.pad(values, window // 2, mode="edge")
    view: np.ndarray = np.lib.stride_tricks.sliding_window_view(padded, window)[: len(values)]
    return view.min(axis=1) if reducer == "min" else view.max(axis=1)


def _settled(signal: np.ndarray, frame: int, span: int, config: DipConfig) -> bool:
    """True if back at the top, or near the top and no longer moving."""
    if signal[frame] <= config.edge:
        return True
    if signal[frame] > config.settle_depth:
        return False
    other: int = int(np.clip(frame + span, 0, len(signal) - 1))
    return abs(float(signal[other] - signal[frame])) < config.settle_move


def find_dips(
    signal: np.ndarray, gate: np.ndarray, fps: float, config: DipConfig = DEFAULT_DIPS
) -> list[Rep]:
    """Find each dip deeper than `config.min_depth` and expand it to where it settles.

    A dip only counts when `gate` is also at least `min_depth` at the bottom (a second
    body part confirming the movement) and the rep is complete within the video.
    """
    reps: list[Rep] = []
    settle: int = max(1, round(config.settle_seconds * fps))
    frame: int = 0
    total: int = len(signal)
    while frame < total:
        if signal[frame] < config.min_depth:
            frame += 1
            continue
        low: int = frame
        while low < total and signal[low] >= config.min_depth:
            low += 1
        bottom: int = frame + int(np.argmax(signal[frame:low]))
        start: int = bottom
        while start > 0 and not _settled(signal, start, -settle, config):
            start -= 1
        end: int = bottom
        while end < total - 1 and not _settled(signal, end, settle, config):
            end += 1
        complete: bool = start > 0 and end < total - 1
        plausible: bool = (
            signal[bottom] <= config.max_depth and (end - start) / fps <= config.max_seconds
        )
        if complete and plausible and gate[bottom] >= config.min_depth:
            reps.append(Rep(start=start, bottom=bottom, end=end))
        frame = end + 1
    return reps
