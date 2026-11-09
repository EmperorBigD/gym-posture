"""Turn raw landmarks into clean joint signals and find squat reps."""

from dataclasses import dataclass

import numpy as np

from gym_posture.pose import PoseTrack

# MediaPipe landmark indices (the person's left / right).
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

SMOOTH_SECONDS: float = 0.15
# A rep starts / ends when the hips are within this fraction of thigh length of standing.
REP_EDGE_DEPTH: float = 0.12
# A dip only counts as a rep if the hips drop at least this fraction of thigh length.
REP_MIN_DEPTH: float = 0.6
# Window for the local "standing" reference (longer than one rep).
BASELINE_SECONDS: float = 8.0
# Near the top, the hips count as "settled" (rep over) once they move less than
# SETTLE_MOVE thigh lengths over SETTLE_SECONDS - e.g. lifter stepping to the rack.
SETTLE_DEPTH: float = 0.3
SETTLE_MOVE: float = 0.03
SETTLE_SECONDS: float = 0.4


@dataclass
class Rep:
    start: int
    bottom: int
    end: int


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
    """Return (frames, 33, 2) x/y pixel positions with gaps filled and jitter smoothed."""
    window: int = max(1, round(SMOOTH_SECONDS * track.fps))
    out: np.ndarray = np.empty((len(track.points), 33, 2))
    for joint in range(33):
        for axis in range(2):
            out[:, joint, axis] = _fill_and_smooth(track.points[:, joint, axis], window)
    return out


def _rolling(values: np.ndarray, window: int, reducer: str) -> np.ndarray:
    """Centred rolling min or max."""
    padded: np.ndarray = np.pad(values, window // 2, mode="edge")
    view: np.ndarray = np.lib.stride_tricks.sliding_window_view(padded, window)[: len(values)]
    return view.min(axis=1) if reducer == "min" else view.max(axis=1)


def depth_signal(points: np.ndarray, fps: float) -> np.ndarray:
    """How far the hips have dropped, in thigh lengths (0 = standing, ~1 = hips at knees).

    The standing reference is local (rolling over BASELINE_SECONDS) so that the lifter
    stepping back or the camera shifting does not skew it.
    """
    hip_y: np.ndarray = (points[:, L_HIP, 1] + points[:, R_HIP, 1]) / 2
    knee_y: np.ndarray = (points[:, L_KNEE, 1] + points[:, R_KNEE, 1]) / 2
    window: int = round(BASELINE_SECONDS * fps) | 1
    standing_hip: np.ndarray = _rolling(hip_y, window, "min")
    standing_thigh: np.ndarray = _rolling(knee_y - hip_y, window, "max")
    return (hip_y - standing_hip) / standing_thigh


def shoulder_drop(points: np.ndarray, fps: float) -> np.ndarray:
    """Shoulder drop from local standing height, in thigh lengths (same scale as depth)."""
    shoulder_y: np.ndarray = (points[:, L_SHOULDER, 1] + points[:, R_SHOULDER, 1]) / 2
    hip_y: np.ndarray = (points[:, L_HIP, 1] + points[:, R_HIP, 1]) / 2
    knee_y: np.ndarray = (points[:, L_KNEE, 1] + points[:, R_KNEE, 1]) / 2
    window: int = round(BASELINE_SECONDS * fps) | 1
    standing_shoulder: np.ndarray = _rolling(shoulder_y, window, "min")
    standing_thigh: np.ndarray = _rolling(knee_y - hip_y, window, "max")
    return (shoulder_y - standing_shoulder) / standing_thigh


def _settled(depth: np.ndarray, frame: int, span: int) -> bool:
    """True if the hips are standing, or near the top and no longer moving."""
    if depth[frame] <= REP_EDGE_DEPTH:
        return True
    if depth[frame] > SETTLE_DEPTH:
        return False
    other: int = int(np.clip(frame + span, 0, len(depth) - 1))
    return abs(float(depth[other] - depth[frame])) < SETTLE_MOVE


def find_reps(depth: np.ndarray, shoulders: np.ndarray, fps: float) -> list[Rep]:
    """Find each dip below REP_MIN_DEPTH and expand it to where the lifter is standing.

    A dip only counts when the shoulders drop with the hips (a real squat), which rules
    out walking towards or away from the camera.
    """
    reps: list[Rep] = []
    settle: int = max(1, round(SETTLE_SECONDS * fps))
    frame: int = 0
    total: int = len(depth)
    while frame < total:
        if depth[frame] < REP_MIN_DEPTH:
            frame += 1
            continue
        low: int = frame
        while low < total and depth[low] >= REP_MIN_DEPTH:
            low += 1
        bottom: int = frame + int(np.argmax(depth[frame:low]))
        start: int = bottom
        while start > 0 and not _settled(depth, start, -settle):
            start -= 1
        end: int = bottom
        while end < total - 1 and not _settled(depth, end, settle):
            end += 1
        complete: bool = start > 0 and end < total - 1
        if complete and shoulders[bottom] >= REP_MIN_DEPTH:
            reps.append(Rep(start=start, bottom=bottom, end=end))
        frame = end + 1
    return reps
