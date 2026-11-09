"""Pick the frames for the story, overlay them and encode an H.264 MP4 with ffmpeg."""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from gym_posture.render.overlay import StoryRenderer
from gym_posture.render.theme import H, W

PAD_BEFORE_S: float = 1.0
PAD_AFTER_S: float = 1.0
CARD_SECONDS: float = 5.0


def clip_frames(reps_bounds: list[tuple[int, int]], fps: float, total: int) -> list[int]:
    """Source frame indices to keep: each rep plus padding, overlapping windows merged."""
    keep: list[int] = []
    for start, end in reps_bounds:
        a: int = max(0, start - round(PAD_BEFORE_S * fps))
        b: int = min(total - 1, end + round(PAD_AFTER_S * fps))
        keep.extend(range(max(a, keep[-1] + 1 if keep else 0), b + 1))
    return keep


def write_story(
    video_path: Path, out_path: Path, renderer: StoryRenderer, card: np.ndarray
) -> None:
    """Decode the source once, overlay the kept frames, append the scorecard, encode."""
    command: list[str] = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{renderer.fps:.4f}",
        "-i", "-",
        "-c:v", "libopenh264", "-b:v", "10M", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out_path),
    ]  # fmt: skip
    encoder: subprocess.Popen[bytes] = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    wanted: dict[int, int] = {source: position for position, source in enumerate(renderer.keep)}
    last: int = renderer.keep[-1]
    source: int = 0
    while source <= last:
        ok: bool
        frame: np.ndarray
        ok, frame = capture.read()
        if not ok:
            break
        if source in wanted:
            frame = cv2.resize(frame, (W, H))
            encoder.stdin.write(renderer.overlay(frame, wanted[source]).tobytes())
        source += 1
    capture.release()
    for _ in range(round(CARD_SECONDS * renderer.fps)):
        encoder.stdin.write(card.tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("ffmpeg failed to encode the story video")
