"""Extract per-frame body landmarks from a video with MediaPipe Pose Landmarker."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarkerOptions,
    PoseLandmarkerResult,
)

# Frames are downscaled to this height before pose detection (speed vs accuracy).
DETECT_HEIGHT: int = 960


@dataclass
class PoseTrack:
    """Landmarks for the lifter across the whole video.

    `points` has shape (frames, 33, 4): x, y (pixels in the original frame),
    z (MediaPipe relative depth) and visibility. Missing frames are NaN.
    """

    points: np.ndarray
    fps: float
    width: int
    height: int

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, points=self.points, meta=np.array([self.fps, self.width, self.height])
        )

    @classmethod
    def load(cls, path: Path) -> "PoseTrack":
        data: np.lib.npyio.NpzFile = np.load(path)
        meta: np.ndarray = data["meta"]
        return cls(
            points=data["points"], fps=float(meta[0]), width=int(meta[1]), height=int(meta[2])
        )


def _pick_lifter(poses: list[list[NormalizedLandmark]]) -> int:
    """Pick the pose most likely to be the lifter: the tallest, most central one."""
    best_index: int = 0
    best_score: float = -1.0
    for index, pose in enumerate(poses):
        xs: list[float] = [lm.x for lm in pose if lm.x is not None]
        ys: list[float] = [lm.y for lm in pose if lm.y is not None]
        height: float = max(ys) - min(ys)
        centre_x: float = float(np.mean(xs))
        score: float = height - abs(centre_x - 0.5)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def extract_pose(video_path: Path, model_path: Path) -> PoseTrack:
    """Run pose detection on every frame and return the lifter's landmarks."""
    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    fps: float = capture.get(cv2.CAP_PROP_FPS)
    options: PoseLandmarkerOptions = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3,
    )
    frames: list[np.ndarray] = []
    width: int = 0
    height: int = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index: int = 0
        while True:
            ok: bool
            frame: np.ndarray
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale: float = DETECT_HEIGHT / height
            small: np.ndarray = cv2.resize(frame, (round(width * scale), DETECT_HEIGHT))
            rgb: np.ndarray = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image: mp.Image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms: int = round(frame_index * 1000 / fps)
            result: PoseLandmarkerResult = landmarker.detect_for_video(image, timestamp_ms)
            row: np.ndarray = np.full((33, 4), np.nan)
            if result.pose_landmarks:
                pose: list[NormalizedLandmark] = result.pose_landmarks[
                    _pick_lifter(result.pose_landmarks)
                ]
                for joint, lm in enumerate(pose):
                    row[joint] = (
                        (lm.x or 0.0) * width,
                        (lm.y or 0.0) * height,
                        lm.z or 0.0,
                        lm.visibility or 0.0,
                    )
            frames.append(row)
            frame_index += 1
    capture.release()
    return PoseTrack(points=np.stack(frames), fps=fps, width=width, height=height)
