"""Extract per-frame body joints from a video.

Two open-source pose models sit behind one output format (17 COCO joints):

- MediaPipe Pose Landmarker (Google): fast, good for an upright lifter facing the camera.
- RTMPose (OpenMMLab, via rtmlib): more robust for lying / foreshortened poses such as a
  bench press filmed from the feet.
"""

from dataclasses import dataclass
from enum import StrEnum
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

# COCO-17 joint indices (the person's left / right).
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
JOINT_COUNT: int = 17

# MediaPipe's 33 landmarks -> COCO-17 order.
MEDIAPIPE_TO_COCO: tuple[int, ...] = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)

MEDIAPIPE_MODEL: Path = Path("models/pose_landmarker_heavy.task")
# Frames are downscaled to this height before MediaPipe detection (speed vs accuracy).
MEDIAPIPE_HEIGHT: int = 960

RTMLIB_BASE: str = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
RTM_DETECTOR: str = RTMLIB_BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
RTM_POSE: str = RTMLIB_BASE + "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip"
# The person detector is the slow part; the lifter barely moves, so re-detect every N frames.
RTM_DETECT_EVERY: int = 10


class Backend(StrEnum):
    MEDIAPIPE = "mediapipe"
    RTMPOSE = "rtmpose"


@dataclass
class PoseTrack:
    """Joints for the lifter across the whole video.

    `points` has shape (frames, 17, 3): x, y in pixels of the original frame and a
    confidence score. Frames where nobody was found are NaN.
    """

    points: np.ndarray
    fps: float
    width: int
    height: int

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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


def _lifter_score(xs: np.ndarray, ys: np.ndarray, width: int, height: int) -> float:
    """Higher for the tallest, most central person: that is the lifter, not passers-by."""
    tall: float = float(np.nanmax(ys) - np.nanmin(ys)) / height
    off_centre: float = abs(float(np.nanmean(xs)) / width - 0.5)
    return tall - off_centre


def _mediapipe_row(poses: list[list[NormalizedLandmark]], width: int, height: int) -> np.ndarray:
    row: np.ndarray = np.full((JOINT_COUNT, 3), np.nan)
    if not poses:
        return row
    candidates: list[np.ndarray] = [
        np.array([[lm.x or 0.0, lm.y or 0.0, lm.visibility or 0.0] for lm in pose])
        for pose in poses
    ]
    best: np.ndarray = max(
        candidates, key=lambda c: _lifter_score(c[:, 0] * width, c[:, 1] * height, width, height)
    )
    mapped: np.ndarray = best[list(MEDIAPIPE_TO_COCO)]
    row[:, 0] = mapped[:, 0] * width
    row[:, 1] = mapped[:, 1] * height
    row[:, 2] = mapped[:, 2]
    return row


def _extract_mediapipe(capture: cv2.VideoCapture, fps: float) -> list[np.ndarray]:
    options: PoseLandmarkerOptions = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MEDIAPIPE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=3,
    )
    rows: list[np.ndarray] = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok: bool
            frame: np.ndarray
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            scale: float = MEDIAPIPE_HEIGHT / height
            small: np.ndarray = cv2.resize(frame, (round(width * scale), MEDIAPIPE_HEIGHT))
            image: mp.Image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            )
            timestamp_ms: int = round(len(rows) * 1000 / fps)
            result: PoseLandmarkerResult = landmarker.detect_for_video(image, timestamp_ms)
            rows.append(_mediapipe_row(result.pose_landmarks, width, height))
    return rows


def _extract_rtmpose(capture: cv2.VideoCapture) -> list[np.ndarray]:
    from rtmlib import YOLOX, RTMPose  # imported lazily: only needed for this backend

    detector: YOLOX = YOLOX(RTM_DETECTOR, model_input_size=(640, 640))
    pose_model: RTMPose = RTMPose(RTM_POSE, model_input_size=(288, 384))
    rows: list[np.ndarray] = []
    boxes: np.ndarray = np.empty((0, 4))
    while True:
        ok: bool
        frame: np.ndarray
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if len(rows) % RTM_DETECT_EVERY == 0 or len(boxes) == 0:
            boxes = np.asarray(detector(frame))
        row: np.ndarray = np.full((JOINT_COUNT, 3), np.nan)
        if len(boxes):
            keypoints: np.ndarray
            scores: np.ndarray
            keypoints, scores = pose_model(frame, bboxes=boxes.tolist())
            best: int = max(
                range(len(keypoints)),
                key=lambda i: _lifter_score(keypoints[i][:, 0], keypoints[i][:, 1], width, height),
            )
            row[:, :2] = keypoints[best]
            row[:, 2] = scores[best]
        rows.append(row)
    return rows


def extract_pose(video_path: Path, backend: Backend) -> PoseTrack:
    """Run pose detection on every frame and return the lifter's joints."""
    capture: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    fps: float = capture.get(cv2.CAP_PROP_FPS)
    # OpenCV applies the phone's rotation flag, so a portrait video reads as portrait.
    width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows: list[np.ndarray] = (
        _extract_mediapipe(capture, fps)
        if backend is Backend.MEDIAPIPE
        else _extract_rtmpose(capture)
    )
    capture.release()
    return PoseTrack(points=np.stack(rows), fps=fps, width=width, height=height)
