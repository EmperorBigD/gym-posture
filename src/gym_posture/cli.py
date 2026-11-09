"""Analyse a lifting video and write the story video, scorecard and JSON report.

gym-posture squat data/input/squat.mp4 --load 120 --bodyweight 75
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gym_posture.core.pose import PoseTrack, extract_pose
from gym_posture.core.scoring import FrameScores, SetScore
from gym_posture.core.signals import Rep, clean_points
from gym_posture.exercises import EXERCISES
from gym_posture.exercises.base import Exercise, Lift
from gym_posture.render.overlay import StoryRenderer
from gym_posture.render.scorecard import scorecard
from gym_posture.render.video import clip_frames, write_story

OUTPUT_DIR: Path = Path("output")
CACHE_DIR: Path = OUTPUT_DIR / "cache"


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="gym-posture", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("exercise", choices=sorted(EXERCISES))
    parser.add_argument("video", type=Path)
    parser.add_argument("--load", type=float, required=True,
                        help="weight in kg (the bar, or ONE dumbbell)")  # fmt: skip
    parser.add_argument("--bodyweight", type=float, required=True, help="your weight in kg")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _load_track(video: Path, exercise: Exercise) -> PoseTrack:
    """Pose is slow to compute, so it is cached per video and pose model."""
    cache: Path = CACHE_DIR / f"{video.stem}_{exercise.backend}.npz"
    if cache.exists():
        return PoseTrack.load(cache)
    print(f"Detecting pose on every frame with {exercise.backend} (a few minutes)...")
    track: PoseTrack = extract_pose(video, exercise.backend)
    track.save(cache)
    return track


def _report(result: SetScore) -> dict[str, object]:
    return {
        "overall": round(result.overall, 1),
        "checks": {name: round(value, 1) for name, value in result.checks.items()},
        "reps": [
            {
                "overall": round(rep.overall, 1),
                "checks": {name: round(value, 1) for name, value in rep.checks.items()},
                **{name: round(value, 3) for name, value in rep.details.items()},
            }
            for rep in result.reps
        ],
    }


def main() -> None:
    args: argparse.Namespace = _parse_args()
    exercise: Exercise = EXERCISES[args.exercise]
    video: Path = args.video
    lift: Lift = Lift(args.load, args.bodyweight, implements=exercise.implements)

    track: PoseTrack = _load_track(video, exercise)
    points: np.ndarray = clean_points(track)
    reps: list[Rep] = exercise.find_reps(points, track.fps)
    if not reps:
        raise SystemExit(f"No {exercise.slug} reps found in {video}.")
    frames: FrameScores = exercise.score_frames(points, track.fps)
    result: SetScore = exercise.score_set(
        [exercise.score_rep(rep, points, frames, track.fps) for rep in reps]
    )

    out_dir: Path = args.out / exercise.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    card: np.ndarray = scorecard(exercise, lift, result)
    card_path: Path = out_dir / f"{video.stem}_scorecard.png"
    cv2.imwrite(str(card_path), card)

    keep: list[int] = clip_frames([(r.start, r.end) for r in reps], track.fps, len(points))
    progress: np.ndarray = exercise.progress(points, track.fps)
    renderer: StoryRenderer = StoryRenderer(
        exercise, lift, points, frames, progress, result, keep, track.fps
    )
    video_path: Path = out_dir / f"{video.stem}_story.mp4"
    write_story(video, video_path, renderer, card)

    report_path: Path = out_dir / f"{video.stem}_report.json"
    report_path.write_text(json.dumps(_report(result), indent=2))
    print(f"Overall form score: {result.overall:.0f}/100 over {len(reps)} reps")
    for path in (video_path, card_path, report_path):
        print(path.resolve())
