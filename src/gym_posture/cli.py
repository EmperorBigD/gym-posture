"""Command line: analyse a squat video and write the story video + scorecard."""

import argparse
import json
from pathlib import Path

import cv2

from gym_posture.pose import PoseTrack, extract_pose
from gym_posture.render import LiftInfo, StoryRenderer, clip_frames, scorecard, write_story
from gym_posture.reps import clean_points, depth_signal, find_reps, shoulder_drop
from gym_posture.scoring import score_frames, score_rep, score_set

DEFAULT_MODEL: Path = Path("models/pose_landmarker_heavy.task")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--load", type=float, required=True, help="bar weight in kg")
    parser.add_argument("--bodyweight", type=float, required=True, help="your weight in kg")
    parser.add_argument("--out", type=Path, default=Path("output"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args: argparse.Namespace = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stem: str = Path(args.video).stem
    cache: Path = out_dir / f"{stem}_pose.npz"
    if cache.exists():
        track: PoseTrack = PoseTrack.load(cache)
    else:
        print("Detecting pose on every frame (a few minutes)...")
        track = extract_pose(args.video, args.model)
        track.save(cache)

    points = clean_points(track)
    depth = depth_signal(points, track.fps)
    reps = find_reps(depth, shoulder_drop(points, track.fps), track.fps)
    if not reps:
        raise SystemExit("No squat reps found in this video.")
    frames = score_frames(points)
    result = score_set([score_rep(rep, points, frames, track.fps) for rep in reps])

    lift: LiftInfo = LiftInfo("BACK SQUAT", args.load, args.bodyweight)
    card = scorecard(result, lift)
    card_path: Path = out_dir / f"{stem}_scorecard.png"
    cv2.imwrite(str(card_path), card)

    keep: list[int] = clip_frames([(r.start, r.end) for r in reps], track.fps, len(points))
    renderer: StoryRenderer = StoryRenderer(points, frames, depth, result, keep, track.fps, lift)
    video_path: Path = out_dir / f"{stem}_story.mp4"
    write_story(args.video, video_path, renderer, card)

    report: dict[str, object] = {
        "overall": round(result.overall, 1),
        "checks": {name: round(value, 1) for name, value in result.checks.items()},
        "reps": [
            {
                "overall": round(rep.overall, 1),
                "checks": {name: round(value, 1) for name, value in rep.checks.items()},
                "depth_gap_thigh": round(rep.depth_gap, 3),
                "descent_s": round(rep.descent_s, 2),
                "ascent_s": round(rep.ascent_s, 2),
            }
            for rep in result.reps
        ],
    }
    report_path: Path = out_dir / f"{stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Overall form score: {result.overall:.0f}/100 over {len(reps)} reps")
    for path in (video_path, card_path, report_path):
        print(path.resolve())
