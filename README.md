# Gym Posture — AI gym coach

Scores your lifting form from a phone video and renders an Instagram story:
your set with a colour-coded skeleton, a live 0–100% form meter and graph, a
pop-up score after each rep, and a closing "AI coach" scorecard.

| Exercise | Command | Camera | Pose model |
|---|---|---|---|
| Back squat | `squat` | in front, at hip height | MediaPipe Pose (Google) |
| Incline dumbbell press | `press` | at your feet, looking up the bench | RTMPose (OpenMMLab) |

All scoring is deterministic Python rules on the joint positions — no LLM calls.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` (with `libopenh264`).

```bash
uv sync
# MediaPipe model for squats (RTMPose models download themselves to ~/.cache/rtmlib)
curl -L -o models/pose_landmarker_heavy.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
```

## Usage

Put videos in `data/input/`, then:

```bash
uv run gym-posture squat data/input/squat.mp4 --load 120 --bodyweight 75
uv run gym-posture press data/input/press.mp4 --load 30 --bodyweight 75   # --load = one dumbbell
```

Results go to `output/<exercise>/`:

- `<video>_story.mp4` — 1080×1920 H.264, ready to post
- `<video>_scorecard.png` — the summary card on its own
- `<video>_report.json` — every score and raw measurement per rep

Pose detection is cached in `output/cache/`, so re-rendering takes seconds.

## Project layout

```
data/input/            your videos (git-ignored)
models/                MediaPipe model (git-ignored)
output/                results + pose cache (git-ignored)
src/gym_posture/
├── cli.py             command line
├── core/
│   ├── pose.py        pose models → 17 COCO joints per frame
│   ├── signals.py     smoothing, rolling baselines, rep finder
│   └── scoring.py     score containers, ramp, grades
├── exercises/
│   ├── base.py        Exercise interface + shared scoring flow
│   ├── squat.py
│   └── dumbbell_press.py
└── render/
    ├── theme.py       colours, fonts
    ├── overlay.py     skeleton, live meter, graph, pop-ups
    ├── scorecard.py   end card
    └── video.py       frame selection + ffmpeg encoding
tests/
```

To add an exercise: subclass `Exercise` in `exercises/`, register it in
`exercises/__init__.py`, and it gets the CLI command, video and scorecard for free.

## Development

```bash
uv run ruff format . && uv run ruff check . && uv run pyright && uv run pytest
```
