# Gym Posture — AI squat coach

Analyse a squat video, score form 0–100% per rep and overall, and render an
Instagram-story video (skeleton + live form meter + graph) ending on a scorecard.

## How it works
- Pose: MediaPipe Pose Landmarker (heavy, open-source Google model, 33 landmarks).
- Reps: hip drop vs a rolling standing baseline; shoulders must drop too; a rep ends
  when hips are standing or settled near the top.
- Checks (front camera), weights: Depth 30, Knees (valgus) 25, Balance (hip shift) 20,
  Tempo 15, Bar level 10. Ideas adapted from open-source MediaPipe squat analysers
  (LearnOpenCV AI Fitness Trainer, Pose-Trainer), which use side-view angles.
- All scoring is deterministic Python — no LLM.

## Run
    curl -L -o models/pose_landmarker_heavy.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
    uv run gym-posture <video.mp4> --load 120 --bodyweight 75

Outputs in `output/`: `<name>_story.mp4`, `<name>_scorecard.png`, `<name>_report.json`
(pose is cached as `<name>_pose.npz`, so re-renders take ~30 s).

## Modules
- `pose.py` — landmarks per frame · `reps.py` — signals + rep detection
- `scoring.py` — checks and scores · `render.py` — video overlay + scorecard · `cli.py`

## Done
- First video (20261103_064310.mp4, 120 kg × 4, BW 75 kg): overall 96; reps 99/98/99/87.

## Pending
- Depth from the front is an estimate; a side-view mode (hip/knee angles) would be more exact.
- Calibrate thresholds on more videos (good and bad reps).
- Optional: keep original audio; side-by-side multi-set history; the full "AI trainer" app.
