# Gym Posture — AI gym coach

Analyse a lifting video, score form 0–100% per rep and overall, and render an
Instagram-story video (skeleton + live form meter + graph) ending on a scorecard.
Usage and layout: see README.md.

## How it works
- Pose (`core/pose.py`): two open-source models, one 17-joint (COCO) output.
  - MediaPipe Pose (heavy) for the squat (lifter upright, facing the camera).
  - RTMPose-x + YOLOX-m via rtmlib for the press: MediaPipe lost the arms when
    filmed from the feet. Person detection runs every 10th frame (~0.2 s/frame on CPU).
- Reps (`core/signals.py`): a 0 (top) → 1 (bottom) signal per exercise, a rolling
  local baseline, and a dip finder gated by a second body part; dips that are too
  deep / long (getting on or off the bench) are rejected.
- Exercises (`exercises/`): each declares its checks, weights, cues and skeleton.
  - Squat (front view): Depth 30, Knees 25, Balance 20, Tempo 15, Bar level 10.
  - Incline DB press (from the feet): Range 25, Lockout 25, Symmetry 25, Wrists 15, Tempo 10.
- All scoring is deterministic Python — no LLM.

## Decisions
- Squat depth from the front is an estimate (parallel still shows a ~0.1-thigh gap).
- Press symmetry scores only lag between the arms; a steady height offset is ignored
  because the low, tilted camera can cause it. Measuring "up" against the shoulder
  line was tried and made it worse.
- Press elbow flare is not scored: the torso is too foreshortened from the feet.
- The first press off the chest has no lowering phase, so it is not scored as a rep.

## Done
- Squat, 20261103_064310.mp4 (120 kg × 4, BW 75 kg): overall 96; reps 99/98/99/87.
- Incline DB press, 20261009_074023.mp4 (2×30 kg × 3, BW 75 kg): overall 96; reps 99/95/93.
- Restructured into core/ exercises/ render/; videos in data/input/, results in output/<exercise>/.

## Pending
- Calibrate thresholds on more videos (good and bad reps) for both lifts.
- Side / 45° camera mode for squat depth and press elbow flare.
- Optional: keep original audio; history across sessions; the full "AI trainer" app.
