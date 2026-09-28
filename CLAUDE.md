# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Context

Research prototype for a paper (student self-analysis of engagement in online classes). Everything runs locally on the learner's machine; video never leaves the device. `README.md` documents the math (formulas, thresholds, window lengths). **The code is the source of truth**: when you change a formula, weight, threshold, or window length in `app/attention_rules.py`, update the matching formula in `README.md` in the same change.

## Environment & commands

- Python env: conda env `demo` (Python 3.10) at `/Users/fang/Documents/Code/miniconda3/envs/demo/bin/python` (also set in `.vscode/settings.json`). The base `python3` on PATH is 3.13 and lacks the deps.
- Pinned in that env: `mediapipe 0.9.2.1` (code uses the legacy `mp.solutions.face_mesh` API), `ultralytics 8.4.19`.
- Run: `python main.py` **from the repo root**. `ExpressionClassifier` loads `weights/best.pt` via a cwd-relative path.
- Weights are not in git (`weights/*.pt`, `*.dat`, `*.onnx` are ignored). `weights/best.pt` must exist locally; if loading fails, the classifier sets `model = None` and silently returns no probabilities.
- No test suite or linter. Verify with `python -m py_compile <files>`. For logic checks, write a throwaway script that imports `app.*` directly and runs without the GUI or a camera, for example:
  - Feed a still image to `FaceMeshInference.process_frame`.
  - Patch `attention_rules.time.time` to drive `AttentionAnalyzer.process_frame` through simulated seconds.
  - Face images: `ultralytics/assets/zidane.jpg` works if cropped to a close-up face (FaceMesh misses small faces).

## Per-frame pipeline (spans `camera_thread.py` → `mediapipe_inference.py` → `yolo_inference.py` → `attention_rules.py`)

1. **MediaPipe first**: `FaceMeshInference.process_frame(frame)` returns `ear, mar, pitch, yaw, roll, bbox, display_frame`.
   - The mesh overlay is drawn only on `display_frame`. The original `frame` stays clean because it is what gets cropped for YOLO. Never draw on `frame` before the crop.
   - `bbox` is a **square** (longest side × 1.4, shifted inward at image edges) with EMA smoothing. It is square because Ultralytics classify preprocessing is "resize short side → center-crop 224", which would cut off part of a non-square face crop.
   - Pitch is sign-flipped so that looking down is negative.
2. **YOLO second**: `frame[bbox]` goes to `ExpressionClassifier.process_face`, which returns a 7-class probability vector.
3. **Calibration** (in `CameraThread.run`): for the first 3 s, the median pitch/yaw becomes the per-user baseline. After that, the analyzer only receives `delta_pitch` / `delta_yaw` relative to that baseline.
4. **`AttentionAnalyzer.process_frame`** uses three time-based deques:
   - micro, 3 s: smooths the expression probabilities.
   - perclos, 8 s, with a 7.5 s warm-up before the fatigue index is computed: EAR/MAR/pitch → fatigue index.
   - macro, 60 s: one cognitive-state label per frame → score.
   After scoring, spatial rules override the score and status:
   - head down longer than 15 s;
   - face lost: 5 s grace period if the student was looking down beforehand, otherwise cubic decay over 120 s, then `ABSENT`.
   It returns `(score, status_text, alert_data)`.
5. Results reach the UI only via the `pyqtSignal`s on `CameraThread` (pixmap, score, status text, alert).

## Cross-file couplings to watch

- **Class order**: `AttentionAnalyzer.EMOTION_IDX` hardcodes the 7-class order of the current `best.pt` (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise). Expression processing is gated on `len(all_probs) == 7`. The paper's planned model is a transfer-learned **4-class** model (Understand/Doubt/Neutral/Disgust). Swapping it in would silently skip all expression logic until the 7→4 mapping in `attention_rules.py` is replaced.
- **Status-text colors**: `MonitorInterface.update_status_text` picks the label color by ordered, case-sensitive substring match on the status text. The order is: blue `CALIBRATING` → red `PHONE`/`HEAD DOWN`/`ABSENT`/`DISTRACTED`/`ABNORMAL` → green `Focus`/`Active` → orange for everything else. Red must be checked before green, because `Focus: DISTRACTED` also contains `Focus`. Renaming status strings in `attention_rules.py` or `camera_thread.py` changes the UI colors. `show_async_alert` uses error styling only when the alert title contains `CRITICAL`.
- **Alert rate limiting**: all alert types share one `last_alert_time` cooldown.
- **Thread lifetime**: `CameraThread.stop()` sets the run flag and calls `wait()`. `MainWindow.closeEvent` also stops the thread, so keep `camera_thread` as the attribute name on `MonitorInterface`.
- **Report page**: `ReportInterface` is a placeholder. Session data is not persisted or passed to it yet (`load_session_data` / `render_charts` are stubs).
