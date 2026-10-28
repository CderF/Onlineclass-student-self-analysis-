# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Context

Research prototype for a paper (student self-analysis of engagement in online classes). Everything runs locally on the learner's machine; video never leaves the device. `README.md` documents the math (formulas, thresholds, window lengths). **The code is the source of truth**: when you change a formula, weight, threshold, or window length in `app/attention_rules.py`, update the matching formula in `README.md` in the same change.

## Environment & commands

- Python env: conda env `demo` (Python 3.10) at `/Users/fang/Documents/Code/miniconda3/envs/demo/bin/python` (also set in `.vscode/settings.json`). The base `python3` on PATH is 3.13 and lacks the deps.
- Pinned in that env: `mediapipe 0.9.2.1` (code uses the legacy `mp.solutions.face_mesh` API), `ultralytics 8.4.19`.
- Report page stack (`docs/adr/0001-qt-echarts-report.md`): `PyQtWebEngine 5.15.7` with `PyQtWebEngine-Qt5==5.15.17`, which must match `PyQt5-Qt5`. ECharts 6.1.0 is vendored at `app/view/web/vendor/echarts/` and loaded from disk, never from a CDN. Import `PyQt5.QtWebEngineWidgets` before the `QApplication` is created.
- The UI language is Japanese. The font is `Hiragino Sans`, set both via `app.setFont` and via qfluentwidgets `setFontFamilies` in `main.py`. PingFang SC would render Japanese kanji with Chinese glyph forms. Console `print`s stay in Chinese because they are developer-facing.
- Run: `python main.py` **from the repo root**. `ExpressionClassifier` loads `weights/best.pt` via a cwd-relative path.
- Weights are not in git (`weights/*.pt`, `*.dat`, `*.onnx` are ignored). `weights/best.pt` must exist locally; if loading fails, the classifier sets `model = None` and silently returns no probabilities.
- Every issue ships with tests, starting from #4: pytest in `tests/`, plus dedicated verification scripts in `test-scripts/` (`verify_issue<N>_*.py`; see its README) for what pytest can't simulate well: real crashes, concurrency, long runs, performance.
- Tests: run `python -m pytest` from the repo root (`tests/`, about 80 s; the `CameraThread` flow tests account for most of it and are skipped without `weights/best.pt`). `conftest.py` sets `QT_QPA_PLATFORM=offscreen`, puts the repo root on `sys.path`, and `chdir`s there. There is no linter; `python -m py_compile <files>` is the syntax check. Patterns used in the tests:
  - Patch `attention_rules.time.time` with a fake clock to drive `AttentionAnalyzer.process_frame` through simulated seconds (`tests/test_attention_rules.py`). `attention_rules.time` is the global `time` module, so this patches `time.time` process-wide. Use `monkeypatch`, or save the real function before patching and restore it afterwards.
  - Patch `camera_thread.cv2.VideoCapture` with a fake camera to run `CameraThread` headless (`tests/test_monitor_interface.py`). That test is skipped when `weights/best.pt` is missing.
  - Face images: `ultralytics/assets/zidane.jpg` works if cropped to a close-up face (`[0:400, 800:1200]`; FaceMesh misses small faces). Use the shared `fake_camera` fixture in `conftest.py`; setting `show_face=False` simulates leaving the seat.
  - `MainWindow` (a `FluentWindow`) **segfaults under `QT_QPA_PLATFORM=offscreen` on macOS**, because qframelesswindow calls Cocoa directly. In pytest, test `MonitorInterface` directly, or call `MainWindow.closeEvent` on a stub. Real-window checks go in `test-scripts/` with the native platform.
  - `QTest.keyClicks` cannot type non-ASCII text. Set Japanese text directly on the widget.

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
   It returns an `AnalysisResult` dataclass: score, `status_text` (display only), alert, `cognitive_state` (`None` when the frame has no expression result), `fatigue_index`, `spatial_state` (one of 8 codes, see `GLOSSARY.md`), deltas. Consumers other than the status label (recording, reports) must read the structured fields, never parse `status_text`.
5. Results reach the UI only via the `pyqtSignal`s on `CameraThread`: pixmap, score, `(status_text, tone)`, alert.
6. **Pause/resume** (`CameraThread.pause` / `resume`): pausing releases the camera. Resuming reopens it, keeps the calibration baseline (`is_calibrated`), and replaces the `AttentionAnalyzer`, so all windows and timers start empty.

## Session recording (`docs/adr/0002-session-storage-schema.md`)

- `app/session_store.py` holds the schema, `connect()`, and the session status helpers. The DB is `data/self_analysis.db` under the repo root (gitignored). `main.py` calls `mark_interrupted_sessions` on startup.
- `app/session_recorder.py`: `SessionRecorder` takes one `AnalysisResult` per frame and writes one `samples` row per second of the **listening timeline** (`t`; pauses excluded, calibration excluded). It also writes `events` (frame-precise spatial-state intervals and alerts) and `pauses`. It commits once per completed second.
- Create one `SessionRecorder` per capture segment, inside `CameraThread.run()`: sqlite connections are thread-bound. On `pause(wall)`, it records a pause row. On the next construction it resumes from the next whole second after the pause.
- Spatial-state codes and cognitive-state labels are persisted. Renaming them requires a data migration.
- Wiring (#4), all in `CameraThread`:
  - The session row is created when the camera first opens.
  - The recorder starts at calibration end, or at resume.
  - When `run()` exits, it either `pause()`s or `close()`s the recorder.
  - `stop()` then marks the session `completed` from the caller's thread, and is safe to call while paused. A session with no `samples` rows is deleted instead (`finish_session` returns `False`) and `session_id` is reset, so no course-name dialog appears. `mark_interrupted_sessions` deletes empty crash leftovers the same way.
  - `MainWindow.closeEvent` always calls `stop()`, which also completes a paused session.
  - Recording errors go through `_guard_record`: monitoring continues and a single 記録エラー alert is shown.
- After Stop, `MonitorInterface._after_stop` asks for the course name. In tests, patch `_ask_course_name`. `MonitorInterface.db_path` / `CameraThread(db_path=)` override the DB, and `tests/conftest.py` redirects `DEFAULT_DB_PATH` to a temp file for every test.

## Report page (#5, `app/view/report_interface.py` + `app/view/web/`)

- The whole report tab is one `QWebEngineView` showing `app/view/web/report.html` (Japanese, tabs 授業ごと / 全体の推移). Course selection and later record management (#10) live in the page, not in Qt widgets.
- Data path: `report.js` calls slots on `ReportBridge` over `QWebChannel` (`qrc:///qtwebchannel/qwebchannel.js`). Each slot opens its own connection in the UI thread, reads via `app/report_data.py` (`list_sessions`, `load_session`), and returns `{"data": ..., "error": ...}` as a JSON string.
- `ReportInterface.showEvent` emits `data_changed_signal`; the page then reloads the list. If a new record appeared it selects the newest one, otherwise it keeps the current selection. `body[data-state]` is `loading` / `ready` / `empty` / `error`.
- The page uses its own off-the-record `QWebEngineProfile` with `LocalOnlyInterceptor`, which blocks every non-`file`/`qrc`/`data` request (`interceptor.blocked` records them). `ReportPage.js_errors` collects console errors.
- The profile is a child of the page so it is released after the page. A `ReportInterface` still alive at interpreter exit (e.g. a pending `deleteLater` in tests) crashes WebEngine, so tests destroy it with `sip.delete`. Pages may be polled before `<body>` exists: use `document.body && ...`.

## Cross-file couplings to watch

- **Class order**: `AttentionAnalyzer.EMOTION_IDX` hardcodes the 7-class order of the current `best.pt` (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise). Expression processing is gated on `len(all_probs) == 7`. The paper's planned model is a transfer-learned **4-class** model (Understand/Doubt/Neutral/Disgust). Swapping it in would silently skip all expression logic until the 7→4 mapping in `attention_rules.py` is replaced.
- **Status-text colors**: the color comes from a tone (`info` / `success` / `warning` / `danger` / `muted`) sent alongside the text, never from the text itself. `SPATIAL_STATE_TONES` in `camera_thread.py` maps each spatial state to a tone, and `MonitorInterface.STATUS_TONE_STYLES` maps each tone to a color. Status strings are display-only Japanese and can be reworded freely. `show_async_alert` uses error styling only when the alert title contains `CRITICAL`.
- **Alert rate limiting**: all alert types share one `last_alert_time` cooldown.
- **Thread lifetime**: `CameraThread.stop()` sets the run flag and calls `wait()`. `MainWindow.closeEvent` also stops the thread, so keep `camera_thread` as the attribute name on `MonitorInterface`.
- **Report page**: the page reads only through `ReportBridge`'s JSON (`app/report_data.py`). Fields the page uses are named after the DB columns, so renaming a column means updating `report.js` as well.
