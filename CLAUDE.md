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
- Weights are not in git (`weights/*.pt`, `*.dat`, `*.onnx` are ignored). `weights/best.pt` must exist locally. If loading fails, the classifier sets `model = None` and returns no probabilities; `CameraThread` sends a モデルエラー message on `error_signal` and keeps monitoring without expressions. `MonitorInterface.show_persistent_alert` shows it as an InfoBar that stays until the user closes it.
- Every issue ships with tests, starting from #4: pytest in `tests/`, plus dedicated verification scripts in `test-scripts/` (`verify_issue<N>_*.py`; see its README) for what pytest can't simulate well: real crashes, concurrency, long runs, performance.
- Tests: run `python -m pytest` from the repo root (`tests/`, about 120 s; the `CameraThread` flow tests account for most of it and are skipped without `weights/best.pt`). `conftest.py` sets `QT_QPA_PLATFORM=offscreen`, puts the repo root on `sys.path`, and `chdir`s there. There is no linter; `python -m py_compile <files>` is the syntax check. Patterns used in the tests:
  - Patch `attention_rules.time.time` with a fake clock to drive `AttentionAnalyzer.process_frame` through simulated seconds (`tests/test_attention_rules.py`). `attention_rules.time` is the global `time` module, so this patches `time.time` process-wide. Use `monkeypatch`, or save the real function before patching and restore it afterwards.
  - Patch `camera_thread.cv2.VideoCapture` with a fake camera to run `CameraThread` headless (`tests/test_monitor_interface.py`). That test is skipped when `weights/best.pt` is missing.
  - Face images: `ultralytics/assets/zidane.jpg` works if cropped to a close-up face (`[0:400, 800:1200]`; FaceMesh misses small faces). Use the shared `fake_camera` fixture in `conftest.py`; setting `show_face=False` simulates leaving the seat.
  - `MainWindow` (a `FluentWindow`) **segfaults under `QT_QPA_PLATFORM=offscreen` on macOS**, because qframelesswindow calls Cocoa directly. In pytest, test `MonitorInterface` directly, or call `MainWindow.closeEvent` on a stub. Real-window checks go in `test-scripts/` with the native platform.
  - `QTest.keyClicks` cannot type non-ASCII text. Set Japanese text directly on the widget.
  - `w.grab()` repaints widgets in software and does not show how the GPU-composited web view actually stacks with other widgets on screen. To check what is really visible, capture the screen with `app.primaryScreen().grabWindow(0, ...)`; it works for the app's own windows without the screen-recording permission (`test-scripts/verify_issue10_record_management.py`).
  - Mouse input to the report page inside `MainWindow`: `QTest.mouseMove` moves the OS cursor and never reaches the page. Send `QMouseEvent` / `QWheelEvent` to `webView.focusProxy()` with `QApplication.sendEvent` (`test-scripts/verify_issue6_session_chart.py`).

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
- Create one `SessionRecorder` per capture segment, inside `CameraThread.run()`: sqlite connections are thread-bound.
  - On `pause(wall)` it records a pause row whose `t` is the **last frame's** listening time. A segment with no frames writes no row.
  - The next recorder resumes at `floor(t) + 1`, so the timeline is continuous by construction (#15).
- Deleted data must not stay recoverable in the files (#17). `connect()` sets `PRAGMA secure_delete = ON` explicitly, because the default depends on how SQLite was built. `delete_session` then runs `truncate_wal` (`wal_checkpoint(TRUNCATE)`), because the WAL still holds the old pages from when the data was written. If another connection still holds an older read snapshot, the truncate cannot finish; the WAL is then checkpointed and removed when the last connection closes.
- Spatial-state codes and cognitive-state labels are persisted. Renaming them requires a data migration.
- Wiring (#4), all in `CameraThread`:
  - Models (`FaceMeshInference`, `ExpressionClassifier`) are built at the start of `run()`, on the background thread, by `_load_models` (#12). They are built once per `CameraThread`, so pause/resume reuses them; each Start builds them again, in about 40 ms. If FaceMesh fails, `run()` shows an alert and returns without opening the camera. Stopping while the models load opens no camera.
  - The session row is created when the camera first opens.
  - The recorder starts at calibration end, or on the first processed frame after a resume (camera warm-up counts as pause time).
  - A frame read after a pause or stop was requested is discarded.
  - When `run()` exits, it either `pause()`s or `close()`s the recorder.
  - `stop()` then marks the session `completed` from the caller's thread, and is safe to call while paused. A session with no `samples` rows is deleted instead (`finish_session` returns `False`) and `session_id` is reset, so no course-name dialog appears. `mark_interrupted_sessions` deletes empty crash leftovers the same way.
  - `MainWindow.closeEvent` always calls `stop()`, which also completes a paused session.
  - Recording errors go through `_guard_record`: monitoring continues and a single 記録エラー alert is shown.
- After Stop, `MonitorInterface._after_stop` asks for the course name. In tests, patch `_ask_course_name`. `MonitorInterface.db_path` / `CameraThread(db_path=)` override the DB, and `tests/conftest.py` redirects `DEFAULT_DB_PATH` to a temp file for every test.

## Report page (#5, `app/view/report_interface.py` + `app/view/web/`)

- The whole report tab is one `QWebEngineView` showing `app/view/web/report.html` (Japanese, tabs 授業ごと / 全体の推移 / 記録の管理). Course selection and record management live in the page, not in Qt widgets.
- Data path (#16): `report.js` `call()` invokes the single slot `ReportBridge.request(id, method, argsJson)` over `QWebChannel` (`qrc:///qtwebchannel/qwebchannel.js`). The result, a JSON string `{"data": ..., "error": ...}`, comes back through `response_signal(id, json)`.
  - Reads (`list_sessions` / `load_session` / `summarize_sessions`, via `app/report_data.py`) run on the bridge's `QThreadPool`, each task with its own connection, so the UI thread is not blocked.
  - Rename and delete run on the UI thread, because they open Qt dialogs.
  - CSV export picks the folder on the UI thread, then writes on the pool. `export_started_signal` fires when writing starts. Only one export runs at a time; while it runs, the page disables every management button.
  - The bridge methods can still be called synchronously from Python, which is what the bridge-level tests do.
  - A read that takes longer than 0.3 s shows 読み込み中… (`body[data-loading]`). Responses that arrive out of order are dropped by the existing `refreshSeq` / `summarySeq` checks.
  - **Call `ReportInterface.shutdown()` before the report page is destroyed.** `MainWindow.closeEvent` and the `open_report` fixture already do this. The C++ destructor waits for the pool while holding the GIL, so a running task can never finish, and the two deadlock. `waitForDone` releases the GIL.
- `ReportInterface.showEvent` emits `data_changed_signal`; the page then reloads the list. If a new record appeared it selects the newest one, otherwise it keeps the current selection. `body[data-state]` is `loading` / `ready` / `empty` / `error`.
- The page uses its own off-the-record `QWebEngineProfile` with `LocalOnlyInterceptor`, which blocks every non-`file`/`qrc`/`data` request (`interceptor.blocked` records them). `ReportPage.js_errors` collects console errors.
- Single-session chart (#6): `app/view/web/session_chart.js` (`window.SessionChart`). One ECharts instance with four grids sharing the listening timeline in minutes: score, fatigue index, a cognitive-state band, and spatial-state rows (3 groups, one row each) plus alerts. Pauses are dashed `markLine`s in every grid. The data transforms (`cognitiveRuns`, `spatialIntervals`, `describeSecond`, ...) are pure functions that tests call through `runJavaScript`. Display labels for spatial states and alert types live there. A new spatial-state code or alert type must be added to `SPATIAL_GROUPS` / `ALERT_LABELS`; otherwise it is not drawn, or shows its raw code.
- Problem segments (#7): `app/problem_segments.py` `detect(samples, events, pauses)` is pure Python. Every threshold lives in `ProblemConfig`, mirrored in the `README.md` problem-segment table. Change both together, the same rule as for `attention_rules.py`. Windows are centered on each second and truncated at pauses (`floor(pause.t) + 1`) and at timeline gaps. Same-category segments never merge across a pause. `report_data.load_session` returns them as `problems`. The chart draws them as a top row (grid 4) plus same-colored `markArea` shading behind the score line; both use the same series name, so one legend toggle hides both.
- Multi-session summary (#8), tab 全体の推移:
  - Data: `report_data.summarize_sessions` (via the `summarize_sessions` slot) returns sessions that have samples, oldest first, with per-category problem `count` / `seconds`. It also returns `courses` (`course_summary`: grouped by course name, `None` is its own group, `avg_score` weighted by listening seconds, most recent course first).
  - It runs `problem_segments.detect` for every session, about 0.5 s for 60 × 90-minute lessons, on the background pool. It loads only when the tab is shown.
  - `app/view/web/summary_chart.js` (`window.SummaryChart`) draws three grids over a category axis (one entry per session): average score, listening minutes, and problem share. Problem shares are grouped bars, not stacked, because categories can overlap; each bar is that category's seconds / listening seconds, capped at 100.
  - Interrupted or recording sessions get a hollow point and a third label line.
  - Clicking a column opens that session in 授業ごと (`openSession`). The course filter and the course-table rows share `filterKey` (`all` / `none` / `name:<course>`). `body[data-summary]` is `loading` / `ready` / `error`.
- Record management (#10), tab 記録の管理: a table of every session (`list_sessions`, so it includes sessions without samples) with 名前を変更 / 削除 per row; clicking a row opens it in 授業ごと. The `rename_session` / `delete_session` slots open a modal Qt dialog inside the slot (`RenameCourseDialog`, qfluentwidgets `MessageBox`) and return `{"done": bool}`; the page then reloads (`body[data-manage]` is `busy` / `idle`). In tests, replace `bridge.ask_course_name` / `bridge.confirm_delete`. qfluentwidgets mask dialogs (`MessageBoxBase` / `MessageBox`) are child widgets, and the GPU-composited web view is drawn over them (#18). So these dialogs run inside `ReportInterface.page_frozen()`, which swaps the web view for a window-grab snapshot while the dialog is open. Any new mask dialog opened over the report page needs the same wrapper. `recording` sessions are refused twice: the buttons are disabled, and `session_store.rename_session` / `delete_session` only touch rows whose status is not `recording`. Deleting cascades to samples, events and pauses through the foreign keys.
- CSV export (#11): `app/export_csv.py` `export(conn, parent_dir, session_ids=None)` (pure Python, no Qt) writes `sessions` / `samples` / `events` / `pauses` / `problems` CSVs into a new folder. It writes into a temp folder first and renames it when done, so a failure leaves nothing behind. `recording` sessions are never exported. The columns are documented in the `README.md` CSV section; change both together. The bridge calls `choose_export_dir` (a `QFileDialog`; replace it in tests) on the UI thread, then writes the files on the background pool: about 2.5 s for 60 × 90-minute lessons. `export` reads everything inside one read transaction, so a rename or delete that happens during the export does not change the result.
- Text feedback (#9): `app/feedback.py`. `report_data.load_session` returns `feedback` (`None` when there are no samples). The page renders it as the フィードバック card above the chart; clicking a 気になった時間帯 item zooms the chart to that segment ±30 s.
  - Swappable: `build_context` assembles a pure-data `FeedbackContext` (score, problems, spatial events, earlier same-course averages), and `feedback.GENERATOR.generate(ctx)` returns `{summary, segments, more, hints, comparison, generator}`. A future LLM generator only has to implement `generate`.
  - Thresholds live in `FeedbackConfig`. All wording lives in `RuleFeedbackGenerator` and the `SEGMENT_TEXT` / `HINT_TEXT` dicts.
  - `test-scripts/verify_issue9_feedback.py` writes `feedback_samples.txt` for reviewing the wording.
- Tooltip time comes from an invisible per-second series (`__time__`). Custom-series bars snap `axisValue` to the bar start, and the category y-axes have `triggerTooltip: false`, otherwise the tooltip reports the row name. Colors were checked with the dataviz palette validator.
- The profile is a child of the page so it is released after the page. A `ReportInterface` still alive at interpreter exit (e.g. a pending `deleteLater` in tests) crashes WebEngine, so tests destroy it with `sip.delete`. Pages may be polled before `<body>` exists: use `document.body && ...`.

## Cross-file couplings to watch

- **Class order**: `AttentionAnalyzer.EMOTION_IDX` hardcodes the 7-class order of the current `best.pt` (Anger, Disgust, Fear, Happy, Neutral, Sad, Surprise). Expression processing is gated on `len(all_probs) == 7`. The paper's planned model is a transfer-learned **4-class** model (Understand/Doubt/Neutral/Disgust). Swapping it in would silently skip all expression logic until the 7→4 mapping in `attention_rules.py` is replaced.
- **Status-text colors**: the color comes from a tone (`info` / `success` / `warning` / `danger` / `muted`) sent alongside the text, never from the text itself. `SPATIAL_STATE_TONES` in `camera_thread.py` maps each spatial state to a tone, and `MonitorInterface.STATUS_TONE_STYLES` maps each tone to a color. Status strings are display-only Japanese and can be reworded freely. `show_async_alert` uses error styling only when the alert title contains `CRITICAL`; its InfoBars disappear after 3 s. Messages the user must not miss go through `error_signal` / `show_persistent_alert` instead (`duration=-1`, closed manually).
- **Alert rate limiting**: all alert types share one `last_alert_time` cooldown.
- **Thread lifetime**: `CameraThread.stop()` sets the run flag and calls `wait()`. `MainWindow.closeEvent` also stops the thread, so keep `camera_thread` as the attribute name on `MonitorInterface`.
- **Report page**: the page reads only through `ReportBridge`'s JSON (`app/report_data.py`). Fields the page uses are named after the DB columns, so renaming a column means updating `report.js` as well.
