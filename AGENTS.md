# AI Agent Guide: Self-Analysis System Demo

This document provides context, commands, and standards for AI agents working on the "Self-Analysis System Demo" project.
Project Type: Desktop GUI Application (PyQt5 + qfluentwidgets + YOLO/OpenCV).

## 1. Project Structure
- `main.py`: Application entry point. Sets up the main window and navigation.
- `app/`: Main application source code.
  - `camera_thread.py`: `QThread` subclass for background video capture and processing.
  - `view/`: UI interface definitions.
    - `monitor_interface.py`: Real-time camera feed and attention monitoring tab.
    - `report_interface.py`: Statistics and analysis reporting tab.
- `runs/`, `weights/`: (Implied) Directories for YOLO models and inference results.

## 2. Build & Run Commands

### Environment Setup
No lockfiles present. Install core dependencies manually:
```bash
pip install PyQt5 qfluentwidgets opencv-python numpy torch torchvision ultralytics
```
*Note: PyTorch/YOLOv5 dependencies may vary based on hardware (CUDA).*

### Execution
Run the application:
```bash
python main.py
```
*Note: Ensure a webcam is available or modify `camera_thread.py` to use a video file.*

### Testing & Linting
**Status**: No formal testing or linting framework is currently configured.
- **Syntax Check**: `python -m py_compile main.py`
- **Recommended**: Use `flake8` or `ruff` for ad-hoc style checks if modifying core logic.

## 3. Code Style & Conventions

### Python Conventions
- **Naming**:
  - Classes: `PascalCase` (e.g., `MainWindow`, `MonitorInterface`, `CameraThread`)
  - Functions/Variables: `snake_case` (e.g., `init_navigation`, `update_frame`, `video_label`)
  - Private Members: `_underscore_prefix` (e.g., `_run_flag`, `_on_clicked`)
  - Signals: `snake_case` + `_signal` suffix (e.g., `change_pixmap_signal`)
- **Type Hints**: Strongly encouraged for function arguments and return types.
  - Example: `def update_image(self, image: QImage) -> None:`
- **Imports**: Grouped: Stdlib -> Third-party (PyQt, cv2) -> Local (`app.*`).
  - Prefer explicit imports over wildcards (`from PyQt5.QtCore import Qt` NOT `from PyQt5.QtCore import *`).

### UI/PyQt Patterns
- **Threading**: heavy CV/ML operations MUST run in `QThread` (e.g., `CameraThread`) to avoid freezing the UI.
- **Signals/Slots**: Use `pyqtSignal` to communicate between threads and UI.
  - *Never* update UI widgets directly from a background thread. Emit a signal with data instead.
- **Fluent Widgets**: Use `qfluentwidgets` components (`CardWidget`, `PrimaryPushButton`, `SubtitleLabel`) for consistent styling.

### YOLO/OpenCV Integration
- **Input**: OpenCV reads images in BGR format.
- **Display**: Convert BGR to RGB before creating `QImage`/`QPixmap` for Qt display.
  - `rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- **Inference**: Run inference frame-by-frame in `run()` method of `CameraThread`.

## 4. Feature Implementation Guidelines (Student Analysis)
- **Context**: Adapted from driver fatigue detection.
- **Terminology**:
  - "Driver" -> "Student"
  - "Fatigue" -> "Distraction" / "Inattention"
  - "Drowsy" -> "Unfocused" / "Sleepy"
- **Metrics**: Focus duration, distraction frequency, head pose estimation (optional).

## 5. Critical Constraints
- **Do NOT** block the main thread.
- **Do NOT** hardcode absolute file paths. Use `os.path` or `pathlib`.
- **Do NOT** commit large weight files (`.pt`, `.onnx`). Add them to `.gitignore`.
