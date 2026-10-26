import os
import time

import pytest

from app.camera_thread import SPATIAL_STATE_TONES

BLUE, RED, GREEN, ORANGE, GRAY = "#3B82F6", "#EF4444", "#10B981", "#F59E0B", "#808080"


@pytest.fixture
def monitor(qapp):
    from app.view.monitor_interface import MonitorInterface
    return MonitorInterface()


@pytest.mark.parametrize("state, color", [
    ("normal", ORANGE), ("head_down", ORANGE), ("away", ORANGE),
    ("taking_notes", GREEN), ("hidden_notes", GREEN),
    ("distracted", RED), ("hidden_abnormal", RED), ("absent", RED),
])
def test_spatial_state_color(monitor, state, color):
    monitor.update_status_text("x", SPATIAL_STATE_TONES.get(state, "warning"))
    assert color in monitor.scoreLabel.styleSheet()


@pytest.mark.parametrize("tone, color", [
    ("info", BLUE), ("success", GREEN), ("muted", GRAY), ("unknown", ORANGE),
])
def test_tone_color(monitor, tone, color):
    monitor.update_status_text("x", tone)
    assert color in monitor.scoreLabel.styleSheet()


def test_mesh_switch_labels(monitor):
    assert monitor.meshSwitch.text == "顔メッシュ：表示"
    monitor.meshSwitch.setChecked(False)
    assert monitor.meshSwitch.text == "顔メッシュ：非表示"


@pytest.mark.skipif(not os.path.exists("weights/best.pt"), reason="需要本地权重 weights/best.pt")
def test_pause_resume_keeps_calibration_and_clears_windows(monitor, monkeypatch):
    import cv2
    import ultralytics
    from PyQt5.QtTest import QTest

    import app.camera_thread as ct

    img = cv2.imread(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "zidane.jpg"))[0:400, 800:1200]
    opened = []

    class FakeCapture:
        def __init__(self, index):
            self.open = True
            opened.append(self)

        def isOpened(self):
            return self.open

        def read(self):
            time.sleep(0.03)
            return True, img.copy()

        def release(self):
            self.open = False

    monkeypatch.setattr(ct.cv2, "VideoCapture", FakeCapture)
    statuses = []

    monitor._on_start_clicked()
    thread = monitor.camera_thread
    thread.update_status_signal.connect(lambda text, tone: statuses.append((text, tone)))
    QTest.qWait(4500)
    assert thread.is_calibrated
    assert any(tone == "info" for _, tone in statuses)
    baseline = (thread.baseline_pitch, thread.baseline_yaw)
    analyzer_before = thread.analyzer

    monitor._on_pause_clicked()
    QTest.qWait(300)
    assert not thread.isRunning() and not opened[-1].open
    assert monitor.scoreLabel.text() == "一時停止中" and GRAY in monitor.scoreLabel.styleSheet()
    assert monitor.pauseBtn.text() == "再開"

    statuses.clear()
    monitor._on_pause_clicked()
    QTest.qWait(1500)
    try:
        assert thread.isRunning() and len(opened) == 2 and opened[-1].open
        assert not any(tone == "info" for _, tone in statuses)
        assert ("モニタリング中", "success") in statuses
        assert (thread.baseline_pitch, thread.baseline_yaw) == baseline
        assert thread.analyzer is not analyzer_before
        assert thread.analyzer.macro_buffer
        assert all(ts > time.time() - 2 for ts, _ in thread.analyzer.macro_buffer)
    finally:
        monitor._on_stop_clicked()
        QTest.qWait(200)
    assert not thread.isRunning() and not monitor.pauseBtn.isEnabled()
