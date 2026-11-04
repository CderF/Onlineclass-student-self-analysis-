"""模型在后台线程加载（#12）：点击开始不在界面线程构造模型、暂停后继续不重复加载、加载失败时提示"""
import threading
import time

import pytest

from tests.conftest import needs_weights
from tests.test_recording_flow import CALIBRATION_WAIT_MS, db_rows, monitor, wait  # noqa: F401


@pytest.fixture
def built(monkeypatch):
    """记录两个模型类每次构造时所在的线程（是否主线程）"""
    import app.camera_thread as ct
    log = {"ExpressionClassifier": [], "FaceMeshInference": []}
    for name in log:
        real = getattr(ct, name)

        def spy(*args, real=real, name=name, **kwargs):
            log[name].append(threading.current_thread() is threading.main_thread())
            return real(*args, **kwargs)

        monkeypatch.setattr(ct, name, spy)
    return log


@pytest.fixture
def alerts(monitor, monkeypatch):
    """
    在开始前替换，CameraThread 的信号连接到的就是它们。
    模型错误必须走不自动消失的提示（show_persistent_alert），不能走 3 秒消失的普通警报。
    """
    class Alerts(list):
        transient: list

    got, transient = Alerts(), []
    monkeypatch.setattr(monitor, "show_persistent_alert", lambda title, msg: got.append((title, msg)))
    monkeypatch.setattr(monitor, "show_async_alert", lambda title, msg: transient.append((title, msg)))
    got.transient = transient
    return got


@needs_weights
def test_start_click_does_not_build_models_on_ui_thread(monitor, fake_camera, built):
    start = time.perf_counter()
    monitor._on_start_clicked()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert built == {"ExpressionClassifier": [], "FaceMeshInference": []}   # 返回时还没有构造
    assert elapsed_ms < 50, f"_on_start_clicked 耗时 {elapsed_ms:.0f} ms"
    wait(1500)
    assert built == {"ExpressionClassifier": [False], "FaceMeshInference": [False]}   # 在后台线程构造
    assert len(fake_camera.instances) == 1


@needs_weights
def test_pause_resume_does_not_reload(monitor, fake_camera, built):
    monitor._on_start_clicked()
    wait(1500)
    for _ in range(2):
        monitor._on_pause_clicked()
        wait(300)
        monitor._on_pause_clicked()
        wait(800)
    assert built == {"ExpressionClassifier": [False], "FaceMeshInference": [False]}
    assert len(fake_camera.instances) == 3


@needs_weights
def test_each_start_builds_its_own_models_in_background(monitor, fake_camera, built):
    for _ in range(2):
        monitor._on_start_clicked()
        wait(1200)
        monitor._on_stop_clicked()
        wait(300)
    assert built == {"ExpressionClassifier": [False, False], "FaceMeshInference": [False, False]}


def test_classifier_load_failure_alerts_and_keeps_monitoring(monitor, fake_camera, alerts, monkeypatch):
    """weights 加载失败（ExpressionClassifier.model 为 None）：提示一次，继续监控并记录（不需要真实权重）"""
    import app.camera_thread as ct

    class Broken:
        model = None

        def process_face(self, face_img):
            return "Unknown", 0.0, None

    monkeypatch.setattr(ct, "ExpressionClassifier", Broken)
    scores = []
    monitor._on_start_clicked()
    monitor.camera_thread.update_score_signal.connect(scores.append)
    wait(CALIBRATION_WAIT_MS + 1500)
    assert [a[0] for a in alerts] == ["モデルエラー"] and "weights/best.pt" in alerts[0][1]
    assert "モデルエラー" not in [a[0] for a in alerts.transient]
    assert len(scores) > 10                           # 画面与分数照常更新
    sid = monitor.camera_thread.session_id
    samples = db_rows("SELECT cognitive_state FROM samples WHERE session_id = ?", sid)
    assert samples and all(s["cognitive_state"] is None for s in samples)   # 没有表情判定
    monitor._on_stop_clicked()
    wait(300)


def test_facemesh_failure_alerts_and_does_not_open_camera(monitor, fake_camera, alerts, monkeypatch):
    import app.camera_thread as ct

    def broken(*args, **kwargs):
        raise RuntimeError("mediapipe graph failed")

    monkeypatch.setattr(ct, "FaceMeshInference", broken)
    monitor._on_start_clicked()
    wait(800)
    assert not monitor.camera_thread.isRunning()
    assert [a[0] for a in alerts] == ["モデルエラー"] and "顔検出モデル" in alerts[0][1]
    assert alerts.transient == []
    assert monitor.scoreLabel.text() == "モデルを読み込めません"
    assert fake_camera.instances == [] and db_rows("SELECT * FROM sessions") == []
    monitor._on_stop_clicked()
    wait(300)
    assert monitor.asked == [] and monitor.videoLabel.text() == "カメラ停止中"


def test_stop_while_loading_opens_no_camera(monitor, fake_camera, monkeypatch):
    import app.camera_thread as ct
    real = ct.FaceMeshInference

    def slow(*args, **kwargs):
        time.sleep(0.6)
        return real(*args, **kwargs)

    monkeypatch.setattr(ct, "FaceMeshInference", slow)

    class Quiet:
        model = object()

        def process_face(self, face_img):
            return "Unknown", 0.0, None

    monkeypatch.setattr(ct, "ExpressionClassifier", Quiet)
    monitor._on_start_clicked()
    wait(100)
    start = time.perf_counter()
    monitor._on_stop_clicked()                   # stop() 等加载结束后线程直接退出
    assert time.perf_counter() - start < 2.0
    wait(300)
    assert fake_camera.instances == [] and db_rows("SELECT * FROM sessions") == []
    assert monitor.asked == []


def test_persistent_alert_stays_until_closed(monitor):
    """模型错误的提示不自动消失（普通警报 3 秒后消失），点关闭按钮后才消失"""
    from PyQt5.QtTest import QTest
    from qfluentwidgets import InfoBar
    monitor.resize(1000, 700)
    monitor.show()
    monitor.show_async_alert("学習アラート", "普通の警報")
    monitor.show_persistent_alert("モデルエラー", "表情認識モデルを読み込めませんでした。")
    bars = {b.title: b for b in monitor.findChildren(InfoBar)}
    assert set(bars) == {"学習アラート", "モデルエラー"} and bars["モデルエラー"].duration < 0
    wait(4000)
    visible = {b.title for b in monitor.findChildren(InfoBar) if b.isVisible()}
    assert visible == {"モデルエラー"}                     # 普通警报已消失，模型错误仍在
    QTest.mouseClick(bars["モデルエラー"].closeButton, 1)
    wait(800)
    assert not [b for b in monitor.findChildren(InfoBar) if b.isVisible()]
