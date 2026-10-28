"""#4 监控页接入课程记录：开始 / 暂停 / 继续 / 结束 / 关闭窗口 与数据库的对应关系"""
import pytest

from app import session_store
from tests.conftest import needs_weights

CALIBRATION_WAIT_MS = 4500  # 3 秒校准 + 模型启动余量


@pytest.fixture
def monitor(qapp, monkeypatch):
    from app.view.monitor_interface import MonitorInterface
    m = MonitorInterface()
    m.asked = []

    def fake_ask(names, answer="線形代数"):
        m.asked.append(list(names))
        return m.answer

    m.answer = "線形代数"
    monkeypatch.setattr(m, "_ask_course_name", fake_ask)
    yield m
    if hasattr(m, "camera_thread"):
        m.camera_thread.stop()


def wait(ms):
    from PyQt5.QtTest import QTest
    QTest.qWait(ms)


def db_rows(sql, *args):
    conn = session_store.connect()
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def assert_timeline_continuous(session_id):
    ts = [r["t"] for r in db_rows("SELECT t FROM samples WHERE session_id = ? ORDER BY t", session_id)]
    assert ts and ts == list(range(len(ts)))
    return ts


# ---------- 课程名对话框 ----------

def test_course_name_dialog(qapp):
    from PyQt5.QtWidgets import QWidget
    from app.view.course_name_dialog import CourseNameDialog
    window = QWidget()
    window.resize(800, 600)
    d = CourseNameDialog(["線形代数", "英語"], window)
    assert [d.nameBox.itemText(i) for i in range(d.nameBox.count())] == ["線形代数", "英語"]
    assert d.course_name() is None  # 默认不预选
    d.nameBox.setText("  統計学 第2回  ")
    assert d.course_name() == "統計学 第2回"
    assert (d.yesButton.text(), d.cancelButton.text()) == ("保存", "スキップ")


def test_course_name_helpers():
    conn = session_store.connect()
    a = session_store.create_session(conn, 100.0)
    b = session_store.create_session(conn, 200.0)
    c = session_store.create_session(conn, 300.0)
    session_store.set_course_name(conn, a, "英語")
    session_store.set_course_name(conn, b, " 線形代数 ")
    session_store.set_course_name(conn, c, "英語")
    assert session_store.list_course_names(conn) == ["英語", "線形代数"]  # 最近使用的在前
    session_store.set_course_name(conn, b, "   ")
    assert conn.execute("SELECT course_name FROM sessions WHERE id = ?", (b,)).fetchone()[0] is None
    conn.close()


# ---------- 完整流程 ----------

@needs_weights
def test_start_pause_resume_stop_records_session(monitor, fake_camera):
    monitor._on_start_clicked()
    wait(CALIBRATION_WAIT_MS + 1500)
    thread = monitor.camera_thread
    sid = thread.session_id
    assert sid is not None

    monitor._on_pause_clicked()
    wait(800)
    paused = db_rows("SELECT * FROM pauses WHERE session_id = ?", sid)
    assert len(paused) == 1 and paused[0]["resumed_at"] is None

    monitor._on_pause_clicked()
    wait(2000)
    monitor._on_stop_clicked()
    wait(300)

    (s,) = db_rows("SELECT * FROM sessions")
    assert s["id"] == sid and s["status"] == "completed" and s["course_name"] == "線形代数"
    assert s["baseline_pitch"] is not None and s["ended_at"] >= s["started_at"]
    assert monitor.asked == [[]]  # 第一次没有历史课程名

    (p,) = db_rows("SELECT * FROM pauses WHERE session_id = ?", sid)
    assert p["resumed_at"] - p["paused_at"] >= 0.7

    ts = assert_timeline_continuous(sid)
    assert len(ts) >= 3  # 暂停前约 1.5 秒 + 继续后约 2 秒
    samples = db_rows("SELECT * FROM samples WHERE session_id = ? ORDER BY t", sid)
    assert samples[0]["wall_time"] >= s["started_at"] + 3.0  # 校准期不记录
    assert all(r["face_ratio"] > 0 for r in samples)
    # 继续后的第一条样本在暂停之后
    resumed = [r for r in samples if r["wall_time"] >= p["resumed_at"]]
    assert resumed and resumed[0]["t"] == max(r["t"] for r in samples if r["wall_time"] < p["paused_at"]) + 1


@needs_weights
def test_second_session_offers_previous_course_names(monitor, fake_camera):
    for answer in ("線形代数", None):
        monitor.answer = answer
        monitor._on_start_clicked()
        wait(CALIBRATION_WAIT_MS + 1500)
        monitor._on_stop_clicked()
        wait(300)
    assert monitor.asked == [[], ["線形代数"]]
    names = [r["course_name"] for r in db_rows("SELECT course_name FROM sessions ORDER BY id")]
    assert names == ["線形代数", None]  # 跳过则不填写


@needs_weights
def test_stop_while_paused_completes_session(monitor, fake_camera):
    monitor._on_start_clicked()
    wait(CALIBRATION_WAIT_MS + 1000)
    monitor._on_pause_clicked()
    wait(300)
    monitor._on_stop_clicked()
    wait(300)
    (s,) = db_rows("SELECT * FROM sessions")
    assert s["status"] == "completed" and s["course_name"] == "線形代数"
    (p,) = db_rows("SELECT * FROM pauses")
    assert p["resumed_at"] is None and s["ended_at"] >= p["paused_at"]
    assert_timeline_continuous(s["id"])


@needs_weights
def test_pause_during_calibration_recalibrates_and_records(monitor, fake_camera):
    statuses = []
    monitor._on_start_clicked()
    wait(1200)
    monitor._on_pause_clicked()
    wait(300)
    thread = monitor.camera_thread
    assert not thread.is_calibrated
    thread.update_status_signal.connect(lambda text, tone: statuses.append(tone))
    monitor._on_pause_clicked()
    wait(CALIBRATION_WAIT_MS)
    assert "info" in statuses and thread.is_calibrated  # 继续后重新校准
    monitor._on_stop_clicked()
    wait(300)
    (s,) = db_rows("SELECT * FROM sessions")
    assert s["status"] == "completed"
    assert db_rows("SELECT * FROM pauses") == []  # 记录器尚未开始时的暂停不写暂停点
    assert_timeline_continuous(s["id"])


@needs_weights
def test_away_frames_create_spatial_event(monitor, fake_camera):
    monitor._on_start_clicked()
    wait(CALIBRATION_WAIT_MS + 1000)
    fake_camera.show_face = False
    wait(2500)
    fake_camera.show_face = True
    wait(1500)
    monitor._on_stop_clicked()
    wait(300)
    events = db_rows("SELECT * FROM events WHERE kind = 'spatial'")
    assert [e["type"] for e in events] == ["away"]
    assert events[0]["t_end"] - events[0]["t_start"] >= 2.0
    away = db_rows("SELECT * FROM samples WHERE spatial_state = 'away'")
    assert away and all(r["face_ratio"] < 1 for r in away) and all(r["cognitive_state"] is None for r in away if r["face_ratio"] == 0)


@needs_weights
def test_stop_during_calibration_leaves_no_record(monitor, fake_camera):
    monitor._on_start_clicked()
    wait(1500)  # 仍在 3 秒校准期内
    assert monitor.camera_thread.session_id is not None  # 摄像头打开时已建立
    monitor._on_stop_clicked()
    wait(300)
    assert monitor.camera_thread.session_id is None
    assert db_rows("SELECT * FROM sessions") == []  # 空记录被删除
    assert monitor.asked == []  # 不询问课程名
    assert monitor.videoLabel.text() == "カメラ停止中"


@needs_weights
def test_close_window_during_calibration_leaves_no_record(monitor, fake_camera):
    from types import SimpleNamespace
    from PyQt5.QtGui import QCloseEvent
    import main

    monitor._on_start_clicked()
    wait(1500)
    main.MainWindow.closeEvent(SimpleNamespace(monitorInterface=monitor), QCloseEvent())
    assert db_rows("SELECT * FROM sessions") == []


@needs_weights
def test_camera_failure_creates_no_session(monitor, fake_camera):
    fake_camera.can_open = False
    alerts = []
    monitor._on_start_clicked()
    monitor.camera_thread.alert_signal.connect(lambda title, msg: alerts.append(title))
    wait(1500)
    monitor._on_stop_clicked()
    wait(300)
    assert monitor.camera_thread.session_id is None
    assert db_rows("SELECT * FROM sessions") == []
    assert monitor.asked == []  # 没有记录就不询问课程名
    assert monitor.videoLabel.text() == "カメラ停止中"


@needs_weights
def test_close_window_while_paused_completes_without_dialog(monitor, fake_camera):
    # offscreen 平台下无法创建 FluentWindow（qframelesswindow 在 macOS 上直接调用 Cocoa），
    # 这里对 closeEvent 传入只含 monitorInterface 的替身；真实窗口的关闭见 test-scripts/verify_issue4_recording.py
    from types import SimpleNamespace
    from PyQt5.QtGui import QCloseEvent
    import main

    monitor._on_start_clicked()
    wait(CALIBRATION_WAIT_MS + 1000)
    monitor._on_pause_clicked()
    wait(300)
    event = QCloseEvent()
    main.MainWindow.closeEvent(SimpleNamespace(monitorInterface=monitor), event)
    assert event.isAccepted()
    (s,) = db_rows("SELECT * FROM sessions")
    assert s["status"] == "completed" and s["course_name"] is None
    assert monitor.asked == []  # 关闭窗口属于正常结束，不弹课程名


@needs_weights
def test_recording_failure_does_not_stop_monitoring(monitor, fake_camera, tmp_path):
    bad = tmp_path / "not_a_db"
    bad.mkdir()  # 目录不能作为数据库文件打开
    monitor.db_path = bad
    alerts, statuses = [], []
    monitor._on_start_clicked()
    monitor.camera_thread.alert_signal.connect(lambda title, msg: alerts.append(title))
    monitor.camera_thread.update_status_signal.connect(lambda text, tone: statuses.append(text))
    wait(CALIBRATION_WAIT_MS + 1000)
    assert monitor.camera_thread.isRunning()
    assert alerts.count("記録エラー") == 1  # 只提示一次
    assert "モニタリング中" in statuses or any(t.startswith("認知状態") for t in statuses)
    monitor._on_stop_clicked()
    wait(300)
    assert not monitor.camera_thread.isRunning()


@needs_weights
def test_frames_and_ui_keep_flowing_while_recording(monitor, fake_camera):
    frames = []
    monitor._on_start_clicked()
    monitor.camera_thread.change_pixmap_signal.connect(lambda p: frames.append(1))
    wait(CALIBRATION_WAIT_MS + 2000)
    monitor._on_stop_clicked()
    wait(300)
    recorded = db_rows("SELECT SUM(frame_count) AS n FROM samples")[0]["n"]
    assert recorded and len(frames) > recorded  # 校准期的帧显示但不记录
