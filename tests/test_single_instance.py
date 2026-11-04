"""单实例（#13）：第二个实例启动不影响第一个实例正在记录的课程；崩溃遗留的记录仍会标为 interrupted"""
import os
import signal
import subprocess
import sys
import time

import pytest

from app import session_store
from app.session_recorder import SessionRecorder
from app.single_instance import SingleInstance
from tests.conftest import ROOT, needs_weights
from tests.test_recording_flow import CALIBRATION_WAIT_MS, assert_timeline_continuous, db_rows, monitor, wait  # noqa: F401
from tests.test_session_store import T0, frame

# 子进程：拿锁后输出 ready；收到调出窗口的通知时输出 activated 并退出；kill 参数时拿锁并写入一条记录后自杀
HOLDER = r"""
import os, signal, sys
sys.path.insert(0, sys.argv[1])
from PyQt5.QtCore import QCoreApplication, QTimer
from app.single_instance import SingleInstance
app = QCoreApplication(sys.argv)
inst = SingleInstance(sys.argv[2])
print("ready" if inst.acquire() else "locked", flush=True)
if sys.argv[3] == "kill":
    from app import session_store
    from app.session_recorder import SessionRecorder
    from tests.test_session_store import T0, frame
    conn = session_store.connect(sys.argv[2])
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, sys.argv[2])
    for i in range(15):
        rec.add_frame(frame(spatial="away"), T0 + i * 0.1)
    print("recorded", sid, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
inst.activated.connect(lambda: (print("activated", flush=True), app.quit()))
QTimer.singleShot(15000, app.quit)
app.exec_()
inst.release()
"""


def start_holder(db, mode="wait"):
    proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(ROOT), str(db), mode], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
    return proc


def wait_until(cond, timeout_ms=3000):
    from PyQt5.QtTest import QTest
    deadline = time.monotonic() + timeout_ms / 1000
    while not cond() and time.monotonic() < deadline:
        QTest.qWait(20)
    return cond()


def recording_session_with_samples(db):
    """写入一条仍在记录中的课程（有 1 秒样本，空间状态事件未关闭）"""
    conn = session_store.connect(db)
    sid = session_store.create_session(conn, T0)
    conn.close()
    rec = SessionRecorder(sid, T0, db)
    for i in range(15):
        rec.add_frame(frame(spatial="away"), T0 + i * 0.1)
    return sid, rec


def session(sid):
    rows = db_rows("SELECT status, ended_at FROM sessions WHERE id = ?", sid)
    return rows[0] if rows else None


# ---------- 锁与通知 ----------

def test_second_instance_on_same_db_is_refused_and_activates_first(qapp, isolated_db):
    first, second = SingleInstance(), SingleInstance()
    activated = []
    assert first.acquire() is True
    assert first.lock_path == isolated_db.with_name("self_analysis.db.lock") and first.lock_path.exists()
    first.activated.connect(lambda: activated.append(1))
    assert second.acquire() is False
    assert second.notify_running() is True
    assert wait_until(lambda: activated == [1])
    first.release()
    assert not first.lock_path.exists()
    assert second.acquire() is True                     # 第一个退出后可以正常启动
    second.release()


def test_each_db_has_its_own_lock(qapp, tmp_path):
    a, b = SingleInstance(tmp_path / "a" / "x.db"), SingleInstance(tmp_path / "b" / "x.db")
    assert a.server_name != b.server_name
    assert a.acquire() and b.acquire()
    a.release()
    b.release()


def test_lock_held_by_another_process(qapp, isolated_db):
    holder = start_holder(isolated_db)
    try:
        me = SingleInstance()
        assert me.acquire() is False
        assert me.notify_running() is True
        out, err = holder.communicate(timeout=10)
        assert "activated" in out and holder.returncode == 0, err
        assert me.acquire() is True                     # 对方正常退出后锁已释放
        me.release()
    finally:
        holder.kill()


def test_lock_of_killed_process_is_taken_over(qapp, isolated_db):
    holder = start_holder(isolated_db)
    holder.kill()
    holder.wait(5)
    assert isolated_db.with_name("self_analysis.db.lock").exists()   # 被强制结束，锁文件留在原处
    me, later = SingleInstance(), SingleInstance()
    assert me.acquire() is True                         # 按 PID 判定已失效，直接接管
    activated = []
    me.activated.connect(lambda: activated.append(1))
    assert later.acquire() is False and later.notify_running() is True   # 留下的套接字文件不妨碍新的监听
    assert wait_until(lambda: activated == [1])
    me.release()


def test_old_lock_of_running_instance_is_not_taken_over(qapp, isolated_db):
    """QLockFile 把 30 秒以前的锁视为疑似失效，但另一个进程仍持有它时不能被接管（运行了 1 小时的实例）"""
    holder = start_holder(isolated_db)
    try:
        old = time.time() - 3600
        os.utime(isolated_db.with_name("self_analysis.db.lock"), (old, old))
        me = SingleInstance()
        assert me.acquire() is False
        assert me.notify_running() is True
        out, err = holder.communicate(timeout=10)
        assert "activated" in out, err
    finally:
        holder.kill()


def test_notify_without_running_instance_gives_up(qapp, isolated_db):
    start = time.monotonic()
    assert SingleInstance().notify_running(timeout_s=0.3) is False
    assert time.monotonic() - start < 2


# ---------- 启动流程（main.start_instance） ----------

def test_second_start_leaves_running_recording_untouched(qapp, isolated_db):
    import main
    first = main.start_instance()
    assert first is not None
    sid, rec = recording_session_with_samples(isolated_db)
    conn = session_store.connect()
    empty = session_store.create_session(conn, T0 + 10)   # 第一个实例刚打开摄像头、还在校准：没有样本
    conn.close()

    assert main.start_instance() is None                 # 第二个实例：不检查遗留，直接退出
    assert session(sid) == {"status": "recording", "ended_at": None}
    assert session(empty) == {"status": "recording", "ended_at": None}   # 空记录也没被删
    assert db_rows("SELECT t_end FROM events WHERE session_id = ?", sid) == [{"t_end": None}]

    for i in range(15, 30):                               # 第一个实例继续记录并正常结束
        rec.add_frame(frame(), T0 + i * 0.1)
    rec.close(T0 + 3.0)
    conn = session_store.connect()
    assert session_store.finish_session(conn, sid, T0 + 3.0) is True
    conn.close()
    assert session(sid) == {"status": "completed", "ended_at": T0 + 3.0}
    first.release()


def test_start_marks_crash_leftovers_when_no_instance_running(qapp, isolated_db):
    import main
    sid, rec = recording_session_with_samples(isolated_db)
    del rec
    instance = main.start_instance()
    assert instance is not None
    assert session(sid) == {"status": "interrupted", "ended_at": T0}
    instance.release()


def test_start_after_killed_instance_marks_its_session_interrupted(qapp, isolated_db):
    import main
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(ROOT), str(isolated_db), "kill"], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = holder.communicate(timeout=30)
    assert holder.returncode == -signal.SIGKILL and "ready" in out, err
    sid = int(out.split("recorded")[1].split()[0])
    assert session(sid)["status"] == "recording"
    instance = main.start_instance()
    assert instance is not None
    assert session(sid) == {"status": "interrupted", "ended_at": T0}
    instance.release()


@needs_weights
@pytest.mark.parametrize("paused", [False, True])
def test_second_start_during_real_recording(qapp, monitor, fake_camera, paused):
    """真实的 CameraThread 记录中（含暂停中）启动第二个实例：记录不受影响，之后照常结束为 completed"""
    import main
    first = main.start_instance()
    monitor._on_start_clicked()
    wait(CALIBRATION_WAIT_MS + 1500)
    sid = monitor.camera_thread.session_id
    if paused:
        monitor._on_pause_clicked()
        wait(500)
    before = len(db_rows("SELECT t FROM samples WHERE session_id = ?", sid))
    assert before > 0
    assert main.start_instance() is None
    assert session(sid)["status"] == "recording"
    if paused:
        monitor._on_pause_clicked()
    wait(1500)
    assert len(db_rows("SELECT t FROM samples WHERE session_id = ?", sid)) > before   # 记录继续
    monitor._on_stop_clicked()
    wait(300)
    assert session(sid)["status"] == "completed"
    assert_timeline_continuous(sid)
    first.release()
