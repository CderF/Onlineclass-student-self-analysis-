"""
#17 删除记录后数据库文件中不留残留的专项验证（补充 tests/test_secure_delete.py）

运行：在仓库根目录执行（会短暂弹出真实窗口，约 1 分钟）
    python test-scripts/verify_issue17_secure_delete.py

检查项：被删除的课的课程名与一个特征浮点数（以 SQLite 存储的 8 字节大端形式）在数据库文件与 WAL 中出现的次数
  1. 对照实验：关闭 secure_delete、删除后不清空 WAL（修复前在 SQLite 默认未开启 SECURE_DELETE 的环境里的行为）
     时能找到残留，确认检查方法有效
  2. 真实窗口：录制进行中（另一连接每秒写入），用真实 Qt 鼠标在「記録の管理」删除一节 90 分钟的课并在真实确认框点「削除」，
     删除后立即无残留、录制继续；停止并关闭程序后仍无残留
  3. 崩溃：删除完成后进程被 SIGKILL（录制连接未关闭、WAL 未经正常关闭），文件中无残留
  4. 耗时：删除 90 分钟课程（含清空 WAL）
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import os
import signal
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

failures = []
NAME = "ZZ削除される授業ZZ"
FATIGUE = 0.123456789012345
MARKS = (NAME.encode(), struct.pack(">d", FATIGUE))


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def residue(db):
    out = {}
    for suffix in ("", "-wal"):
        path = Path(str(db) + suffix)
        data = path.read_bytes() if path.exists() else b""
        out[suffix or "db"] = sum(data.count(m) for m in MARKS)
    return out


def scalar(db, sql, *args):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(sql, args).fetchone()[0]
    finally:
        conn.close()


def secret_lesson(db, start, seconds):
    """90 分钟级的课：每秒一帧，疲劳指数为特征值，课程名为特征字符串"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder
    conn = session_store.connect(db)
    sid = session_store.create_session(conn, start)
    rec = SessionRecorder(sid, start, db)
    for i in range(seconds):
        rec.add_frame(AnalysisResult(score=70, status_text="", alert=None, has_face=True, cognitive_state="Neutral",
                                     fatigue_index=FATIGUE, spatial_state="normal", delta_pitch=0.0, delta_yaw=0.0),
                      start + i)
    rec.close(start + seconds)
    session_store.finish_session(conn, sid, start + seconds)
    session_store.set_course_name(conn, sid, NAME)
    conn.close()
    return sid


def control(tmp: Path):
    print("\n[1] 对照实验：不做 #17 的处理")
    from app import session_store
    db = tmp / "control.db"
    sid = secret_lesson(db, time.time() - 86400, 600)
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA secure_delete = OFF")
    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    open_res = residue(db)
    conn.close()
    closed_res = residue(db)
    check("关闭 secure_delete 且不清空 WAL：删除后能找到残留（检查方法有效）",
          open_res["-wal"] > 0 and closed_res["db"] > 0, f"连接打开时 {open_res}，关闭后 {closed_res}")

    db2 = tmp / "fixed.db"
    sid = secret_lesson(db2, time.time() - 86400, 600)
    conn = session_store.connect(db2)
    session_store.delete_session(conn, sid)
    open_res = residue(db2)
    conn.close()
    check("同样的数据用 session_store.delete_session：无残留", open_res == {"db": 0, "-wal": 0} and residue(db2) == open_res,
          f"连接打开时 {open_res}，关闭后 {residue(db2)}")


CRASH_CHILD = r"""
import os, signal, sys, time
sys.path.insert(0, sys.argv[2])
from app import session_store
from app.session_recorder import SessionRecorder
from tests.test_session_store import frame
db = sys.argv[1]
conn = session_store.connect(db)
live = session_store.create_session(conn, time.time())
rec = SessionRecorder(live, time.time(), db)          # 模拟录制中的连接
t0 = time.time()
for i in range(20):
    rec.add_frame(frame(), t0 + i)
sid = int(sys.argv[3])
print("deleted", session_store.delete_session(conn, sid), flush=True)
for i in range(20, 25):
    rec.add_frame(frame(), t0 + i)
os.kill(os.getpid(), signal.SIGKILL)
"""


def crash(tmp: Path):
    print("\n[3] 删除后进程被 SIGKILL")
    db = tmp / "crash.db"
    sid = secret_lesson(db, time.time() - 86400, 600)
    before = residue(db)
    proc = subprocess.run([sys.executable, "-c", CRASH_CHILD, str(db), str(ROOT), str(sid)],
                          capture_output=True, text=True, timeout=120)
    after = residue(db)
    check("删除完成后被 SIGKILL：数据库与 WAL 中均无残留",
          proc.returncode == -signal.SIGKILL and "deleted True" in proc.stdout and after == {"db": 0, "-wal": 0},
          f"退出码 {proc.returncode}，删除前 {before}，崩溃后 {after}，WAL {'存在' if Path(str(db) + '-wal').exists() else '不存在'}")
    from app import session_store
    conn = session_store.connect(db)
    n = session_store.mark_interrupted_sessions(conn)
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    check("重启后恢复正常：录制中的课标为中断、完整性检查通过、仍无残留",
          n == 1 and ok == "ok" and residue(db) == {"db": 0, "-wal": 0}, f"中断 {n} 节，{ok}，{residue(db)}")


def in_window(tmp: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    from verify_issue5_report import run_js, switch_to_report, wait_state
    from verify_issue8_summary import click
    from verify_issue10_record_management import DialogDriver, lesson

    db = tmp / "window.db"
    session_store.DEFAULT_DB_PATH = db
    lesson(db, time.time() - 3 * 86400, 300, "英語")
    sid = secret_lesson(db, time.time() - 2 * 86400, 5400)

    import main as app_main
    import verify_issue4_recording as v4
    from app.view.course_name_dialog import CourseNameDialog
    v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 850)
    w.show()
    report, m = w.reportInterface, w.monitorInterface
    target = report.webView.focusProxy() or report.webView
    driver = DialogDriver(w)
    QTest.qWait(1500)

    print("\n[2] 真实窗口：录制中删除")
    check("删除前文件中能找到这节课的数据", sum(residue(db).values()) > 0, str(residue(db)))
    alerts = []
    QTest.mouseClick(m.startBtn, 1)
    m.camera_thread.alert_signal.connect(lambda title, msg: alerts.append(title))
    QTest.qWait(8000)
    live = m.camera_thread.session_id
    switch_to_report(w)
    wait_state(report.page)
    run_js(report.page, "document.getElementById('tab-manage').click()")
    QTest.qWait(400)
    pt = run_js(report.page, f"""(() => {{
        const el = document.querySelector('#record-table tr[data-id="{sid}"] button[data-action="delete"]');
        el.scrollIntoView({{block: 'center'}});
        const r = el.getBoundingClientRect();
        return [r.left + r.width / 2, r.top + r.height / 2];
    }})()""")
    timing = {}
    driver.expect(lambda d: (timing.setdefault("t", time.perf_counter()), QTest.mouseClick(d.yesButton, 1)))
    run_js(report.page, "document.body.dataset.manage = 'stale'")
    click(target, pt)
    end = time.time() + 15
    while time.time() < end and run_js(report.page, "document.body.dataset.manage") != "idle":
        QTest.qWait(20)
    ms = (time.perf_counter() - timing.get("t", time.perf_counter())) * 1000
    n0 = scalar(db, "SELECT COUNT(*) FROM samples WHERE session_id = ?", live)
    res = residue(db)
    gone = scalar(db, "SELECT COUNT(*) FROM sessions WHERE id = ?", sid) == 0
    check("录制中删除 90 分钟的课：删除后立即在数据库与 WAL 中都无残留", gone and res == {"db": 0, "-wal": 0},
          f"{res}，WAL {Path(str(db) + '-wal').stat().st_size} 字节")
    check("删除 90 分钟课程（含清空 WAL）：确认到列表刷新完成 < 1 秒", ms < 1000, f"{ms:.0f} ms")
    QTest.qWait(3000)
    n1 = scalar(db, "SELECT COUNT(*) FROM samples WHERE session_id = ?", live)
    check("录制继续写入、无「記録エラー」", n1 >= n0 + 2 and "記録エラー" not in alerts, f"样本 {n0} → {n1}，警报 {alerts}")

    def fill():
        for d in w.findChildren(CourseNameDialog):
            if d.isVisible():
                d.nameBox.setText("英語")
                QTest.mouseClick(d.yesButton, 1)

    from PyQt5.QtCore import QTimer
    w.switchTo(m)
    QTimer.singleShot(1500, fill)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    driver.timer.stop()
    w.close()
    QTest.qWait(500)
    import gc
    gc.collect()
    check("停止录制并关闭窗口后仍无残留", residue(db) == {"db": 0, "-wal": 0}, str(residue(db)))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        control(tmp)
        crash(tmp)
        in_window(tmp)
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
