"""
#4 监控页接入课程记录的专项验证（补充 tests/test_recording_flow.py）

运行：在仓库根目录执行（会短暂弹出真实窗口）
    python test-scripts/verify_issue4_recording.py
    python test-scripts/verify_issue4_recording.py --real-camera   # 人工验证：用真实摄像头，自己操作后关闭窗口

检查项：
  1. 真实窗口端到端：原生平台创建 MainWindow，用按钮完成 开始 → 离席 → 暂停 → 继续 → 终了，
     在真实的课程名对话框里输入并点「保存」，再开一节课后暂停中直接关闭窗口
  2. 时间轴核算：记录的秒数 ≈ 墙上时长 − 校准 − 暂停
  3. 真实崩溃：子进程运行真实程序并开始记录，SIGKILL 后再次启动，检查遗留记录被标为 interrupted
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def install_fake_camera():
    """用人脸近景图模拟摄像头；FakeCapture.show_face=False 时模拟离席"""
    import cv2
    import numpy as np
    import ultralytics
    import app.camera_thread as ct

    face = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))[0:400, 800:1200]

    class FakeCapture:
        show_face = True

        def __init__(self, index):
            self.open = True

        def isOpened(self):
            return self.open

        def read(self):
            time.sleep(0.03)
            return True, (face.copy() if FakeCapture.show_face else np.zeros_like(face))

        def release(self):
            self.open = False

    ct.cv2.VideoCapture = FakeCapture
    return FakeCapture


def rows(db, sql, *args):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


# ------------------------------------------------------------------ 1 + 2. 真实窗口端到端
def end_to_end(db: Path):
    print("\n[1] 真实窗口端到端（原生平台）")
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    session_store.DEFAULT_DB_PATH = db
    camera = install_fake_camera()

    import main
    from app.view.course_name_dialog import CourseNameDialog

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = main.MainWindow()
    w.show()
    m = w.monitorInterface
    QTest.qWait(500)

    # 校准期内就终了：不留空记录、不弹课程名
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(1500)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(1000)
    check("校准期内终了不留记录、不弹对话框",
          rows(db, "SELECT * FROM sessions") == [] and not any(d.isVisible() for d in w.findChildren(CourseNameDialog)))

    timeline = {}
    QTest.mouseClick(m.startBtn, 1)
    timeline["start"] = time.time()
    QTest.qWait(7000)
    camera.show_face = False
    QTest.qWait(4000)                       # 离席 4 秒
    camera.show_face = True
    QTest.qWait(3000)
    QTest.mouseClick(m.pauseBtn, 1)
    timeline["pause"] = time.time()
    QTest.qWait(1000)
    check("暂停时显示「一時停止中」", m.scoreLabel.text() == "一時停止中" and m.videoLabel.text() == "一時停止中")
    QTest.qWait(4000)                       # 暂停共约 5 秒
    QTest.mouseClick(m.pauseBtn, 1)
    timeline["resume"] = time.time()
    QTest.qWait(6000)

    dialog_seen = {}

    def fill_dialog():
        dialogs = w.findChildren(CourseNameDialog)
        dialog_seen["count"] = len(dialogs)
        if dialogs:
            d = dialogs[0]
            dialog_seen["visible"] = d.isVisible()
            # QTest.keyClicks 不支持非 ASCII（日语需经输入法），直接写入输入框，按钮仍真实点击
            d.nameBox.setText("線形代数 第1回")
            QTest.mouseClick(d.yesButton, 1)

    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    timeline["stop"] = time.time()
    QTest.qWait(2500)
    check("终了后弹出课程名对话框并可输入保存", dialog_seen.get("count") == 1 and dialog_seen.get("visible"), str(dialog_seen))
    check("终了后「カメラ停止中」未被最后一帧覆盖", m.videoLabel.pixmap() is None or m.videoLabel.pixmap().isNull(),
          repr(m.videoLabel.text()))
    check("终了后状态为「停止しました」", m.scoreLabel.text() == "停止しました")

    # 第二节课：暂停中直接关闭窗口
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(5500)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(500)
    w.close()
    QTest.qWait(500)
    check("关闭窗口时没有弹出课程名对话框", not w.findChildren(CourseNameDialog) or
          all(not d.isVisible() for d in w.findChildren(CourseNameDialog)))

    sessions = rows(db, "SELECT * FROM sessions ORDER BY id")
    check("共两条课程记录且都正常结束", [s["status"] for s in sessions] == ["completed", "completed"],
          str([(s["id"], s["status"], s["course_name"]) for s in sessions]))
    first, second = sessions
    check("第一节课的课程名已保存", first["course_name"] == "線形代数 第1回")
    check("第二节课（关闭窗口）课程名为空", second["course_name"] is None)

    events = rows(db, "SELECT * FROM events WHERE session_id = ? AND kind = 'spatial'", first["id"])
    away = [e for e in events if e["type"] == "away"]
    check("离席 4 秒被记为 away 事件", len(away) == 1 and 3.0 <= away[0]["t_end"] - away[0]["t_start"] <= 5.0,
          str([(e["type"], round(e["t_start"], 1), round(e["t_end"], 1)) for e in events]))
    pauses = rows(db, "SELECT * FROM pauses WHERE session_id = ?", first["id"])
    check("第一节课 1 次暂停且已继续", len(pauses) == 1 and pauses[0]["resumed_at"] is not None)
    p2 = rows(db, "SELECT * FROM pauses WHERE session_id = ?", second["id"])
    check("第二节课的暂停未继续（关闭窗口时仍在暂停）", len(p2) == 1 and p2[0]["resumed_at"] is None)

    print("\n[2] 时间轴核算")
    samples = rows(db, "SELECT t, wall_time FROM samples WHERE session_id = ? ORDER BY t", first["id"])
    ts = [s["t"] for s in samples]
    check("时间轴从 0 开始连续", ts == list(range(len(ts))), f"{len(ts)} 秒")
    pause_len = pauses[0]["resumed_at"] - pauses[0]["paused_at"]
    expected = (timeline["stop"] - timeline["start"]) - 3.0 - pause_len
    # 下限：模型加载与重新打开摄像头的延迟；上限：每段采集末尾不足一秒的部分也写成一行（ADR 0002），
    # 两段采集最多多出 2 行
    check("记录秒数 ≈ 墙上时长 − 校准 − 暂停", expected - 4.0 <= len(ts) <= expected + 2.5,
          f"记录 {len(ts)} 秒，预期约 {expected:.1f} 秒（暂停 {pause_len:.1f} 秒）")
    gap = [b["wall_time"] - a["wall_time"] for a, b in zip(samples, samples[1:])]
    check("暂停只在墙上时间留下一次大间隔", sum(g > 2.5 for g in gap) == 1, f"最大间隔 {max(gap):.1f} 秒")


# ------------------------------------------------------------------ 3. 真实崩溃
def run_app_child(db: Path, auto_start: bool):
    """子进程：运行真实 main.py（假摄像头 + 临时数据库），可选自动点「開始」"""
    code = f"""
import os, runpy, sys, time
sys.path.insert(0, {str(ROOT)!r}); os.chdir({str(ROOT)!r})
sys.path.insert(0, {str(Path(__file__).parent)!r})
import verify_issue4_recording as v
from app import session_store
session_store.DEFAULT_DB_PATH = __import__("pathlib").Path({str(db)!r})
v.install_fake_camera()
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
_exec = QApplication.exec
def exec_hook(self):
    w = [x for x in self.topLevelWidgets() if type(x).__name__ == "MainWindow"][0]
    if {auto_start}:
        QTimer.singleShot(500, w.monitorInterface.startBtn.click)
    else:
        QTimer.singleShot(1500, self.quit)
    print("READY", flush=True)
    return _exec()  # QApplication.exec 是静态方法
QApplication.exec = exec_hook
runpy.run_path("main.py", run_name="__main__")
"""
    log = open(db.with_suffix(".child.log"), "a")
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=log, text=True)


def real_crash(db: Path):
    print("\n[3] 真实崩溃：程序记录中被 SIGKILL，再次启动")
    proc = run_app_child(db, auto_start=True)
    ready = any(line.strip() == "READY" for line in iter(proc.stdout.readline, ""))  # 之前有模型加载等输出
    time.sleep(10)  # 启动 + 3 秒校准 + 约 6 秒记录
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    check("程序启动并被 SIGKILL 终止", ready and proc.returncode == -signal.SIGKILL)
    before = rows(db, "SELECT * FROM sessions")
    n_samples = rows(db, "SELECT COUNT(*) AS n FROM samples")[0]["n"]
    check("崩溃时记录仍为 recording，且已有数据", len(before) == 1 and before[0]["status"] == "recording" and n_samples >= 3,
          f"{n_samples} 秒")

    proc = run_app_child(db, auto_start=False)
    proc.wait(timeout=60)
    after = rows(db, "SELECT * FROM sessions")
    check("再次启动后标记为 interrupted", after[0]["status"] == "interrupted" and after[0]["ended_at"] is not None)
    check("再次启动没有丢失数据", rows(db, "SELECT COUNT(*) AS n FROM samples")[0]["n"] == n_samples)
    conn = sqlite3.connect(str(db))
    check("数据库完整性检查通过", conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
    conn.close()


# ------------------------------------------------------------------ 人工：真实摄像头
def real_camera():
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies
    from app import session_store
    import main

    db = Path(tempfile.mkdtemp()) / "real_camera.db"
    session_store.DEFAULT_DB_PATH = db
    app = QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = main.MainWindow()
    w.show()
    print(f"请操作：開始 → 一時停止 / 再開 → 終了（填写课程名），然后关闭窗口。数据库：{db}")
    app.exec()
    for s in rows(db, "SELECT * FROM sessions"):
        n = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", s["id"])[0]["n"]
        ev = rows(db, "SELECT type, COUNT(*) AS n FROM events WHERE session_id = ? GROUP BY type", s["id"])
        print(f"课程记录 {s['id']}：{s['status']}，课程名 {s['course_name']!r}，{n} 秒，事件 {ev}")


if __name__ == "__main__":
    if "--real-camera" in sys.argv:
        real_camera()
        sys.exit(0)
    with tempfile.TemporaryDirectory() as d:
        end_to_end(Path(d) / "e2e.db")
        real_crash(Path(d) / "crash.db")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)
