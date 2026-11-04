"""
#15 暂停 / 继续后听课时间轴连续的专项验证（补充 tests/test_recording_flow.py、tests/test_session_store.py）

运行：在仓库根目录执行（无窗口，约 2 分钟）
    python test-scripts/verify_issue15_resume_timeline.py

检查项（真实 CameraThread + 假摄像头，随机时机）：
  6 节课，每节课在随机时刻暂停 / 继续 3 次，继续时摄像头首帧随机延迟 0–2 秒
  （包括首帧到来之前就再次暂停或终了），对每节课检查：
  1. 时间轴从 0 开始连续、无重复
  2. 每次继续后的第一秒 = floor(暂停点) + 1，且该秒的墙上时间不早于继续时刻
  3. 暂停点 = 暂停前最后一秒内最后一帧的听课时间（落在最后一条样本那一秒之内）
  4. 暂停时长（resumed_at − paused_at）包含等待首帧的时间
  5. 问题时段判定的切分点与实际继续点一致
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import math
import os
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def install_fake_camera():
    import cv2
    import ultralytics
    import app.camera_thread as ct

    face = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))[0:400, 800:1200]

    class FakeCapture:
        next_delay = 0.0  # 下一次打开时首帧前的等待

        def __init__(self, index):
            self.open = True
            self.first_delay = FakeCapture.next_delay

        def isOpened(self):
            return self.open

        def read(self):
            time.sleep(0.03 + self.first_delay)
            self.first_delay = 0.0
            return True, face.copy()

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


def run_session(db, camera, rng, log):
    from PyQt5.QtTest import QTest
    from app.camera_thread import CameraThread

    camera.next_delay = 0.0
    th = CameraThread(db_path=db)
    th.start()
    QTest.qWait(int((3.0 + rng.uniform(1.0, 3.0)) * 1000))
    plan = []
    for _ in range(3):
        th.pause()
        QTest.qWait(int(rng.uniform(0.2, 1.0) * 1000))
        delay = rng.choice([0.0, 0.6, 1.2, 2.0])
        run = rng.uniform(0.3, 2.5)          # 可能短于首帧延迟：首帧到来之前就再次暂停
        camera.next_delay = delay
        th.resume()
        QTest.qWait(int(run * 1000))
        plan.append((delay, round(run, 1)))
    th.stop()
    log.append((th.session_id, plan))
    return th.session_id


def verify(db, sid, plan):
    from app import report_data, session_store
    samples = rows(db, "SELECT t, wall_time FROM samples WHERE session_id = ? ORDER BY t", sid)
    pauses = rows(db, "SELECT t, paused_at, resumed_at FROM pauses WHERE session_id = ? ORDER BY id", sid)
    ts = [s["t"] for s in samples]
    problems = []

    if ts != list(range(len(ts))):
        problems.append(f"时间轴不连续：缺 {sorted(set(range(ts[-1] + 1)) - set(ts)) if ts else '全部'}")
    for p in pauses:
        before = [s for s in samples if s["t"] <= p["t"]]
        if not before or not (before[-1]["t"] <= p["t"] < before[-1]["t"] + 1):
            problems.append(f"暂停点 {p['t']:.3f} 不在最后一条样本那一秒之内")
        if p["resumed_at"] is None:
            continue
        after = [s for s in samples if s["t"] > p["t"]]
        if after:
            if after[0]["t"] != math.floor(p["t"]) + 1:
                problems.append(f"继续后第一秒 {after[0]['t']} ≠ floor({p['t']:.3f})+1")
            if after[0]["wall_time"] < p["resumed_at"] - 1e-6:
                problems.append("继续后第一秒早于继续时刻")
        if p["resumed_at"] - p["paused_at"] < 0:
            problems.append("暂停时长为负")
    conn = session_store.connect(db)
    data = report_data.load_session(conn, sid)
    conn.close()
    from app.problem_segments import _segments
    bounds = sorted({float(math.floor(p["t"]) + 1) for p in data["pauses"]})
    seg_starts = [seg[0]["t"] for seg in _segments(data["samples"], bounds)][1:]
    resumes = [math.floor(p["t"]) + 1 for p in pauses if any(s["t"] > p["t"] for s in samples)]
    if seg_starts != resumes:
        problems.append(f"问题时段切分点 {seg_starts} ≠ 继续点 {resumes}")
    return problems, len(ts), len(pauses)


def main():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841  必须保持引用，否则立即被回收
    camera = install_fake_camera()
    rng = random.Random(15)
    with tempfile.TemporaryDirectory() as d:
        db = Path(d) / "resume.db"
        log = []
        for _ in range(6):
            run_session(db, camera, rng, log)
        print()
        waits_counted = []
        for sid, plan in log:
            problems, n, n_pauses = verify(db, sid, plan)
            delays = ", ".join(f"首帧延迟 {dl}s/继续后 {r}s" for dl, r in plan)
            check(f"课程 {sid}：时间轴连续、继续点与暂停点一致", not problems,
                  f"{n} 秒、{n_pauses} 次暂停；{delays}" + (f"；问题：{problems}" if problems else ""))
            for p in rows(db, "SELECT paused_at, resumed_at FROM pauses WHERE session_id = ? AND resumed_at IS NOT NULL", sid):
                waits_counted.append(p["resumed_at"] - p["paused_at"])
        # 每节课第 2 段之后都有首帧延迟 ≥ 1.2 秒的情况：暂停时长应包含它
        long_waits = [w for w in waits_counted if w >= 1.4]
        check("等待首帧的时间计入暂停时长", long_waits, f"最长暂停 {max(waits_counted):.2f} 秒，≥1.4 秒的 {len(long_waits)} 次")
        check("数据库完整性检查通过", rows(db, "PRAGMA integrity_check")[0]["integrity_check"] == "ok")


if __name__ == "__main__":
    main()
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    os._exit(1 if failures else 0)  # 跳过解释器清理：PyQt 退出清理偶发 SIGSEGV（#14）
