"""
#3 课程记录存储层的专项验证（补充 tests/test_session_store.py，覆盖 pytest 不便模拟的情况）

运行：在仓库根目录执行
    python test-scripts/verify_issue3_storage.py

检查项：
  1. 场景模拟：真实的 AttentionAnalyzer 跑一节含暂停 / 低头 / 离席 / 离开超 120 秒的课，
     用独立重算的方式核对数据库中的每秒汇总、事件、暂停与时间轴
  2. 真实崩溃：子进程按真实时间写入，被 SIGKILL 强杀后检查数据库完整性与丢失量
  3. 并发：记录线程写入的同时，界面线程用独立连接读取和修改其他记录
  4. 性能与体积：90 分钟课程的写入耗时与数据库大小
所有数据写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app import attention_rules, session_store  # noqa: E402
from app.attention_rules import AnalysisResult, AttentionAnalyzer  # noqa: E402
from app.session_recorder import SessionRecorder  # noqa: E402

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def synthetic(spatial="normal", cognitive="Neutral", score=80):
    return AnalysisResult(score=score, status_text="", alert=None, has_face=True, cognitive_state=cognitive,
                          fatigue_index=0.1, spatial_state=spatial, delta_pitch=0.0, delta_yaw=0.0)


# ------------------------------------------------------------------ 1. 场景模拟
def scenario(tmp: Path):
    print("\n[1] 场景模拟：真实分析器 + 记录器，15 fps")
    db = tmp / "scenario.db"
    conn = session_store.connect(db)
    clock = [1_700_000_000.0]
    # attention_rules.time 就是全局 time 模块：必须保存原函数并在结束时恢复
    real_time = time.time
    attention_rules.time.time = lambda: clock[0]
    try:
        _scenario(db, conn, clock)
    finally:
        attention_rules.time.time = real_time
    conn.close()


def _scenario(db, conn, clock):
    fps = 15
    dt = 1.0 / fps
    neutral = np.zeros(7); neutral[4] = 1.0
    doubt = np.zeros(7); doubt[0] = 0.6; doubt[5] = 0.4  # Anger + Sad → Doubt

    # (时长秒, 帧输入)；None 表示暂停这么久
    plan = [
        (40, (neutral, 0.3, 0.1, 0.0, 0.0)),
        (20, (doubt, 0.3, 0.1, 0.0, 0.0)),
        (18, (neutral, 0.3, 0.1, -30.0, 0.0)),   # 低头 18 秒 → head_down → taking_notes
        (10, (neutral, 0.3, 0.1, 0.0, 0.0)),
        ("pause", 300),                           # 暂停 5 分钟
        (30, (neutral, 0.3, 0.1, 0.0, 0.0)),
        (130, (None, None, None, None, None)),    # 离开 130 秒 → away → absent
        (15, (neutral, 0.3, 0.1, 0.0, 0.0)),
    ]

    session_id = session_store.create_session(conn, clock[0])
    analyzer = AttentionAnalyzer()
    recorder = SessionRecorder(session_id, clock[0], db)
    log = []  # 独立记录每帧 (听课时间, 结果)，用于重算
    seg_base, seg_start = 0.0, clock[0]  # 听课时间 = 本段起点 + (墙上时间 - 本段开始时刻)
    alerts = 0
    for step in plan:
        if step[0] == "pause":
            paused_t = seg_base + (clock[0] - seg_start)
            recorder.pause(clock[0])
            clock[0] += step[1]
            seg_base, seg_start = float(math.ceil(paused_t)), clock[0]  # ADR 0002：继续时从下一整秒开始
            analyzer = AttentionAnalyzer()  # 与 CameraThread.resume 一致：窗口清空
            recorder = SessionRecorder(session_id, clock[0], db)
            continue
        seconds, inputs = step
        for _ in range(int(seconds * fps)):
            r = analyzer.process_frame(*inputs)
            recorder.add_frame(r, clock[0])
            log.append((seg_base + (clock[0] - seg_start), r))
            alerts += r.alert is not None
            clock[0] += dt
    recorder.close(clock[0])
    session_store.finish_session(conn, session_id, clock[0])

    rows = [dict(r) for r in conn.execute("SELECT * FROM samples WHERE session_id = ? ORDER BY t", (session_id,))]
    ts = [r["t"] for r in rows]
    check("时间轴连续无空洞、无重复", ts == list(range(ts[0], ts[-1] + 1)) and ts[0] == 0, f"t=0..{ts[-1]}，共 {len(ts)} 秒")
    check("帧数守恒", sum(r["frame_count"] for r in rows) == len(log), f"{len(log)} 帧")

    buckets = defaultdict(list)
    for t, r in log:
        buckets[int(t)].append(r)
    mismatches = []
    for row in rows:
        frames = buckets[row["t"]]
        states = [f.cognitive_state for f in frames if f.cognitive_state]
        expect = {
            "score": sum(f.score for f in frames) / len(frames),
            "fatigue_index": sum(f.fatigue_index for f in frames) / len(frames),
            "cognitive_state": Counter(states).most_common(1)[0][0] if states else None,
            "spatial_state": frames[-1].spatial_state,
            "face_ratio": sum(f.has_face for f in frames) / len(frames),
        }
        for k, v in expect.items():
            ok = math.isclose(row[k], v, abs_tol=1e-9) if isinstance(v, float) else row[k] == v
            if not ok:
                mismatches.append((row["t"], k, row[k], v))
    check("每秒汇总与独立重算一致", not mismatches, f"{len(mismatches)} 处不一致 {mismatches[:3]}")

    states_seen = {r["spatial_state"] for r in rows}
    check("场景中的空间状态都被记录", {"normal", "head_down", "taking_notes", "away", "absent"} <= states_seen,
          ", ".join(sorted(states_seen)))
    check("认知状态出现 Doubt", any(r["cognitive_state"] == "Doubt" for r in rows))

    events = [dict(r) for r in conn.execute(
        "SELECT * FROM events WHERE session_id = ? AND kind = 'spatial' ORDER BY t_start", (session_id,))]
    closed = all(e["t_end"] is not None and e["t_end"] >= e["t_start"] for e in events)
    no_overlap = all(a["t_end"] <= b["t_start"] + 1e-9 for a, b in zip(events, events[1:]))
    check("空间事件全部关闭且互不重叠", closed and no_overlap, f"{len(events)} 个事件")
    consistent = all(
        r["spatial_state"] == e["type"]
        for e in events for r in rows
        if e["t_start"] + 1 <= r["t"] and r["t"] + 1 <= e["t_end"]  # 完全落在事件内部的秒
    )
    check("事件区间内的每秒空间状态与事件类型一致", consistent)
    n_alerts = conn.execute("SELECT COUNT(*) FROM events WHERE session_id = ? AND kind = 'alert'", (session_id,)).fetchone()[0]
    check("警报事件数与分析器发出的警报数一致", n_alerts == alerts and alerts > 0, f"{alerts} 次")

    pauses = [dict(r) for r in conn.execute("SELECT * FROM pauses WHERE session_id = ?", (session_id,))]
    check("暂停记录完整", len(pauses) == 1 and pauses[0]["resumed_at"] is not None
          and pauses[0]["resumed_at"] - pauses[0]["paused_at"] == 300)
    status = conn.execute("SELECT status FROM sessions WHERE id = ?", (session_id,)).fetchone()[0]
    check("正常结束后状态为 completed", status == "completed")


# ------------------------------------------------------------------ 2. 真实崩溃
CHILD = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from app import session_store
from app.attention_rules import AnalysisResult
from app.session_recorder import SessionRecorder
db = sys.argv[2]
conn = session_store.connect(db)
sid = session_store.create_session(conn, time.time())
rec = SessionRecorder(sid, time.time(), db)
print("READY", flush=True)
while True:
    rec.add_frame(AnalysisResult(80, "", None, True, "Neutral", 0.1, "away", 0.0, 0.0), time.time())
    time.sleep(1 / 30)
"""


def real_crash(tmp: Path):
    print("\n[2] 真实崩溃：子进程写入 3.5 秒后 SIGKILL")
    db = tmp / "crash.db"
    proc = subprocess.Popen([sys.executable, "-c", CHILD, str(ROOT), str(db)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "READY"
    start = time.time()
    time.sleep(3.5)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    elapsed = time.time() - start
    check("子进程被 SIGKILL 终止", proc.returncode == -signal.SIGKILL)

    conn = session_store.connect(db)
    check("数据库完整性检查通过", conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
    ts = [r[0] for r in conn.execute("SELECT t FROM samples ORDER BY t")]
    check("崩溃前已完成的秒都在，最多丢失约 1 秒", ts == list(range(len(ts))) and len(ts) >= math.floor(elapsed) - 1,
          f"运行约 {elapsed:.1f} 秒，保留 {len(ts)} 秒")
    open_events = conn.execute("SELECT COUNT(*) FROM events WHERE t_end IS NULL").fetchone()[0]
    check("崩溃时有未关闭的事件", open_events == 1)
    check("下次启动标记 1 条 interrupted", session_store.mark_interrupted_sessions(conn) == 1)
    row = conn.execute("SELECT status FROM sessions").fetchone()
    t_end = conn.execute("SELECT t_end FROM events").fetchone()[0]
    check("遗留事件在最后一条样本末尾关闭", row[0] == "interrupted" and t_end == ts[-1] + 1)
    conn.close()


# ------------------------------------------------------------------ 3. 并发读写
def concurrency(tmp: Path):
    print("\n[3] 并发：记录线程 30 fps 写入，同时另一个连接读取与修改")
    db = tmp / "concurrent.db"
    conn = session_store.connect(db)
    other = session_store.create_session(conn, time.time())
    session_store.finish_session(conn, other, time.time())
    sid = session_store.create_session(conn, time.time())
    errors = []
    stop = threading.Event()

    def writer():
        try:
            rec = SessionRecorder(sid, time.time(), db)
            while not stop.is_set():
                rec.add_frame(synthetic(), time.time())
                time.sleep(1 / 30)
            rec.close(time.time())
        except Exception as e:  # noqa: BLE001
            errors.append(f"writer: {e!r}")

    th = threading.Thread(target=writer)
    th.start()
    reads = 0
    deadline = time.time() + 3.0
    try:
        while time.time() < deadline:
            conn.execute("SELECT COUNT(*), AVG(score) FROM samples WHERE session_id = ?", (sid,)).fetchone()
            with conn:
                conn.execute("UPDATE sessions SET course_name = ? WHERE id = ?", (f"テスト{reads}", other))
            reads += 1
            time.sleep(0.01)
    except Exception as e:  # noqa: BLE001
        errors.append(f"reader: {e!r}")
    stop.set()
    th.join()
    n = conn.execute("SELECT COUNT(*) FROM samples WHERE session_id = ?", (sid,)).fetchone()[0]
    check("读写同时进行无 database is locked 等错误", not errors, "; ".join(errors))
    check("写入线程的数据完整", n >= 2, f"{n} 秒，读取/修改 {reads} 次")
    conn.close()


# ------------------------------------------------------------------ 4. 性能与体积
def performance(tmp: Path):
    print("\n[4] 性能与体积：90 分钟、30 fps 的合成数据")
    db = tmp / "perf.db"
    conn = session_store.connect(db)
    sid = session_store.create_session(conn, 0.0)
    rec = SessionRecorder(sid, 0.0, db)
    fps, minutes = 30, 90
    frame_costs, commit_costs = [], []
    states = ["normal"] * 50 + ["head_down"] * 10 + ["away"] * 5
    for i in range(fps * 60 * minutes):
        r = synthetic(spatial=states[(i // fps) % len(states)])
        t0 = time.perf_counter()
        rec.add_frame(r, i / fps)
        cost = time.perf_counter() - t0
        (commit_costs if i % fps == 0 and i else frame_costs).append(cost)
    rec.close(fps * 60 * minutes / fps)
    conn.close()
    size_mb = sum(p.stat().st_size for p in tmp.glob("perf.db*")) / 1e6
    p99_commit = sorted(commit_costs)[int(len(commit_costs) * 0.99)] * 1000
    check("普通帧平均耗时 < 200 µs", np.mean(frame_costs) * 1e6 < 200, f"{np.mean(frame_costs) * 1e6:.2f} µs")
    check("每秒提交的 p99 耗时 < 20 ms（30 fps 一帧约 33 ms）", p99_commit < 20, f"{p99_commit:.2f} ms")
    check("90 分钟数据库 < 5 MB", size_mb < 5, f"{size_mb:.2f} MB，{minutes * 60} 行样本")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        scenario(tmp)
        real_crash(tmp)
        concurrency(tmp)
        performance(tmp)
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)
