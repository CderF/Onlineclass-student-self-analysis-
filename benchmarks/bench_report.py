"""
课后打开报告页时的一次性开销（benchmarks/proposal.md 第 4 节第 4 项）。

运行：在仓库根目录执行（每次测量在新进程中进行，默认 5 次）
    python benchmarks/bench_report.py [--runs N] [--machine 名称] [--offscreen]

数据：在临时数据库中用真实的 SessionRecorder 写入 5 节同课程名、各 90 分钟的合成课程记录，
测量对象是最后一节（含一次暂停、疑问集中、疲劳、低头、离席与警报，三类问题时段都会出现）；
前 4 节用于文字反馈的「前回との比較」与多节课汇总。
测量项目：
  纯 Python（界面线程之外的后台线程池里执行的部分）：
    load_session（读库 + 问题时段判定 + 文字反馈）、其中的 detect 与 feedback、summarize_sessions、
    单节课 CSV 导出、全部 CSV 导出
  报告页（QWebEngineView + ECharts）：
    首次打开到显示完成（含 WebEngine 子进程启动）、切换到「全体の推移」到显示完成
每项记录墙上时间、CPU 时间（本进程 + 子进程 QtWebEngineProcess）与内存峰值。
结果写到 benchmarks/results/<机器>/report/run<k>.json。
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from common import RESULTS_DIR, ResourceSampler, default_machine_tag, machine_info, setup_path, write_json

MINUTES = 90
COURSE = "線形代数"
DAY = 86400


def seed(db: Path, rng: random.Random) -> int:
    """写入 4 节历史课 + 1 节测量对象（每秒 1 帧送入 SessionRecorder，与真实记录相同的表结构与事件）"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    conn = session_store.connect(db)
    start = time.time() - 10 * DAY

    def frame(i, minute, target):
        cog, fat, spatial, face, alert = rng.choice(["Understand", "Neutral", "Neutral"]), 0.1, "normal", True, None
        if target:
            if 10 <= minute < 14:
                cog = rng.choice(["Doubt", "Doubt", "Doubt", "Neutral"])          # 可能没听懂
            if 55 <= minute < 60:
                fat = 0.5                                                          # 疲劳
                alert = ("fatigue", "", "") if i % 600 == 0 else None
            if 20 * 60 <= i < 20 * 60 + 15:
                spatial = "head_down"
            elif 20 * 60 + 15 <= i < 20 * 60 + 40:
                spatial = "taking_notes"
            if 30 * 60 <= i < 31 * 60 + 30:
                spatial, face, cog = "away", False, None                           # 离席 90 秒
            if 70 * 60 <= i < 70 * 60 + 40:
                spatial = "distracted"
                alert = ("head_down", "", "") if i == 70 * 60 + 20 else None
        score = 25 if not face else int(rng.uniform(50, 90) - 30 * fat)
        return AnalysisResult(score=score, status_text="", alert=alert, has_face=face, cognitive_state=cog,
                              fatigue_index=fat, spatial_state=spatial,
                              delta_pitch=-20.0 if spatial != "normal" else 0.0, delta_yaw=0.0)

    target_id = None
    for k in range(5):
        target = k == 4
        st = start + k * DAY
        sid = session_store.create_session(conn, st)
        rec = SessionRecorder(sid, st, db)
        wall = st
        for i in range(MINUTES * 60):
            if target and i == 45 * 60:                                        # 第 45 分钟暂停 5 分钟
                rec.pause(wall)
                wall += 300
                rec = SessionRecorder(sid, wall, db)
            rec.add_frame(frame(i, i // 60, target), wall)
            wall += 1
        rec.close(wall)
        session_store.finish_session(conn, sid, wall)
        session_store.set_course_name(conn, sid, COURSE)
        target_id = sid
    conn.close()
    return target_id


def measure(sampler: ResourceSampler, func, *args):
    """纯 Python 操作：墙上时间、本线程 CPU 时间（单线程执行，不含采样线程）与区间内的内存峰值"""
    before = sampler.snapshot()["rss_mb"]
    t0, c0 = time.perf_counter(), time.thread_time()
    value = func(*args)
    t1, c1 = time.perf_counter(), time.thread_time()
    after = sampler.snapshot()["rss_mb"]
    peak = sampler.window(t0, t1).get("rss_peak_mb")
    return value, {"wall_ms": (t1 - t0) * 1000, "cpu_ms": (c1 - c0) * 1000,
                   "rss_peak_mb": max(x for x in (peak, before, after) if x is not None)}


def pure_python(db: Path, sid: int, sampler: ResourceSampler) -> dict:
    from app import export_csv, feedback, problem_segments, report_data, session_store

    conn = session_store.connect(db)
    out = {}
    data, out["load_session"] = measure(sampler, report_data.load_session, conn, sid)
    _, out["detect"] = measure(sampler, problem_segments.detect, data["samples"], data["events"], data["pauses"])
    _, out["feedback"] = measure(sampler, feedback.generate, conn, data["session"], data["events"], data["problems"])
    _, out["summarize_sessions"] = measure(sampler, report_data.summarize_sessions, conn)
    with tempfile.TemporaryDirectory() as tmp:
        _, out["export_one"] = measure(sampler, export_csv.export, conn, Path(tmp), [sid])
        _, out["export_all"] = measure(sampler, export_csv.export, conn, Path(tmp))
    conn.close()
    out["data"] = {"samples": len(data["samples"]), "events": len(data["events"]), "pauses": len(data["pauses"]),
                   "problems": sorted({p["category"] for p in data["problems"]}),
                   "feedback_comparison": bool(data["feedback"] and data["feedback"].get("comparison"))}
    return out


def page(db: Path, sid: int, sampler: ResourceSampler, args) -> dict:
    import PyQt5.QtWebEngineWidgets  # noqa: F401  必须在创建 QApplication 之前导入
    from PyQt5 import sip
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    from app import session_store
    session_store.DEFAULT_DB_PATH = db  # ReportBridge 读默认数据库
    from app.view.report_interface import ReportInterface

    qapp = QApplication.instance() or QApplication(sys.argv)
    QTest.qWait(300)

    def run_js(p, script, timeout=5.0):
        box = []
        p.runJavaScript(script, box.append)
        end = time.time() + timeout
        while not box and time.time() < end:
            QTest.qWait(1)
        return box[0] if box else None

    def until(p, expr, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            if run_js(p, expr):
                return True
            QTest.qWait(1)
        return False

    def painted(p):
        """两次 requestAnimationFrame 之后：setOption 之后的第一帧已经画到屏幕上"""
        run_js(p, "window.__painted = false;"
                  " requestAnimationFrame(() => requestAnimationFrame(() => { window.__painted = true; }))")
        return until(p, "window.__painted === true", 5.0)

    out = {"rss_before_page_mb": sampler.snapshot()["rss_mb"]}
    before = sampler.snapshot(refresh=True)
    t0 = time.perf_counter()
    r = ReportInterface()
    r.resize(1100, 800)
    r.show()
    ready = until(r.page, f"document.body && document.body.dataset.state === 'ready'"
                          f" && document.getElementById('session-select').value === '{sid}'")
    t_ready = time.perf_counter()
    shown = ready and painted(r.page)
    t1 = time.perf_counter()
    after = sampler.snapshot(refresh=True)
    out["open_page"] = {"ok": shown, "wall_ms": (t1 - t0) * 1000, "ready_ms": (t_ready - t0) * 1000,
                        "cpu_ms": (after["cpu_s"] - before["cpu_s"]) * 1000,
                        "rss_peak_mb": max(sampler.window(t0, t1).get("rss_peak_mb") or 0, after["rss_mb"]),
                        "processes": len(sampler._family(refresh=True))}

    QTest.qWait(1500)  # 等 ECharts 动画结束、页面空闲
    before = sampler.snapshot(refresh=True)
    t0 = time.perf_counter()
    run_js(r.page, "document.getElementById('tab-summary').click()")
    ok = until(r.page, "document.body.dataset.summary === 'ready'") and painted(r.page)
    t1 = time.perf_counter()
    after = sampler.snapshot(refresh=True)
    out["open_summary"] = {"ok": ok, "wall_ms": (t1 - t0) * 1000, "cpu_ms": (after["cpu_s"] - before["cpu_s"]) * 1000,
                           "rss_peak_mb": max(sampler.window(t0, t1).get("rss_peak_mb") or 0, after["rss_mb"])}
    QTest.qWait(1500)
    out["rss_idle_after_mb"] = sampler.snapshot()["rss_mb"]

    r.shutdown()  # 销毁前必须先调用（CLAUDE.md：否则与后台线程池死锁）
    r.close()
    QTest.qWait(200)
    sip.delete(r)
    return out


def run_once(args, out_dir, k: int) -> dict:
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    setup_path()
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "bench.db"
        t = time.perf_counter()
        sid = seed(db, random.Random(26))
        print(f"  合成 5 节 × {MINUTES} 分钟课程：{time.perf_counter() - t:.1f} 秒", flush=True)
        sampler = ResourceSampler(interval=0.1, children=True).start()
        result = {"run": k, "config": {"minutes": MINUTES, "sessions": 5, "offscreen": args.offscreen},
                  "machine": machine_info(), "rss_start_mb": sampler.snapshot()["rss_mb"]}
        result["python"] = pure_python(db, sid, sampler)
        result["page"] = page(db, sid, sampler, args)
        sampler.stop()
    if out_dir is not None:
        write_json(out_dir / f"run{k}.json", result)
    return result


def brief(r) -> str:
    p, g = r["python"], r["page"]
    return (f"  load_session {p['load_session']['wall_ms']:.0f} ms（detect {p['detect']['wall_ms']:.0f} ms、"
            f"feedback {p['feedback']['wall_ms']:.0f} ms），汇总 {p['summarize_sessions']['wall_ms']:.0f} ms，"
            f"单节导出 {p['export_one']['wall_ms']:.0f} ms；首次打开报告页 {g['open_page']['wall_ms']:.0f} ms"
            f"（{'OK' if g['open_page']['ok'] else '未完成'}，内存峰值 {g['open_page']['rss_peak_mb']:.0f} MB，"
            f"{g['open_page']['processes']} 个进程），切到汇总 {g['open_summary']['wall_ms']:.0f} ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="只跑 1 次，不保存结果")
    parser.add_argument("--child", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.smoke or args.child:
        out = None if args.smoke else Path(os.environ["BENCH_OUT_DIR"])
        result = run_once(args, out, args.child or 0)
        print(brief(result), flush=True)
        if args.smoke:
            print(json.dumps(result["python"]["data"], ensure_ascii=False))
        return 0 if result["page"]["open_page"]["ok"] and result["page"]["open_summary"]["ok"] else 1

    out_dir = RESULTS_DIR / (args.machine or default_machine_tag()) / "report"
    print(f"结果目录：{out_dir.relative_to(RESULTS_DIR.parent.parent)}")
    for k in range(1, args.runs + 1):
        print(f"第 {k}/{args.runs} 次…", flush=True)
        code = subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--child", str(k)],
                               env={**os.environ, "BENCH_OUT_DIR": str(out_dir)})
        if code != 0:
            print(f"❌ 第 {k} 次测量失败（退出码 {code}）")
            return code
    print("✅ 完成。汇总：python benchmarks/summarize.py" + (f" --machine {args.machine}" if args.machine else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
