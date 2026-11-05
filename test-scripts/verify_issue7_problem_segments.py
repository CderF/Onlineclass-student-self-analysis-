"""
#7 问题时段判定与高亮的专项验证（补充 tests/test_problem_segments.py、tests/test_session_chart.py）

运行：在仓库根目录执行（会短暂弹出真实窗口）
    python test-scripts/verify_issue7_problem_segments.py

检查项：
  1. 独立参考实现对照：用逐秒直接数窗口的朴素写法（不用前缀和、按 t 而非下标）重新实现规则，
     在 300 节随机课程（随机暂停、时间轴空缺、事件、仍在进行的事件）上与 detect() 逐一比对，
     并检查不变量：同类时段不重叠、不跨暂停、不超出时间轴、不短于最短时长
  2. 性能：90 分钟与 3 小时课程的判定耗时
  3. 真实记录：原生窗口用假摄像头记录一节课并离席约 13 秒，报告页「注意散漫・離席」行出现对应时段
  4. 像素核对：60 分钟示例课程截图，在每个问题时段中点核对问题时段行的颜色与分数曲线的底色，
     时段之外没有底色；真实 Qt 鼠标悬停显示「問題：…」
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

from app.problem_segments import DEFAULT_CONFIG, DISTRACTION, DOUBT, FATIGUE, detect  # noqa: E402

failures = []
COLORS = {DOUBT: (0xEB, 0x68, 0x34), FATIGUE: (0x62, 0x50, 0xD6), DISTRACTION: (0x1B, 0xAF, 0x7A)}
LABELS = {DOUBT: "理解不足の可能性", FATIGUE: "疲労", DISTRACTION: "注意散漫・離席"}


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------ 1. 参考实现
def reference(samples, events, pauses, cfg=DEFAULT_CONFIG):
    """与 app/problem_segments.py 独立的朴素实现：逐秒按 t 直接数窗口"""
    bounds = sorted({math.floor(p["t"]) + 1 for p in pauses})
    by_t = {s["t"]: s for s in samples}
    end = samples[-1]["t"] + 1 if samples else 0

    # 段号 = (之前经过的暂停边界数, 之前的时间轴空缺数)
    seg, gaps = {}, 0
    for i, s in enumerate(samples):
        if i and s["t"] != samples[i - 1]["t"] + 1:
            gaps += 1
        seg[s["t"]] = (sum(1 for b in bounds if b <= s["t"]), gaps)

    def window(t, length):
        lo, hi = t - length // 2, t - length // 2 + length
        return [by_t[u] for u in range(lo, hi) if u in by_t and seg[u] == seg[t]]

    def marked(pred):
        out, start, prev = [], None, None
        for s in samples:
            t = s["t"]
            hit = pred(t)
            if start is not None and (not hit or seg[t] != seg[prev] or t != prev + 1):
                out.append((float(start), float(prev + 1)))
                start = None
            if hit and start is None:
                start = t
            prev = t
        if start is not None:
            out.append((float(start), float(prev + 1)))
        return out

    def doubt(t):
        w = window(t, cfg.doubt_window)
        valid = [s for s in w if s["cognitive_state"] is not None]
        return len(valid) >= cfg.doubt_min_valid and sum(s["cognitive_state"] == "Doubt" for s in valid) / len(valid) >= cfg.doubt_ratio

    def fatigue(t):
        w = window(t, cfg.fatigue_window)
        return sum(s["fatigue_index"] for s in w) / len(w) >= cfg.fatigue_threshold

    dist = sorted((float(e["t_start"]), float(e["t_end"] if e["t_end"] is not None else max(end, e["t_start"])))
                  for e in events if e["kind"] == "spatial" and e["type"] in cfg.distraction_states)

    def merge(ivs):
        out = []
        for a, b in sorted(ivs):
            crosses = any(out and out[-1][1] <= x <= a for x in bounds)
            if out and a - out[-1][1] <= cfg.merge_gap and not crosses:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    result = []
    for cat, ivs in ((DOUBT, marked(doubt)), (FATIGUE, marked(fatigue)), (DISTRACTION, dist)):
        result += [{"category": cat, "start": a, "end": b} for a, b in merge(ivs) if b - a >= cfg.min_duration[cat]]
    order = [DOUBT, FATIGUE, DISTRACTION]
    return sorted(result, key=lambda p: (p["start"], order.index(p["category"])))


def random_lesson(rng):
    n = rng.randint(0, 900)
    samples, events, pauses = [], [], []
    t, cog, fat, open_event = 0, "Neutral", rng.random() * 0.5, None
    while len(samples) < n:
        r = rng.random()
        if r < 0.004 and samples:                      # 暂停：暂停点为最后一帧的时间（在最后一秒之内），继续后从下一秒开始
            p = samples[-1]["t"] + rng.choice([0.0, 0.3, 0.999])
            pauses.append({"t": p, "paused_at": 0.0, "resumed_at": 1.0})
            t = math.floor(p) + 1
        elif r < 0.006:                                # 时间轴空缺
            t += rng.randint(1, 20)
        if rng.random() < 0.03:
            cog = rng.choice(["Understand", "Neutral", "Doubt", "Doubt", "Disgust", None])
        if rng.random() < 0.02:
            fat = min(1.0, max(0.0, fat + rng.uniform(-0.4, 0.4)))
        samples.append({"t": t, "cognitive_state": cog, "fatigue_index": fat, "spatial_state": "normal"})
        t += 1
    end = samples[-1]["t"] + 1 if samples else 0
    pos = 0.0
    while samples and pos < end:
        pos += rng.uniform(1, 120)
        length = rng.choice([rng.uniform(0.2, 12), rng.uniform(5, 60)])
        if pos >= end:
            break
        state = rng.choice(["away", "absent", "distracted", "hidden_abnormal", "head_down", "taking_notes"])
        events.append({"kind": "spatial", "type": state, "t_start": round(pos, 3),
                       "t_end": round(min(pos + length, end), 3), "wall_start": 0.0})
        if rng.random() < 0.1:
            events.append({"kind": "alert", "type": "absence", "t_start": pos, "t_end": pos, "wall_start": 0.0})
        pos += length
    # 与真实记录器一致：暂停时关闭进行中的事件，继续后状态仍在则从下一整秒重新开始
    split = []
    for e in events:
        for p in pauses:
            if e["kind"] == "spatial" and e["t_start"] < p["t"] < e["t_end"]:
                split.append({**e, "t_end": p["t"]})
                e = {**e, "t_start": float(math.floor(p["t"]) + 1)}
            elif p["t"] <= e["t_start"] < math.floor(p["t"]) + 1:  # 暂停点到下一秒之间不在时间轴上
                e = {**e, "t_start": float(math.floor(p["t"]) + 1)}
                if e["kind"] == "alert":
                    e["t_end"] = e["t_start"]
        if e["kind"] != "spatial" or e["t_start"] < e["t_end"]:
            split.append(e)
    events = split
    last_resume = max((math.floor(p["t"]) + 1 for p in pauses), default=0)
    if events and events[-1]["t_start"] >= last_resume and rng.random() < 0.2:
        events[-1]["t_end"] = None                     # 记录中仍在进行（只可能在最后一次继续之后）
    return samples, events, pauses


def invariants(result, samples, pauses):
    bounds = {math.floor(p["t"]) + 1 for p in pauses}
    end = samples[-1]["t"] + 1 if samples else 0
    for cat in (DOUBT, FATIGUE, DISTRACTION):
        ivs = sorted((p["start"], p["end"]) for p in result if p["category"] == cat)
        if any(b - a < DEFAULT_CONFIG.min_duration[cat] for a, b in ivs):
            return f"{cat} 短于最短时长"
        if any(ivs[i][1] > ivs[i + 1][0] for i in range(len(ivs) - 1)):
            return f"{cat} 重叠"
        if any(a < 0 or b > end + 1e-9 for a, b in ivs):
            return f"{cat} 超出时间轴"
        if any(a < x < b for a, b in ivs for x in bounds):
            return f"{cat} 跨暂停"
    return None


def fuzz():
    print("\n[1] 独立参考实现对照（300 节随机课程）")
    rng = random.Random(7)
    mismatches, broken, total = [], [], 0
    for i in range(300):
        samples, events, pauses = random_lesson(rng)
        got = detect(samples, events, pauses)
        want = reference(samples, events, pauses)
        total += len(got)
        if got != want:
            mismatches.append(i)
        bad = invariants(got, samples, pauses)
        if bad:
            broken.append((i, bad))
    check("detect() 与参考实现完全一致", not mismatches, f"共 {total} 个时段" + (f"，不一致的课程 {mismatches[:10]}" if mismatches else ""))
    check("不变量：不重叠、不跨暂停、不超出时间轴、不短于最短时长", not broken, str(broken[:5]))


def performance():
    print("\n[2] 性能")
    rng = random.Random(70)
    for minutes in (90, 180):
        samples = [{"t": t, "cognitive_state": rng.choice(["Doubt", "Neutral", None]), "fatigue_index": rng.random(),
                    "spatial_state": "normal"} for t in range(minutes * 60)]
        events = [{"kind": "spatial", "type": "away", "t_start": float(t), "t_end": float(t + 15), "wall_start": 0.0}
                  for t in range(0, minutes * 60, 200)]
        start = time.perf_counter()
        detect(samples, events, [{"t": 1800.5, "paused_at": 0, "resumed_at": 1}])
        ms = (time.perf_counter() - start) * 1000
        check(f"{minutes} 分钟课程判定 < 100 ms", ms < 100, f"{ms:.1f} ms")


# ------------------------------------------------------------------ 3–4. 真实窗口
def seed(db: Path):
    """60 分钟：疑問 10–15 分、疲劳 25–31 分、离席 40–41 分、注意散漫 50–51:40；20 分处暂停 2 分钟"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    rng = random.Random(77)
    conn = session_store.connect(db)
    start = time.time() - 86400
    sid = session_store.create_session(conn, start)
    rec, wall, t = SessionRecorder(sid, start, db), start, 0.0
    paused = False
    while t < 3600:
        if t == 1200.0 and not paused:
            rec.pause(wall)
            wall += 120
            rec = SessionRecorder(sid, wall, db)
            paused = True
        spatial = "away" if 2400 <= t < 2460 else ("distracted" if 3000 <= t < 3100 else "normal")
        no_face = spatial == "away"
        cog = None if no_face else ("Doubt" if 600 <= t < 900 else rng.choice(["Understand", "Neutral", "Neutral"]))
        rec.add_frame(AnalysisResult(
            score=int(25 if no_face else 60 + 10 * math.sin(t / 300) + rng.uniform(-3, 3)), status_text="", alert=None,
            has_face=not no_face, cognitive_state=cog, fatigue_index=0.6 if 1500 <= t < 1860 else 0.1,
            spatial_state=spatial, delta_pitch=0.0, delta_yaw=0.0,
        ), wall)
        wall += 0.5
        t += 0.5
    rec.close(wall)
    session_store.finish_session(conn, sid, wall)
    session_store.set_course_name(conn, sid, "統計学 第2回")
    conn.close()
    return sid


def in_window(db: Path, shot_dir: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import report_data, session_store
    session_store.DEFAULT_DB_PATH = db
    long_id = seed(db)

    import main as app_main
    import verify_issue4_recording as v4
    from verify_issue5_report import rows, run_js, switch_to_report, wait_state
    from verify_issue6_session_chart import chart_point, close_to, mouse_move, option, pixel
    from app.view.course_name_dialog import CourseNameDialog
    camera = v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 950)
    w.show()
    report, m = w.reportInterface, w.monitorInterface
    QTest.qWait(1500)

    print("\n[3] 真实记录：离席约 13 秒")
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(9000)                     # 模型加载 + 3 秒校准 + 约 5 秒记录
    camera.show_face = False
    QTest.qWait(13000)
    camera.show_face = True
    QTest.qWait(3000)

    def fill_dialog():
        for dialog in w.findChildren(CourseNameDialog):
            if dialog.isVisible():
                dialog.nameBox.setText("英語 第4回")
                QTest.mouseClick(dialog.yesButton, 1)

    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    switch_to_report(w)
    wait_state(report.page)
    QTest.qWait(500)
    rec_id = rows(db, "SELECT id FROM sessions WHERE course_name = '英語 第4回'")[0]["id"]
    away = rows(db, "SELECT t_start, t_end FROM events WHERE session_id = ? AND kind = 'spatial' AND type = 'away'", rec_id)
    conn = session_store.connect(db)
    problems = report_data.load_session(conn, rec_id)["problems"]
    conn.close()
    dist = [(p["start"], p["end"]) for p in problems if p["category"] == DISTRACTION]
    check("离席事件约 13 秒", len(away) == 1 and 11 <= away[0]["t_end"] - away[0]["t_start"] <= 15, str(away))
    check("离席被判为「注意散漫・離席」时段，区间与事件相同",
          len(away) == 1 and dist == [(away[0]["t_start"], away[0]["t_end"])], str(dist))
    opt = option(report.page)
    bar = [x for x in opt["series"] if x["name"] == LABELS[DISTRACTION] and x["type"] == "custom"][0]["data"]
    check("报告页问题时段行显示该时段",
          [(round(d[0] * 60, 3), round(d[1] * 60, 3)) for d in bar] == [(round(a, 3), round(b, 3)) for a, b in dist])
    w.grab().save(str(shot_dir / "recorded_with_problem.png"))

    print("\n[4] 60 分钟示例课程：像素核对与悬停")
    w.switchTo(m)
    QTest.qWait(300)
    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select');
        s.value = '{long_id}';
        s.dispatchEvent(new Event('change'));
    }})()""")
    QTest.qWait(500)
    switch_to_report(w)
    wait_state(report.page)
    QTest.qWait(500)
    conn = session_store.connect(db)
    problems = report_data.load_session(conn, long_id)["problems"]
    conn.close()
    cats = sorted(p["category"] for p in problems)
    check("示例课程检出 4 个时段（疑問、疲劳、离席、注意散漫）",
          cats == sorted([DOUBT, FATIGUE, DISTRACTION, DISTRACTION]),
          ", ".join(f"{LABELS[p['category']]} {p['start']:.0f}–{p['end']:.0f}s" for p in problems))

    view = report.webView
    rows_pt = {i: chart_point(report.page, 4, (p["start"] + p["end"]) / 2 / 60, LABELS[p["category"]]) for i, p in enumerate(problems)}
    shade_pt = {i: chart_point(report.page, 0, (p["start"] + p["end"]) / 2 / 60, 97) for i, p in enumerate(problems)}
    clear_pt = chart_point(report.page, 0, 5, 97)        # 5 分处没有问题时段
    image = w.grab().toImage()
    image.save(str(shot_dir / "problems_60min.png"))

    def tinted(c, alpha=0.12):
        return tuple(round(255 - alpha * (255 - v)) for v in c)

    wrong_row = {LABELS[problems[i]["category"]]: pixel(image, view, pt) for i, pt in rows_pt.items()
                 if not close_to(pixel(image, view, pt), COLORS[problems[i]["category"]])}
    check("问题时段行：每个时段在对应行、以对应颜色画出", not wrong_row, str(wrong_row))
    wrong_shade = {LABELS[problems[i]["category"]]: (pixel(image, view, pt), tinted(COLORS[problems[i]["category"]]))
                   for i, pt in shade_pt.items()
                   if not close_to(pixel(image, view, pt), tinted(COLORS[problems[i]["category"]]), tol=10)}
    check("分数曲线背后有对应颜色的浅底色", not wrong_shade, str(wrong_shade))
    clear = pixel(image, view, clear_pt)
    check("问题时段之外没有底色", close_to(clear, (255, 255, 255), tol=6), str(clear))

    target = view.focusProxy() or view
    doubt = [p for p in problems if p["category"] == DOUBT][0]
    mouse_move(target, chart_point(report.page, 4, (doubt["start"] + doubt["end"]) / 2 / 60, LABELS[DOUBT]))
    QTest.qWait(600)
    tip = run_js(report.page, "document.getElementById('score-chart').innerText")
    check("鼠标悬停在问题时段行显示「問題：理解不足の可能性」", "問題：理解不足の可能性" in tip, tip.replace("\n", " / ")[:140])

    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    check("没有被拦截的请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    w.close()
    QTest.qWait(500)


if __name__ == "__main__":
    fuzz()
    performance()
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue7_"))
    with tempfile.TemporaryDirectory() as d:
        in_window(Path(d) / "problems.db", shot_dir)
    print(f"\n截图保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
