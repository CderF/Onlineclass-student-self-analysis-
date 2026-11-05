"""
#9 规则生成的日语文字反馈的专项验证（补充 tests/test_feedback.py、tests/test_feedback_view.py）

运行：在仓库根目录执行（会短暂弹出真实窗口，约 1 分钟）
    python test-scripts/verify_issue9_feedback.py

检查项：
  1. 随机输入：2000 个随机 FeedbackContext，检查不变量（不报错、无 None / NaN / undefined、时段条数与排序、
     「ほか」计数、提示不重复且都来自模板、比较段落出现的条件、总评分档与阈值一致）
  2. 真实记录：原生窗口用假摄像头记录同一课程名的两节课（第二节离席约 13 秒），
     报告页第二节的反馈含「席を離れていた」与「前回との比較」，比较的数值与数据库一致
  3. 真实鼠标：点击「気になった時間帯」的条目，曲线缩放到该时段（前后各 30 秒）
  4. 性能：同课程名已有 60 节 90 分钟课程时，读取一节课（含反馈）的耗时
  5. 文本样例：把多种场景的反馈写到截图目录的 feedback_samples.txt，供人工审阅措辞
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import os
import random
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

from app import feedback  # noqa: E402
from app.feedback import DEFAULT_CONFIG, FeedbackContext, RuleFeedbackGenerator  # noqa: E402

failures = []
HEADS = ("全体としてよく集中できていました", "おおむね集中できていました", "集中が続きにくい授業だったようです")
TIME = r"\d+(時間\d+)?分"
SEGMENT_RE = re.compile(rf"^{TIME}〜{TIME}：.+の?(可能性があります|見られました)（(\d+秒|{TIME})）$")


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def rows(db, sql, *args):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


# ------------------------------------------------------------------ 1. 随机输入
def random_context(rng):
    seconds = rng.choice([30, 600, 3600, 5400, 9000])
    problems, events = [], []
    for _ in range(rng.randint(0, 8)):
        cat = rng.choice(["doubt", "fatigue", "distraction"])
        a = rng.uniform(0, seconds - 1)
        b = min(seconds, a + rng.uniform(10, 900))
        problems.append({"category": cat, "start": a, "end": b})
        if cat == "distraction":
            for _ in range(rng.randint(0, 2)):
                ea = rng.uniform(a, b)
                events.append({"kind": "spatial", "type": rng.choice(["away", "absent", "distracted", "hidden_abnormal"]),
                               "t_start": ea, "t_end": rng.choice([None, min(b, ea + rng.uniform(1, 200))])})
    problems.sort(key=lambda p: p["start"])
    return FeedbackContext(
        course_name=rng.choice([None, "線形代数", "英語"]), status=rng.choice(["completed", "interrupted", "recording"]),
        seconds=seconds, avg_score=rng.uniform(0, 100), problems=problems, spatial_events=events,
        history_scores=[rng.uniform(0, 100) for _ in range(rng.choice([0, 0, 1, 3]))],
    )


def fuzz():
    print("\n[1] 随机输入（2000 个）")
    rng = random.Random(9)
    gen = RuleFeedbackGenerator()
    templates = set(feedback.HINT_TEXT.values())
    errors = []
    for i in range(2000):
        c = random_context(rng)
        try:
            out = gen.generate(c)
        except Exception as e:  # noqa: BLE001
            errors.append((i, f"异常 {e!r}"))
            continue
        strings = [out["summary"], *[s["text"] for s in out["segments"]], *out["hints"],
                   *([out["comparison"]["text"]] if out["comparison"] else [])]
        if any(bad in t for t in strings for bad in ("None", "nan", "NaN", "undefined", "inf")):
            errors.append((i, "含 None/NaN"))
        # 独立重算分档：阈值分三档，问题时段（并集）≥30% 时前两档降一档
        level = 2 if c.avg_score >= DEFAULT_CONFIG.good_score else (1 if c.avg_score >= DEFAULT_CONFIG.fair_score else 0)
        covered = set()
        for p in c.problems:
            covered.update(range(int(p["start"] * 10), int(p["end"] * 10)))  # 0.1 秒精度的并集
        if len(covered) / 10 / c.seconds >= DEFAULT_CONFIG.heavy_problem_share + 0.001 and level > 0:
            level -= 1
        elif abs(len(covered) / 10 / c.seconds - DEFAULT_CONFIG.heavy_problem_share) <= 0.001:
            level = None  # 恰在边界附近，0.1 秒精度判断不了，跳过
        if level is not None and not out["summary"].startswith(HEADS[2 - level]):
            errors.append((i, f"总评分档不对：{c.avg_score:.1f} → {out['summary'][:12]}"))
        if len(out["segments"]) > DEFAULT_CONFIG.max_segments or out["more"] != max(0, len(c.problems) - DEFAULT_CONFIG.max_segments):
            errors.append((i, "时段条数 / ほか 计数不对"))
        if [s["start"] for s in out["segments"]] != sorted(s["start"] for s in out["segments"]):
            errors.append((i, "时段未按时间排序"))
        if not all(SEGMENT_RE.match(s["text"]) for s in out["segments"]):
            errors.append((i, f"时段文字格式：{[s['text'] for s in out['segments']]}"))
        if not out["hints"] or len(set(out["hints"])) != len(out["hints"]) or not set(out["hints"]) <= templates:
            errors.append((i, f"提示：{out['hints']}"))
        if (out["comparison"] is None) != (not c.course_name or not c.history_scores):
            errors.append((i, "比较段落出现条件不对"))
    check("随机输入全部满足不变量", not errors, f"{len(errors)} 个问题：{errors[:5]}" if errors else "2000 个")


# ------------------------------------------------------------------ 5. 文本样例
def samples_text():
    gen = RuleFeedbackGenerator()

    def p(cat, a, b):
        return {"category": cat, "start": float(a), "end": float(b)}

    def away(a, b):
        return {"kind": "spatial", "type": "away", "t_start": float(a), "t_end": float(b)}

    scenes = {
        "集中できた授業（問題なし、初回）": FeedbackContext("線形代数", "completed", 5400, 82, []),
        "おおむね＋3 種の問題＋前回比較": FeedbackContext(
            "線形代数", "completed", 3600, 72, [p("doubt", 725, 1045), p("fatigue", 2400, 2760), p("distraction", 3310, 3370)],
            [away(3310, 3370)], [66]),
        "高得点だが問題時段が多い（降格）": FeedbackContext("英語", "completed", 3600, 80, [p("fatigue", 1800, 3000)], [], [85, 88]),
        "集中が続きにくい授業": FeedbackContext("統計学", "completed", 3600, 52, [p("doubt", 0, 900), p("distraction", 1200, 1500)],
                                       [{"kind": "spatial", "type": "distracted", "t_start": 1200.0, "t_end": 1500.0}], [60, 58, 61]),
        "問題なしだが低得点": FeedbackContext(None, "completed", 1800, 55, []),
        "中断した記録": FeedbackContext("線形代数", "interrupted", 900, 70, [p("distraction", 100, 160)], [away(100, 160)], [70]),
        "時段が多い（ほか N か所）": FeedbackContext("英語", "completed", 5400, 68, [p("distraction", i * 600, i * 600 + 30) for i in range(6)],
                                          [away(i * 600, i * 600 + 30) for i in range(6)]),
    }
    lines = []
    for title, c in scenes.items():
        out = gen.generate(c)
        lines += [f"=== {title} ===", "■ 総評", out["summary"]]
        if out["segments"]:
            lines += ["■ 気になった時間帯", *[f"・{s['text']}" for s in out["segments"]]]
            if out["more"]:
                lines.append(f"ほかに {out['more']} か所あります（グラフで確認できます）。")
        lines += ["■ 次回へのヒント", *[f"・{h}" for h in out["hints"]]]
        if out["comparison"]:
            lines += [f"■ 前回との比較（{out['comparison']['course_name']}）", out["comparison"]["text"]]
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ 2–4. 真实窗口与性能
def in_window(db: Path, shot_dir: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    from verify_issue5_report import run_js, switch_to_report, wait_state
    from verify_issue8_summary import click
    session_store.DEFAULT_DB_PATH = db

    import main as app_main
    import verify_issue4_recording as v4
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

    def record(away_seconds):
        w.switchTo(m)
        QTest.mouseClick(m.startBtn, 1)
        QTest.qWait(9000)
        if away_seconds:
            camera.show_face = False
            QTest.qWait(away_seconds * 1000)
            camera.show_face = True
            QTest.qWait(3000)

        def fill_dialog():
            for dialog in w.findChildren(CourseNameDialog):
                if dialog.isVisible():
                    dialog.nameBox.setText("線形代数")
                    QTest.mouseClick(dialog.yesButton, 1)

        QTimer.singleShot(1500, fill_dialog)
        QTest.mouseClick(m.stopBtn, 1)
        QTest.qWait(2500)

    print("\n[2] 真实记录：同一课程名两节课")
    record(0)
    record(13)
    switch_to_report(w)
    wait_state(report.page)
    QTest.qWait(500)
    first, second = rows(db, "SELECT s.id, AVG(x.score) AS avg FROM sessions s JOIN samples x ON x.session_id = s.id"
                             " GROUP BY s.id ORDER BY s.started_at")
    card = {k: run_js(report.page, v) for k, v in {
        "hidden": "document.getElementById('feedback-card').hidden",
        "summary": "document.getElementById('feedback-summary').textContent",
        "segments": "[...document.querySelectorAll('#feedback-segments li')].map(li => li.textContent).join('|')",
        "title": "document.getElementById('feedback-comparison-title').textContent",
        "comparison": "document.getElementById('feedback-comparison').textContent",
    }.items()}
    check("报告页显示第二节课的反馈卡片",
          card["hidden"] is False and run_js(report.page, "document.getElementById('session-select').value") == str(second["id"]),
          card["summary"])
    check("离席约 13 秒被写成「席を離れていた可能性があります」", "席を離れていた可能性があります" in card["segments"], card["segments"])
    diff = second["avg"] - first["avg"]
    want = (f"（{round(first['avg'])}）と同程度でした。" if abs(diff) <= 5
            else f"（{round(first['avg'])}）より {round(abs(diff))} {'高く' if diff > 0 else '低く'}なりました。")
    check("「前回との比較（線形代数）」的数值与数据库一致",
          card["title"] == "前回との比較（線形代数）" and card["comparison"] == f"平均エンゲージメントは前回{want}",
          card["comparison"])

    print("\n[3] 真实鼠标：点击时段条目")
    seg = run_js(report.page, "JSON.stringify(state.chart.getOption().series.length)")  # 确保图表已就绪
    pt = [float(x) for x in run_js(report.page, """(() => {
        const a = document.querySelector('#feedback-segments .segment-link');
        a.scrollIntoView({block: 'center'});
        const r = a.getBoundingClientRect();
        return [r.left + 30, r.top + r.height / 2].join(',');
    })()""").split(",")]
    w.grab().save(str(shot_dir / "feedback_card.png"))
    target = report.webView.focusProxy() or report.webView
    click(target, pt)
    QTest.qWait(800)
    from app import report_data
    conn = session_store.connect(db)
    fb = report_data.load_session(conn, second["id"])["feedback"]
    conn.close()
    away_seg = next(s for s in fb["segments"] if "席を離れていた" in s["text"])
    first_seg = fb["segments"][0]
    extent = [float(x) for x in run_js(report.page, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        return c.getModel().getComponent('xAxis', 0).axis.scale.getExtent().join(',');
    })()""").split(",")]
    axis_max = float(run_js(report.page, "String(state.chart.getOption().xAxis[0].max)"))
    # 前后各留 30 秒，但不超出听课时间轴（真实记录的课很短）
    want = [max(0, first_seg["start"] - 30) / 60, min(axis_max, (first_seg["end"] + 30) / 60)]
    check("曲线缩放到该时段（前后各 30 秒）", seg and all(abs(a - b) < 1e-6 for a, b in zip(extent, want)),
          f"{extent[0] * 60:.1f}–{extent[1] * 60:.1f} 秒（离席时段 {away_seg['start']:.1f}–{away_seg['end']:.1f}）")
    w.grab().save(str(shot_dir / "feedback_zoomed.png"))
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    w.close()
    QTest.qWait(500)


def performance(db: Path):
    print("\n[4] 性能")
    from verify_issue8_summary import seed
    from app import report_data, session_store
    seed(db, 60, 90, random.Random(90), time.time() - 100 * 86400)
    conn = session_store.connect(db)
    conn.execute("UPDATE sessions SET course_name = '線形代数'")
    conn.commit()
    last = conn.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1").fetchone()[0]
    start = time.perf_counter()
    data = report_data.load_session(conn, last)
    ms = (time.perf_counter() - start) * 1000
    conn.close()
    check("同课程名 60 节历史时读取一节 90 分钟课程（含反馈）< 100 ms", ms < 100,
          f"{ms:.0f} ms；{data['feedback']['comparison']['text']}")


if __name__ == "__main__":
    fuzz()
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue9_"))
    (shot_dir / "feedback_samples.txt").write_text(samples_text(), encoding="utf-8")
    print("\n[5] 文本样例已写入 feedback_samples.txt")
    with tempfile.TemporaryDirectory() as d:
        in_window(Path(d) / "feedback.db", shot_dir)
        performance(Path(d) / "perf.db")
    print(f"\n截图与文本样例保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
