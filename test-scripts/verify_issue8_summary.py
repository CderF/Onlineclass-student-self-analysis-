"""
#8 多节课汇总对比的专项验证（补充 tests/test_report_summary.py、tests/test_summary_view.py）

运行：在仓库根目录执行（会短暂弹出真实窗口，约 2 分钟）
    python test-scripts/verify_issue8_summary.py

检查项：
  1. 独立重算：25 节示例课程（4 个课程名含未填写、含中断），不经过 report_data.summarize_sessions，
     直接用 SQL 与 problem_segments 重算每节课与每个课程名的数值，与图表数据、汇总表文字逐一比对
  2. 真实记录：原生窗口用假摄像头记录一节课并填写课程名，切到「全体の推移」后出现在最右，汇总表随之更新
  3. 真实鼠标：向网页视图发送 Qt 鼠标点击——点图表某一列跳到「授業ごと」显示该课；点汇总表的行按课程名筛选
  4. 像素核对：中断的课的点为空心（中心是底色），正常结束的课为实心
  5. 性能：60 节 90 分钟课程的汇总耗时，以及切换到「全体の推移」到显示完成的耗时
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

failures = []
# QApplication 保持到进程退出：随函数返回释放时，PyQt 清理偶发 SIGSEGV（#14）
KEEP_ALIVE = []
CATS = ("doubt", "fatigue", "distraction")
DAY = 86400


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def js_round(x):
    """与 JS 的 Math.round 一致（.5 向上），Python 的 round 是银行家舍入"""
    return math.floor(x + 0.5)


def rows(db, sql, *args):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def seed(db: Path, n: int, minutes: int, rng: random.Random, start: float):
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    names = ["線形代数", "英語", "統計学", None]
    conn = session_store.connect(db)
    for k in range(n):
        st = start + k * DAY
        sid = session_store.create_session(conn, st)
        rec = SessionRecorder(sid, st, db)
        cog, fat = "Neutral", rng.uniform(0.0, 0.3)
        away_at = rng.sample(range(0, minutes * 60 - 60), 2)
        for i in range(minutes * 60):
            if rng.random() < 0.01:
                cog = rng.choice(["Understand", "Neutral", "Doubt", "Doubt"])
            if rng.random() < 0.005:
                fat = rng.uniform(0.0, 0.7)
            away = any(a <= i < a + rng.choice([15, 25]) for a in away_at)
            rec.add_frame(AnalysisResult(
                score=int(25 if away else rng.uniform(50, 90)), status_text="", alert=None, has_face=not away,
                cognitive_state=None if away else cog, fatigue_index=fat, spatial_state="away" if away else "normal",
                delta_pitch=0.0, delta_yaw=0.0), st + i)
        rec.close(st + minutes * 60)
        if rng.random() > 0.15:
            session_store.finish_session(conn, sid, st + minutes * 60)
        session_store.set_course_name(conn, sid, names[k % 4])
    session_store.mark_interrupted_sessions(conn)
    conn.close()


def expected(db):
    """不经过 report_data.summarize_sessions 的独立重算"""
    from app.problem_segments import detect
    sessions = []
    for s in rows(db, "SELECT s.id, s.course_name, s.status, s.started_at, COUNT(x.t) AS n, AVG(x.score) AS avg"
                      " FROM sessions s JOIN samples x ON x.session_id = s.id GROUP BY s.id ORDER BY s.started_at, s.id"):
        samples = rows(db, "SELECT t, cognitive_state, fatigue_index FROM samples WHERE session_id = ? ORDER BY t", s["id"])
        events = rows(db, "SELECT kind, type, t_start, t_end FROM events WHERE session_id = ?", s["id"])
        pauses = rows(db, "SELECT t FROM pauses WHERE session_id = ?", s["id"])
        secs = {c: 0.0 for c in CATS}
        for p in detect(samples, events, pauses):
            secs[p["category"]] += p["end"] - p["start"]
        sessions.append({**s, "secs": secs})
    courses = {}
    for s in sessions:
        courses.setdefault(s["course_name"], []).append(s)
    return sessions, courses


def summary_js(page, body):
    from verify_issue5_report import run_js
    import json
    return json.loads(run_js(page, f"""JSON.stringify((() => {{
        const c = echarts.getInstanceByDom(document.getElementById('summary-chart'));
        {body}
    }})())"""))


def summary_point(page, grid, index, value):
    """图表数据坐标 → 网页视口坐标"""
    return summary_js(page, f"""
        document.getElementById('summary-chart').scrollIntoView({{block: 'center'}});
        const r = document.getElementById('summary-chart').getBoundingClientRect();
        const [x, y] = c.convertToPixel({{xAxisIndex: {grid}, yAxisIndex: {grid}}}, ['{index}', {value}]);
        return [r.left + x, r.top + y];""")


def click(target, pt):
    """向网页视图发送真实的 Qt 鼠标按下 / 松开"""
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from verify_issue6_session_chart import mouse_move
    mouse_move(target, pt)
    QTest.qWait(100)
    p = QPoint(int(pt[0]), int(pt[1]))
    for kind, buttons in ((QEvent.MouseButtonPress, Qt.LeftButton), (QEvent.MouseButtonRelease, Qt.NoButton)):
        QApplication.sendEvent(target, QMouseEvent(kind, QPointF(p), QPointF(target.mapToGlobal(p)),
                                                   Qt.LeftButton, buttons, Qt.NoModifier))
        QTest.qWait(50)


def open_summary(page, timeout=15.0):
    from PyQt5.QtTest import QTest
    from verify_issue5_report import run_js
    run_js(page, "document.body.dataset.summary = 'stale'; document.getElementById('tab-summary').click()")
    end = time.time() + timeout
    while time.time() < end and run_js(page, "document.body.dataset.summary") != "ready":
        QTest.qWait(5)
    return run_js(page, "document.body.dataset.summary")


def main(db: Path, perf_db: Path, shot_dir: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import report_data, session_store
    from verify_issue5_report import run_js, switch_to_report, wait_state
    from verify_issue6_session_chart import close_to, pixel

    session_store.DEFAULT_DB_PATH = db
    seed(db, 25, 20, random.Random(8), time.time() - 40 * DAY)

    import main as app_main
    import verify_issue4_recording as v4
    from app.view.course_name_dialog import CourseNameDialog
    v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    KEEP_ALIVE.append(app)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 950)
    w.show()
    report, m = w.reportInterface, w.monitorInterface
    QTest.qWait(1500)
    switch_to_report(w)
    wait_state(report.page)

    print("\n[1] 独立重算（25 节课）")
    state = open_summary(report.page)
    QTest.qWait(300)
    sessions, courses = expected(db)
    opt = summary_js(report.page, "return c.getOption();")
    series = {s["name"]: s["data"] for s in opt["series"]}
    check("切到「全体の推移」后显示完成", state == "ready")
    check("平均参与度与数据库一致",
          [d["value"] for d in series["平均エンゲージメント"]] == [js_round(s["avg"] * 10) / 10 for s in sessions], f"{len(sessions)} 节")
    check("中断的课用空心点",
          [d["symbol"] for d in series["平均エンゲージメント"]] == ["circle" if s["status"] == "completed" else "emptyCircle" for s in sessions],
          f"中断 {sum(s['status'] != 'completed' for s in sessions)} 节")
    check("听课时长与数据库一致", series["受講時間"] == [js_round(s["n"] / 60 * 10) / 10 for s in sessions])
    labels = {"doubt": "理解不足の可能性", "fatigue": "疲労", "distraction": "注意散漫・離席"}
    bad = [lab for c, lab in labels.items()
           if [round(v, 6) for v in series[lab]] != [round(js_round(s["secs"][c] / s["n"] * 1000) / 10, 6) for s in sessions]]
    check("三类问题时段占比与独立重算一致", not bad, f"不一致：{bad}" if bad else
          "、".join(f"{lab} 有 {sum(1 for v in series[lab] if v > 0)} 节" for lab in labels.values()))
    overlap = sum(1 for i in range(len(sessions)) if sum(series[lab][i] for lab in labels.values()) > 100)
    check("问题时段并列显示：纵轴固定 0–100%，每根柱不超过 100%",
          opt["yAxis"][2]["max"] == 100 and all(v <= 100 for lab in labels.values() for v in series[lab])
          and all("stack" not in s for s in opt["series"]), f"三类相加超过 100% 的课 {overlap} 节（重叠所致，并列后不会误读）")
    check("超过 15 节课时出现滑条、默认显示最近 15 节",
          len(opt["dataZoom"]) == 2 and opt["dataZoom"][0]["startValue"] == len(sessions) - 15)

    table = summary_js(report.page, """return [...document.querySelectorAll('#course-table tbody tr')]
        .map(tr => [...tr.cells].map(td => td.textContent));""")
    order = sorted(courses, key=lambda k: max(s["started_at"] for s in courses[k]), reverse=True)
    want = []
    for name in order:
        items = courses[name]
        n = sum(s["n"] for s in items)
        inter = sum(1 for s in items if s["status"] == "interrupted")
        minutes = js_round(n / 60)
        want.append([
            name or "授業名なし",
            f"{len(items)}（うち中断 {inter}）" if inter else str(len(items)),
            str(js_round(sum(s["avg"] * s["n"] for s in items) / n)),
            f"{minutes // 60}時間{minutes % 60}分" if minutes >= 60 else f"{minutes}分",
            *[f"{js_round(sum(s['secs'][c] for s in items) / n * 100)}%" for c in CATS],
        ])
    check("课程别汇总表与独立重算一致（加权平均、合计、占比、中断数、顺序）", table == want,
          f"{len(table)} 行" + ("" if table == want else f"；页面 {table}，预期 {want}"))
    w.grab().save(str(shot_dir / "summary_25.png"))

    print("\n[4] 像素核对：空心点")
    view = report.webView
    idx_int = next((i for i, s in enumerate(sessions) if s["status"] != "completed" and i >= len(sessions) - 15), None)
    idx_ok = next(i for i, s in enumerate(sessions) if s["status"] == "completed" and i >= len(sessions) - 15)
    if idx_int is not None:
        # 先算坐标（会把图表滚到可见处）再截图
        p_int = summary_point(report.page, 0, idx_int, js_round(sessions[idx_int]["avg"] * 10) / 10)
        p_ok = summary_point(report.page, 0, idx_ok, js_round(sessions[idx_ok]["avg"] * 10) / 10)
        QTest.qWait(300)
        image = w.grab().toImage()
        c_int, c_ok = pixel(image, view, p_int), pixel(image, view, p_ok)
        check("中断的点中心为底色、正常结束的点中心为主题色",
              close_to(c_int, (255, 255, 255), tol=30) and close_to(c_ok, (0x00, 0x9F, 0xAA), tol=40), f"{c_int} / {c_ok}")
    else:
        check("示例数据中最近 15 节含中断的课", False)

    print("\n[3] 真实鼠标")
    target = view.focusProxy() or view
    idx = len(sessions) - 3
    click(target, summary_point(report.page, 2, idx, 1))
    end = time.time() + 5
    while time.time() < end and run_js(report.page, "document.getElementById('view-single').hidden"):
        QTest.qWait(20)
    QTest.qWait(500)
    check("点击图表某一列后切到「授業ごと」并显示该课",
          run_js(report.page, "document.getElementById('view-single').hidden") is False
          and run_js(report.page, "document.getElementById('session-select').value") == str(sessions[idx]["id"]),
          f"第 {idx + 1} 节（id {sessions[idx]['id']}）")

    open_summary(report.page)
    QTest.qWait(300)
    row_pt = summary_js(report.page, """
        const tr = document.querySelector('#course-table tbody tr[data-key="name:英語"]');
        tr.scrollIntoView({block: 'center'});
        const r = tr.getBoundingClientRect();
        return [r.left + 20, r.top + r.height / 2];""")
    click(target, row_pt)
    QTest.qWait(500)
    n_en = len(courses["英語"])
    shown = summary_js(report.page, "return c.getOption().series[0].data.length;")
    check("点击汇总表「英語」行后只显示英語的课",
          shown == n_en and run_js(report.page, "document.getElementById('course-filter').value") == "name:英語",
          f"{shown} 节")
    click(target, row_pt)
    QTest.qWait(500)
    check("再次点击取消筛选", summary_js(report.page, "return c.getOption().series[0].data.length;") == len(sessions))

    print("\n[2] 真实记录后出现在汇总中")
    w.switchTo(m)
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(9000)

    def fill_dialog():
        for dialog in w.findChildren(CourseNameDialog):
            if dialog.isVisible():
                dialog.nameBox.setText("英語")
                QTest.mouseClick(dialog.yesButton, 1)

    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    switch_to_report(w)
    wait_state(report.page)
    state = open_summary(report.page)
    QTest.qWait(300)
    opt = summary_js(report.page, "return c.getOption();")
    last_label = summary_js(report.page, "return SummaryChart.axisLabel(state.summaryShown[state.summaryShown.length - 1]);")
    en_row = summary_js(report.page, """return [...document.querySelector('#course-table tbody tr[data-key="name:英語"]').cells]
        .map(td => td.textContent);""")
    check("新记录出现在最右（最新）", state == "ready" and len(opt["series"][0]["data"]) == len(sessions) + 1
          and last_label.split("\n")[1] == "英語", last_label.replace("\n", " "))
    check("汇总表「英語」的课次加 1", en_row[1].split("（")[0] == str(n_en + 1), en_row[1])
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    check("没有被拦截的请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    w.grab().save(str(shot_dir / "summary_after_recording.png"))

    print("\n[5] 性能（60 节 90 分钟课程）")
    seed(perf_db, 60, 90, random.Random(80), time.time() - 100 * DAY)
    conn = session_store.connect(perf_db)
    start = time.perf_counter()
    report_data.summarize_sessions(conn)
    ms = (time.perf_counter() - start) * 1000
    conn.close()
    check("汇总 60 节 90 分钟课程 < 1 秒", ms < 1000, f"{ms:.0f} ms（直接同步调用；页面上在后台线程，见 verify_issue16）")
    report.bridge.db_path = perf_db
    run_js(report.page, "document.getElementById('tab-single').click()")
    QTest.qWait(300)
    start = time.time()
    state = open_summary(report.page)
    elapsed = time.time() - start
    check("切换到「全体の推移」并显示完成 < 2 秒", state == "ready" and elapsed < 2.0, f"{elapsed:.2f} 秒")
    w.grab().save(str(shot_dir / "summary_60.png"))
    w.close()
    QTest.qWait(500)


if __name__ == "__main__":
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue8_"))
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "summary.db", Path(d) / "perf.db", shot_dir)
    print(f"\n截图保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    os._exit(1 if failures else 0)  # 跳过解释器清理：PyQt 退出清理偶发 SIGSEGV（#14）
