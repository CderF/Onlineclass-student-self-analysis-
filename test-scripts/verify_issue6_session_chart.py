"""
#6 单节课图表的专项验证（补充 tests/test_session_chart.py）

运行：在仓库根目录执行（会短暂弹出真实窗口）
    python test-scripts/verify_issue6_session_chart.py

检查项：
  1. 真实记录 → 图表：原生窗口用假摄像头记录一节课（含离席与暂停），切到报告页后，
     图表里的曲线、认知状态色带、空间状态区间、警报与暂停位置与数据库独立重算的结果一致
  2. 像素核对：90 分钟示例课程（7 种空间状态、4 种警报、2 次暂停）截图后，在每个区间中点取色，
     确认画在正确的行、颜色正确；每条暂停竖线都画出
  3. 真实鼠标交互：向网页视图发送真实的 Qt 鼠标移动 / 滚轮事件，确认悬停提示内容正确、四行缩放联动
  4. 性能与错误：90 分钟课程切换到显示完成的耗时，全程无 JS 错误、无被拦截的请求
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import math
import os
import random
import sys
import tempfile
import time
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

from verify_issue5_report import rows, run_js, switch_to_report, wait_state  # noqa: E402

failures = []

GROUP_OF = {
    "head_down": "下向き・ノート", "taking_notes": "下向き・ノート", "hidden_notes": "下向き・ノート",
    "distracted": "注意散漫", "hidden_abnormal": "注意散漫", "away": "離席", "absent": "離席",
}
GROUP_COLOR = {"下向き・ノート": (0x62, 0x50, 0xD6), "注意散漫": (0xED, 0xA1, 0x00), "離席": (0xE3, 0x49, 0x48)}


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def chart_js(page, body):
    """在页面里以 c = 图表实例执行 body，结果以 JSON 取回"""
    return json.loads(run_js(page, f"""JSON.stringify((() => {{
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        {body}
    }})())"""))


def option(page):
    return chart_js(page, "return c.getOption();")


def bars(opt, name):
    s = [x for x in opt["series"] if x["name"] == name and x["type"] == "custom"][0]
    return [(round(d[0] * 60, 3), round(d[1] * 60, 3)) for d in s["data"]]


def expected_from_db(db, sid):
    """不经过 JS，直接从数据库重算图表应显示的内容"""
    samples = rows(db, "SELECT t, score, fatigue_index, cognitive_state FROM samples WHERE session_id = ? ORDER BY t", sid)
    runs = {}
    # 状态相同且秒连续的样本合并为一段
    keyed = [(s["cognitive_state"], s["t"] - i) for i, s in enumerate(samples)]
    for (state, _), group in groupby(zip(keyed, samples), key=lambda x: x[0]):
        group = [g[1] for g in group]
        if state is not None:
            runs.setdefault(state, []).append((group[0]["t"], group[-1]["t"] + 1))
    end = samples[-1]["t"] + 1
    spatial = {}
    for e in rows(db, "SELECT type, t_start, t_end FROM events WHERE session_id = ? AND kind = 'spatial' ORDER BY t_start", sid):
        spatial.setdefault(GROUP_OF[e["type"]], []).append(
            (round(e["t_start"], 3), round(e["t_end"] if e["t_end"] is not None else end, 3)))
    alerts = [round(a["t_start"], 3) for a in
              rows(db, "SELECT t_start FROM events WHERE session_id = ? AND kind = 'alert' ORDER BY t_start", sid)]
    pauses = rows(db, "SELECT t, paused_at, resumed_at FROM pauses WHERE session_id = ? ORDER BY t", sid)
    return samples, runs, spatial, alerts, pauses, end


def compare_chart(page, db, sid, label):
    samples, runs, spatial, alerts, pauses, end = expected_from_db(db, sid)
    opt = option(page)
    score = opt["series"][0]["data"]
    check(f"{label}：分数曲线与每秒样本一致",
          [round(p[0] * 60, 6) for p in score] == [s["t"] for s in samples] and [p[1] for p in score] == [s["score"] for s in samples],
          f"{len(score)} 点")
    names = {"Understand": "理解", "Doubt": "疑問", "Neutral": "中立", "Disgust": "嫌悪"}
    got_runs = {code: bars(opt, ja) for code, ja in names.items()}
    want_runs = {code: [(float(a), float(b)) for a, b in runs.get(code, [])] for code in names}
    check(f"{label}：认知状态色带与独立重算一致", got_runs == want_runs,
          ", ".join(f"{names[k]} {len(v)} 段" for k, v in want_runs.items()))
    got_spatial = {g: bars(opt, g) for g in GROUP_COLOR}
    want_spatial = {g: spatial.get(g, []) for g in GROUP_COLOR}
    check(f"{label}：空间状态区间与数据库事件一致", got_spatial == want_spatial,
          ", ".join(f"{g} {len(v)} 段" for g, v in want_spatial.items()))
    got_alerts = [round(a[0] * 60, 3) for a in [x for x in opt["series"] if x["name"] == "アラート"][0]["data"]]
    check(f"{label}：警报位置与数据库一致", got_alerts == alerts, str(alerts))
    lines = [x for x in opt["series"] if x["name"] == "一時停止"]
    got_pauses = [[round(d["xAxis"] * 60, 3) for d in s["markLine"]["data"]] for s in lines]
    want = [round(p["t"], 3) for p in pauses]
    check(f"{label}：四行都在暂停位置画竖线", len(lines) == 4 and all(g == want for g in got_pauses), str(want))
    check(f"{label}：横轴终点为听课时间轴终点", all(abs(x["max"] * 60 - end) < 1e-6 for x in opt["xAxis"]), f"{end} 秒")
    return samples, spatial, alerts, pauses


# ------------------------------------------------------------------ 示例数据
def seed(db: Path):
    """90 分钟：7 种空间状态、4 种警报、认知状态成段变化、2 次暂停"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    rng = random.Random(6)
    conn = session_store.connect(db)
    start = time.time() - 86400
    sid = session_store.create_session(conn, start)
    plan = [(300, 310, "head_down"), (900, 1100, "taking_notes"), (1500, 1530, "hidden_notes"),
            (2000, 2200, "distracted"), (2600, 2700, "hidden_abnormal"), (3300, 3400, "away"), (4000, 4300, "absent")]
    alerts = {2050.0: "head_down", 2650.0: "hidden", 4100.0: "absence", 4800.0: "fatigue"}
    pause_at = {1800.0: 120, 3600.0: 600}

    rec, wall, t, cog = SessionRecorder(sid, start, db), start, 0.0, "Neutral"
    while t < 5400:
        if t in pause_at:
            rec.pause(wall)
            wall += pause_at.pop(t)
            rec = SessionRecorder(sid, wall, db)
        spatial = next((code for a, b, code in plan if a <= t < b), "normal")
        no_face = spatial in ("hidden_notes", "hidden_abnormal", "away", "absent")
        if rng.random() < 0.01:
            cog = rng.choice(["Understand", "Neutral", "Doubt", "Disgust"])
        alert = alerts.get(t)
        rec.add_frame(AnalysisResult(
            score=int(25 if no_face else max(0, min(100, 70 + 15 * math.sin(t / 500) + rng.uniform(-5, 5)))),
            status_text="", alert=(alert, "x", "y") if alert else None, has_face=not no_face,
            cognitive_state=None if no_face else cog, fatigue_index=min(1.0, t / 5400),
            spatial_state=spatial, delta_pitch=None if no_face else 0.0, delta_yaw=None if no_face else 0.0,
        ), wall)
        wall += 0.5
        t += 0.5
    rec.close(wall)
    session_store.finish_session(conn, sid, wall)
    session_store.set_course_name(conn, sid, "線形代数 第5回")
    conn.close()
    return sid


# ------------------------------------------------------------------ 坐标换算
def chart_point(page, grid, minute, row):
    """数据坐标 → 网页视口坐标（CSS 像素）；先把图表滚到可见处"""
    return chart_js(page, f"""
        document.getElementById('score-chart').scrollIntoView({{block: 'center'}});
        const r = document.getElementById('score-chart').getBoundingClientRect();
        const [x, y] = c.convertToPixel({{xAxisIndex: {grid}, yAxisIndex: {grid}}}, [{minute}, {json.dumps(row)}]);
        return [r.left + x, r.top + y];""")


def pixel(image, view, pt):
    """网页视口坐标处的截图颜色（截图为设备像素）"""
    from PyQt5.QtCore import QPoint
    p = view.mapTo(view.window(), QPoint(int(pt[0]), int(pt[1])))
    ratio = image.devicePixelRatio()
    c = image.pixelColor(int(p.x() * ratio), int(p.y() * ratio))
    return c.red(), c.green(), c.blue()


def mouse_move(target, pt):
    """向网页视图发送真实的 Qt 鼠标移动事件（MainWindow 里 QTest.mouseMove 会移动系统光标，事件到不了页面）"""
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication
    p = QPoint(int(pt[0]), int(pt[1]))
    QApplication.sendEvent(target, QMouseEvent(QEvent.MouseMove, QPointF(p), QPointF(target.mapToGlobal(p)),
                                               Qt.NoButton, Qt.NoButton, Qt.NoModifier))


def wheel(target, pt, steps):
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    p = QPoint(int(pt[0]), int(pt[1]))
    for _ in range(steps):
        QApplication.sendEvent(target, QWheelEvent(QPointF(p), QPointF(target.mapToGlobal(p)), QPoint(0, 0),
                                                   QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
        QTest.qWait(80)


def close_to(c, target, tol=40):
    return all(abs(a - b) <= tol for a, b in zip(c, target))


# ------------------------------------------------------------------ 主流程
def main(db: Path, shot_dir: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    session_store.DEFAULT_DB_PATH = db
    long_id = seed(db)

    import main as app_main
    import verify_issue4_recording as v4
    from app.view.course_name_dialog import CourseNameDialog
    camera = v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 900)
    w.show()
    report, m = w.reportInterface, w.monitorInterface
    QTest.qWait(1500)

    print("\n[1] 真实记录 → 图表")
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(8000)                      # 模型加载 + 3 秒校准 + 约 4 秒记录
    camera.show_face = False
    QTest.qWait(5000)                      # 离席 5 秒
    camera.show_face = True
    QTest.qWait(2000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(3000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(5000)

    def fill_dialog():
        for dialog in w.findChildren(CourseNameDialog):
            if dialog.isVisible():
                dialog.nameBox.setText("英語 第3回")
                QTest.mouseClick(dialog.yesButton, 1)

    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    switch_to_report(w)
    state = wait_state(report.page)
    QTest.qWait(500)
    rec_id = rows(db, "SELECT id FROM sessions WHERE course_name = '英語 第3回'")[0]["id"]
    check("报告页显示刚记录的课程",
          state == "ready" and run_js(report.page, "document.getElementById('session-select').value") == str(rec_id))
    _, spatial, _, pauses = compare_chart(report.page, db, rec_id, "真实记录")
    away = spatial.get("離席", [])
    check("离席约 5 秒被画为一段「離席」", len(away) == 1 and 4.0 <= away[0][1] - away[0][0] <= 6.0, str(away))
    check("暂停 1 次", len(pauses) == 1 and pauses[0]["resumed_at"] is not None)
    w.grab().save(str(shot_dir / "recorded_session.png"))

    print("\n[2] 90 分钟示例课程：耗时与像素核对")
    w.switchTo(m)
    QTest.qWait(300)
    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select');
        s.value = '{long_id}';
        s.dispatchEvent(new Event('change'));
    }})()""")  # 页面隐藏时改选，切回时保持该选择（没有新记录）
    QTest.qWait(500)
    t_click = switch_to_report(w)
    state = wait_state(report.page)
    elapsed = time.time() - t_click
    check("切换到报告页并画完 90 分钟课程 < 2 秒", state == "ready" and elapsed < 2.0, f"{elapsed:.2f} 秒")
    _, spatial, alerts, pauses = compare_chart(report.page, db, long_id, "90 分钟")
    check("示例覆盖全部 7 种空间状态与 4 种警报",
          rows(db, "SELECT COUNT(DISTINCT type) AS n FROM events WHERE session_id = ? AND kind = 'spatial'", long_id)[0]["n"] == 7
          and len(alerts) == 4)

    QTest.qWait(300)
    view = report.webView
    centers = {(g, a, b): chart_point(report.page, 3, (a + b) / 2 / 60, g) for g, ivs in spatial.items() for a, b in ivs}
    image = w.grab().toImage()
    image.save(str(shot_dir / "long_session.png"))
    wrong = {f"{g} {a:.0f}–{b:.0f}s": pixel(image, view, pt) for (g, a, b), pt in centers.items()
             if not close_to(pixel(image, view, pt), GROUP_COLOR[g])}
    check("每个空间状态区间都在正确的行、以正确的颜色画出", not wrong and len(centers) == 7,
          f"{len(centers)} 段" + (f"，不符：{wrong}" if wrong else ""))
    # 暂停竖线：在分数行顶部附近、竖线横坐标处应为灰色虚线（取线附近几像素内最暗的点）
    missing = []
    for p in pauses:
        x, y = chart_point(report.page, 0, p["t"] / 60, 95)
        darkest = min(sum(pixel(image, view, (x + dx, y + dy))) for dx in (-1, 0, 1) for dy in range(0, 12))
        if darkest > 3 * 200:
            missing.append(p["t"])
    check("每条暂停竖线都画出", not missing, f"{len(pauses)} 条" + (f"，缺失 {missing}" if missing else ""))

    print("\n[3] 真实鼠标交互")
    target = view.focusProxy() or view
    a, b = spatial["離席"][0]
    minute = (a + b) / 2 / 60
    pt = chart_point(report.page, 3, minute, "離席")
    mouse_move(target, pt)
    QTest.qWait(600)
    tip = run_js(report.page, "document.getElementById('score-chart').innerText")
    second = round(minute * 60)
    check("鼠标移到「離席」区间上显示对应的悬停提示",
          "離席中" in tip and "認知状態：なし" in tip and "NaN" not in tip,
          tip.replace("\n", " / ")[:120])
    check("提示的时间与鼠标位置一致（±10 秒，约 2 像素）",
          any(f"{s // 60}分{s % 60}秒" in tip for s in range(second - 10, second + 11)), f"约 {second} 秒")

    # 警报：鼠标移到三角形上（缩小显示时靠容差匹配）
    pt = chart_point(report.page, 3, 4100 / 60, "アラート")
    mouse_move(target, pt)
    QTest.qWait(600)
    tip = run_js(report.page, "document.getElementById('score-chart').innerText")
    check("鼠标移到警报三角形上显示警报名", "アラート：離席継続" in tip, tip.replace("\n", " / ")[:120])

    before = chart_js(report.page, "return c.getOption().dataZoom[0];")
    pt = chart_point(report.page, 0, 45, 50)
    mouse_move(target, pt)
    wheel(target, pt, 5)
    QTest.qWait(500)
    after = chart_js(report.page, "return c.getOption().dataZoom[0];")
    extents = chart_js(report.page,
                       "return [0, 1, 2, 3].map((i) => c.getModel().getComponent('xAxis', i).axis.scale.getExtent());")
    check("滚轮缩放后四行横轴范围相同且变窄",
          after["end"] - after["start"] < before["end"] - before["start"]
          and all(abs(e[0] - extents[0][0]) < 1e-9 and abs(e[1] - extents[0][1]) < 1e-9 for e in extents),
          f"{before['start']:.0f}–{before['end']:.0f}% → {after['start']:.1f}–{after['end']:.1f}%")
    w.grab().save(str(shot_dir / "zoomed.png"))

    print("\n[4] 错误与网络")
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    check("没有被拦截的请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    w.close()
    QTest.qWait(500)


if __name__ == "__main__":
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue6_"))
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "chart.db", shot_dir)
    print(f"\n截图保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    # 跳过解释器清理：PyQt 退出清理偶发 SIGSEGV（#14），否则检查全部通过时退出码也可能是 139
    os._exit(1 if failures else 0)
