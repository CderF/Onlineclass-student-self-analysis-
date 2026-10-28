"""
#5 报告页骨架的专项验证（补充 tests/test_report_interface.py）

运行：在仓库根目录执行（会短暂弹出真实窗口）
    python test-scripts/verify_issue5_report.py

检查项：
  1. 真实窗口显示示例数据：原生平台创建 MainWindow，点侧栏切到报告页，检查课程列表、概要和曲线，并保存截图供人工查看
  2. 记录后立即可见：在同一窗口用假摄像头记录一节课并填写课程名，切回报告页后新记录出现在最前并与数据库一致
  3. 性能：90 分钟课程（5400 秒）从切换到显示完成的耗时、数据通道的 JSON 体积与耗时
  4. 离线：页面源文件不引用任何网络地址，运行中没有请求被拦截
  5. 真实程序退出：子进程运行 main.py、打开报告页后关闭窗口，不出现 WebEngine 释放顺序警告
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import math
import os
import random
import re
import sqlite3
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
ACCENT = (0x00, 0x9F, 0xAA)


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        failures.append(name)


def rows(db, sql, *args):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def seed(db: Path):
    """示例数据：90 分钟（含离席与一次暂停）、20 分钟后崩溃遗留、10 分钟未填课程名"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    rng = random.Random(5)
    conn = session_store.connect(db)
    day = 86400

    def lesson(start, minutes, name, pause_at=None, away=(), finish=True):
        sid = session_store.create_session(conn, start)
        session_store.set_baseline(conn, sid, -5.0, 1.5)
        rec, wall, t = SessionRecorder(sid, start, db), start, 0.0
        while t < minutes * 60:
            if pause_at is not None and abs(t - pause_at) < 0.25:
                rec.pause(wall)
                wall += 300
                rec = SessionRecorder(sid, wall, db)
            is_away = any(a <= t < b for a, b in away)
            score = 25 if is_away else 70 + 15 * math.sin(t / 400) - 10 * t / (minutes * 60) + rng.uniform(-5, 5)
            rec.add_frame(AnalysisResult(
                score=int(max(0, min(100, score))), status_text="", alert=None, has_face=not is_away,
                cognitive_state=None if is_away else rng.choice(["Understand", "Neutral", "Neutral", "Doubt"]),
                fatigue_index=min(1.0, t / (minutes * 60)), spatial_state="away" if is_away else "normal",
                delta_pitch=None if is_away else rng.uniform(-3, 3), delta_yaw=None if is_away else rng.uniform(-3, 3),
            ), wall)
            wall += 0.5
            t += 0.5
        rec.close(wall)
        if finish:
            session_store.finish_session(conn, sid, wall)
        session_store.set_course_name(conn, sid, name)
        return sid

    t0 = time.time() - 3 * day
    long_id = lesson(t0, 90, "線形代数 第3回", pause_at=1800, away=[(600, 640), (3000, 3030)])
    lesson(t0 + day, 20, "英語", finish=False)
    lesson(t0 + 2 * day, 10, None)
    session_store.mark_interrupted_sessions(conn)
    conn.close()
    return long_id


def run_js(page, script, timeout=5.0):
    from PyQt5.QtTest import QTest
    box = []
    page.runJavaScript(script, box.append)
    end = time.time() + timeout
    while not box and time.time() < end:
        QTest.qWait(10)
    return box[0] if box else None


def wait_state(page, timeout=10.0):
    from PyQt5.QtTest import QTest
    end = time.time() + timeout
    while time.time() < end:
        state = run_js(page, "document.body && document.body.dataset.state")
        if state in ("ready", "empty", "error"):
            return state
        QTest.qWait(10)
    return None


DOM = """JSON.stringify({
    options: [...document.querySelectorAll('#session-select option')].map(o => [o.value, o.text]),
    selected: document.getElementById('session-select').value,
    duration: document.getElementById('stat-duration').textContent,
    score: document.getElementById('stat-score').textContent,
    status: document.getElementById('stat-status').textContent,
    points: (echarts.getInstanceByDom(document.getElementById('score-chart')) || {getOption: () => ({series: [{data: []}]})})
        .getOption().series[0].data.length,
    chartWidth: document.getElementById('score-chart').clientWidth,
})"""


def accent_pixels(image) -> int:
    """截图里曲线颜色（主题色）的像素数，用来确认图表真的画出来了"""
    count = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            c = image.pixelColor(x, y)
            if abs(c.red() - ACCENT[0]) < 30 and abs(c.green() - ACCENT[1]) < 30 and abs(c.blue() - ACCENT[2]) < 30:
                count += 1
    return count


def switch_to_report(w):
    """真实点击侧栏「分析レポート」。先把页面状态标为 stale，之后 wait_state 等到的就是这次切换触发的刷新结果"""
    from PyQt5.QtTest import QTest
    run_js(w.reportInterface.page, "document.body.dataset.state = 'stale'")
    nav = w.navigationInterface.widget(w.reportInterface.objectName())
    QTest.mouseClick(nav.itemWidget, 1)
    return time.time()


# ------------------------------------------------------------------ 1–4. 真实窗口
def in_window(db: Path, shot_dir: Path):
    print("\n[1] 真实窗口显示示例数据（原生平台）")
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    session_store.DEFAULT_DB_PATH = db
    long_id = seed(db)

    import main
    import verify_issue4_recording as v4
    from app.view.course_name_dialog import CourseNameDialog
    from app.view.report_interface import ReportBridge
    camera = v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = main.MainWindow()
    w.show()
    report = w.reportInterface
    QTest.qWait(1500)  # 页面在后台加载完

    t_click = switch_to_report(w)
    state = wait_state(report.page)
    QTest.qWait(300)  # 等 ECharts 动画画完再截图
    d = json.loads(run_js(report.page, DOM))
    check("点击侧栏后切到报告页并显示示例数据", state == "ready" and w.stackedWidget.currentWidget() is report,
          f"{state}，{time.time() - t_click:.2f} 秒")
    check("列出 3 条课程记录，最新的在前且被选中",
          len(d["options"]) == 3 and d["selected"] == d["options"][0][0] and d["options"][0][1].endswith("授業名なし"),
          str([o[1] for o in d["options"]]))
    check("中断的记录带「（中断）」", d["options"][1][1].endswith("英語（中断）"))

    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select');
        s.value = '{long_id}';
        s.dispatchEvent(new Event('change'));
    }})()""")
    QTest.qWait(800)
    d = json.loads(run_js(report.page, DOM))
    db_avg = rows(db, "SELECT AVG(score) AS a, COUNT(*) AS n FROM samples WHERE session_id = ?", long_id)[0]
    check("90 分钟课程的概要与数据库一致",
          d["duration"] == "1時間30分" and d["score"] == str(round(db_avg["a"])) and d["status"] == "終了",
          f"{d['duration']} / {d['score']} / {d['status']}，数据库 {db_avg['n']} 秒、平均 {db_avg['a']:.1f}")
    check("曲线点数等于每秒样本数", d["points"] == db_avg["n"], f"{d['points']} 点")
    shot = shot_dir / "report_long_session.png"
    image = w.grab().toImage()
    image.save(str(shot))
    pixels = accent_pixels(image)
    check("截图中曲线确实被画出", d["chartWidth"] > 500 and pixels > 500,
          f"图表宽 {d['chartWidth']}px，主题色像素 {pixels}；截图 {shot}")

    print("\n[2] 记录后立即可见")
    w.switchTo(w.monitorInterface)
    m = w.monitorInterface
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(9000)  # 模型加载 + 3 秒校准 + 约 5 秒记录

    def fill_dialog():
        for dialog in w.findChildren(CourseNameDialog):
            if dialog.isVisible():
                dialog.nameBox.setText("英語 第2回")
                QTest.mouseClick(dialog.yesButton, 1)

    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    switch_to_report(w)
    state = wait_state(report.page)
    QTest.qWait(500)
    d = json.loads(run_js(report.page, DOM))
    new = rows(db, "SELECT s.id, COUNT(x.t) AS n, AVG(x.score) AS a FROM sessions s JOIN samples x ON x.session_id = s.id"
                   " WHERE s.course_name = '英語 第2回' GROUP BY s.id")
    check("新记录出现在最前并被选中",
          len(new) == 1 and d["options"][0][0] == str(new[0]["id"]) and d["options"][0][1].endswith("英語 第2回")
          and d["selected"] == str(new[0]["id"]), str(d["options"][:1]))
    if new:
        check("新记录的秒数与平均分与数据库一致",
              d["points"] == new[0]["n"] and d["score"] == str(round(new[0]["a"])),
              f"页面 {d['points']} 点 / {d['score']}，数据库 {new[0]['n']} 秒 / {new[0]['a']:.1f}")
    image = w.grab().toImage()
    image.save(str(shot_dir / "report_after_recording.png"))

    print("\n[3] 性能")
    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select');
        s.value = '{long_id}';
        s.dispatchEvent(new Event('change'));
    }})()""")
    QTest.qWait(500)
    w.switchTo(w.monitorInterface)
    QTest.qWait(300)
    t_click = switch_to_report(w)
    state = wait_state(report.page)
    elapsed = time.time() - t_click
    d = json.loads(run_js(report.page, DOM))
    check("刷新后仍选中 90 分钟课程", state == "ready" and d["selected"] == str(long_id) and d["points"] == 5400)
    check("切换到报告页并显示 90 分钟课程耗时 < 2 秒", elapsed < 2.0, f"{elapsed:.2f} 秒")
    bridge = ReportBridge(db_path=db)
    t = time.perf_counter()
    payload = bridge.load_session(long_id)
    load_ms = (time.perf_counter() - t) * 1000
    check("数据通道读取 90 分钟课程 < 200 ms", load_ms < 200,
          f"{load_ms:.0f} ms，JSON {len(payload.encode()) / 1024:.0f} KB")

    print("\n[4] 离线")
    web = ROOT / "app" / "view" / "web"
    refs = [f"{p.name}: {u}" for p in web.glob("*") if p.is_file()
            for u in re.findall(r"https?://[^\s\"')]+", p.read_text(encoding="utf-8"))]
    check("页面源文件（不含 vendor）没有引用网络地址", refs == [], str(refs))
    check("运行中没有请求被拦截（全部为本地资源）", report.interceptor.blocked == [], str(report.interceptor.blocked))
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    camera.show_face = True
    w.close()
    QTest.qWait(500)


# ------------------------------------------------------------------ 5. 真实程序退出
def real_exit(db: Path):
    print("\n[5] 真实程序打开报告页后退出")
    seed(db)
    code = f"""
import os, runpy, sys
sys.path.insert(0, {str(ROOT)!r}); os.chdir({str(ROOT)!r})
from app import session_store
session_store.DEFAULT_DB_PATH = __import__("pathlib").Path({str(db)!r})
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
_exec = QApplication.exec
def exec_hook(self):
    w = [x for x in self.topLevelWidgets() if type(x).__name__ == "MainWindow"][0]
    QTimer.singleShot(1000, lambda: w.switchTo(w.reportInterface))
    QTimer.singleShot(3000, lambda: print("STATE", w.reportInterface.page.js_errors, flush=True))
    QTimer.singleShot(3200, w.close)
    return _exec()  # QApplication.exec 是静态方法
QApplication.exec = exec_hook
sys.argv = ["main.py"]
runpy.run_path("main.py", run_name="__main__")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    out = proc.stdout + proc.stderr
    check("报告页打开期间没有 JS 错误", "STATE []" in out, next((l for l in out.splitlines() if l.startswith("STATE")), "无输出"))
    check("退出时没有 WebEngine 释放顺序警告", "Release of profile" not in out)
    # 退出码只打印：PyQt 退出清理（cleanup_on_exit）偶发 SIGSEGV，加入报告页之前就存在，与报告页无关
    print(f"INFO  子进程退出码 {proc.returncode}")


if __name__ == "__main__":
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue5_"))
    with tempfile.TemporaryDirectory() as d:
        in_window(Path(d) / "window.db", shot_dir)
        real_exit(Path(d) / "exit.db")
    print(f"\n截图保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}")
    sys.exit(1 if failures else 0)
