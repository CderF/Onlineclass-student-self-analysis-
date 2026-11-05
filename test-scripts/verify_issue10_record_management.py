"""
#10 课程记录管理的专项验证（补充 tests/test_record_management.py）

运行：在仓库根目录执行（会短暂弹出真实窗口并保持在最前，约 1 分钟）
    python test-scripts/verify_issue10_record_management.py

检查项：
  1. 真实对话框：原生窗口里用 Qt 鼠标点击网页的「名前を変更」「削除」，由页面经 QWebChannel 弹出真实的
     RenameCourseDialog / MessageBox（槽内模态），点对话框按钮保存、取消、清空课程名、确认删除
  2. 级联删除：删除后该课的 samples / events / pauses 全部消失，其他课逐行不变，外键与完整性检查通过
  3. 录制中并发：真实 CameraThread 记录中，该课的按钮不可用（真实点击不弹框、不改库）；同时删除另一节课，
     记录继续写入、不出现「記録エラー」；停止并填写课程名后该课变为可操作
  4. 耗时：删除一节 90 分钟课程（5400 秒样本）从点击确认到列表刷新完成的时间
  5. 删除后的数据残留：检查数据库与 WAL 文件里是否还能找到已删除的课程名（只报告，不判失败）
  6. 屏幕上的实际显示（#18）：对话框显示期间截取本窗口在屏幕上的实际画面（QScreen.grabWindow，本进程的窗口
     不需要屏幕录制权限），对话框区域应与 Qt 自绘的结果一致（没有被网页盖住），网页背景按遮罩均匀变暗（约 0.70 倍）。
     w.grab() 由 Qt 软件重绘，不反映网页（GPU 合成）与普通子控件的实际叠放，不能单独用来检查遮挡
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import os
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
TABLES = ("samples", "events", "pauses")


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


def snapshot(db, sid):
    """一节课的全部行（用于确认其他课逐行不变）"""
    return {t: rows(db, f"SELECT * FROM {t} WHERE session_id = ? ORDER BY rowid", sid) for t in TABLES}


def lesson(db, start, seconds, name, interrupted=False):
    """每秒一帧，中间有离席（事件 + 警报）和一次暂停"""
    from app import session_store
    from app.attention_rules import AnalysisResult
    from app.session_recorder import SessionRecorder

    def frame(i):
        away = 60 <= i % 600 < 90
        return AnalysisResult(score=30 if away else 75, status_text="", alert="absence" if i % 600 == 70 else None,
                              has_face=not away, cognitive_state=None if away else "Neutral", fatigue_index=0.1,
                              spatial_state="away" if away else "normal", delta_pitch=0.0, delta_yaw=0.0)

    conn = session_store.connect(db)
    sid = session_store.create_session(conn, start)
    half = seconds // 2
    rec = SessionRecorder(sid, start, db)
    for i in range(half):
        rec.add_frame(frame(i), start + i)
    rec.pause(start + half)
    rec = SessionRecorder(sid, start + half + 120, db)
    for i in range(half, seconds):
        rec.add_frame(frame(i), start + 120 + i)
    rec.close(start + 120 + seconds)
    if not interrupted:
        session_store.finish_session(conn, sid, start + 120 + seconds)
    session_store.set_course_name(conn, sid, name)
    session_store.mark_interrupted_sessions(conn)
    conn.close()
    return sid


class DialogDriver:
    """对话框（模态、在网页调用的槽里打开）出现后，在其事件循环中执行预先设定的操作"""

    def __init__(self, window):
        from PyQt5.QtCore import QTimer
        self.window = window
        self.plan = None
        self.seen = []
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

    def expect(self, action):
        self.plan = action

    def _tick(self):
        from qfluentwidgets import MessageBox
        from app.view.course_name_dialog import RenameCourseDialog
        for cls in (RenameCourseDialog, MessageBox):
            for d in self.window.findChildren(cls):
                if d.isVisible() and all(d is not s[1] for s in self.seen):
                    self.seen.append((cls.__name__, d))
                    action, self.plan = self.plan, None
                    if action:
                        action(d)


def main(db: Path, shot_dir: Path):
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
    now = time.time()
    a = lesson(db, now - 5 * 86400, 600, "線形代数")
    b = lesson(db, now - 4 * 86400, 600, "英語")
    c = lesson(db, now - 3 * 86400, 300, "統計学", interrupted=True)
    big = lesson(db, now - 2 * 86400, 5400, "削除テスト用の長い授業")

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
    w.raise_()
    w.activateWindow()
    report, m = w.reportInterface, w.monitorInterface
    target = report.webView.focusProxy() or report.webView
    driver = DialogDriver(w)
    QTest.qWait(1500)
    switch_to_report(w)
    wait_state(report.page)

    def point(selector):
        return run_js(report.page, f"""(() => {{
            const el = document.querySelector('{selector}');
            el.scrollIntoView({{block: 'center'}});
            const r = el.getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];
        }})()""")

    def press(selector, timeout=10.0):
        """真实点击网页里的元素，等到管理操作结束（body[data-manage] 变为 idle）"""
        run_js(report.page, "document.body.dataset.manage = 'stale'")
        click(target, point(selector))
        end = time.time() + timeout
        while time.time() < end and run_js(report.page, "document.body.dataset.manage") != "idle":
            QTest.qWait(20)
        return run_js(report.page, "document.body.dataset.manage")

    def button(sid, action):
        return f'#record-table tr[data-id="{sid}"] button[data-action="{action}"]'

    def table():
        return run_js(report.page, """[...document.querySelectorAll('#record-table tbody tr')].map(tr =>
            [Number(tr.dataset.id), ...[...tr.cells].slice(0, 4).map(td => td.textContent),
             [...tr.querySelectorAll('button')].every(b => b.disabled)])""")

    def name_of(sid):
        r = rows(db, "SELECT course_name FROM sessions WHERE id = ?", sid)
        return r[0]["course_name"] if r else "<deleted>"

    from PyQt5.QtCore import QPoint, QRect
    on_screen = {}
    bg_points = [(300, w.height() - 60), (w.width() - 200, w.height() - 60)]  # 网页透明背景（卡片与对话框之外）

    def screen_image():
        g = w.geometry()
        return app.primaryScreen().grabWindow(0, g.x(), g.y(), g.width(), g.height())

    def lum(img, x, y):
        c = img.pixelColor(int(x * img.width() / w.width()), int(y * img.height() / w.height()))
        return (c.red() + c.green() + c.blue()) / 3

    def before_dialog():
        shot = screen_image()
        shot.save(str(shot_dir / "before_dialog_screen.png"))
        img = shot.toImage()
        on_screen["bg_before"] = [lum(img, x, y) for x, y in bg_points]

    def capture_screen(d, name):
        """在对话框的事件循环里：截取屏幕上的实际画面，与 Qt 自绘结果比较对话框区域"""
        QTest.qWait(500)  # 等淡入动画结束
        shot = screen_image()
        shot.save(str(shot_dir / f"{name}_screen.png"))
        s_img, g_img = shot.toImage(), w.grab().toImage()
        r = QRect(d.widget.mapTo(w, QPoint(0, 0)), d.widget.size())
        diff = n = 0
        for y in range(r.top() + 4, r.bottom() - 4, 4):
            for x in range(r.left() + 4, r.right() - 4, 4):
                a = s_img.pixelColor(int(x * s_img.width() / w.width()), int(y * s_img.height() / w.height()))
                b = g_img.pixelColor(int(x * g_img.width() / w.width()), int(y * g_img.height() / w.height()))
                diff += abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())
                n += 1
        # 屏幕锁定或休眠时截到的是空白（全 0）画面，无法检查
        blank = s_img.width() < w.width() * 1.5 and max(on_screen["bg_before"]) == 0
        ratio = [] if blank else [lum(s_img, x, y) / max(b0, 1) for (x, y), b0 in zip(bg_points, on_screen["bg_before"])]
        on_screen[name] = {"diff": diff / n / 3, "ratio": ratio, "page_hidden": not report.webView.isVisible(),
                           "blank": blank}

    def check_screen(name, label):
        res = on_screen.get(name, {})
        if res.get("blank"):
            check(f"屏幕实际显示：{label}在最上层（#18）", False, "截到的屏幕画面是空白：屏幕已锁定或休眠，请解锁后重跑")
            return
        ok = (res.get("diff", 99) < 3 and all(0.65 <= v <= 0.75 for v in res.get("ratio", [0]))
              and res.get("page_hidden") and report.webView.isVisible() and not report.snapshot.isVisible())
        check(f"屏幕实际显示：{label}在最上层、网页背景按遮罩变暗，关闭后网页恢复（#18）", ok,
              f"对话框区域与自绘差 {res.get('diff', -1):.2f}，背景亮度比 {[round(v, 3) for v in res.get('ratio', [])]}；"
              f"截图 {shot_dir / (name + '_screen.png')}")

    print("\n[1] 真实对话框：改课程名")
    click(target, point("#tab-manage"))
    QTest.qWait(500)
    t = table()
    check("真实点击标签后显示「記録の管理」、按时间倒序列出全部记录",
          run_js(report.page, "document.getElementById('view-manage').hidden") is False and [r[0] for r in t] == [big, c, b, a],
          f"{[(r[0], r[2], r[4]) for r in t]}")
    w.grab().save(str(shot_dir / "manage_list.png"))

    got = {}

    def rename_to(text):
        def act(d):
            got["prefill"], got["title"] = d.nameBox.text(), d.titleLabel.text()
            got["items"] = [d.nameBox.itemText(i) for i in range(d.nameBox.count())]
            capture_screen(d, "rename_dialog")
            d.nameBox.setText(text)          # QTest.keyClicks 不能输入日语
            QTest.mouseClick(d.yesButton, 1)
        return act

    driver.expect(rename_to("線形代数 第2回"))
    before_dialog()
    state = press(button(a, "rename"))
    check_screen("rename_dialog", "改名对话框")
    check("改名对话框由页面弹出：标题、预填当前课程名、可选用过的课程名",
          got.get("title") == "授業名を変更" and got.get("prefill") == "線形代数"
          and set(got.get("items", [])) == {"線形代数", "英語", "統計学", "削除テスト用の長い授業"}, str(got))
    check("保存后数据库与列表、「授業ごと」下拉框都更新",
          state == "idle" and name_of(a) == "線形代数 第2回"
          and [r[2] for r in table() if r[0] == a] == ["線形代数 第2回"]
          and any(o[0] == str(a) and "線形代数 第2回" in o[1] for o in run_js(
              report.page, "[...document.querySelectorAll('#session-select option')].map(o => [o.value, o.text])")),
          name_of(a))

    driver.expect(lambda d: (d.nameBox.setText("ここは保存されない"), QTest.mouseClick(d.cancelButton, 1)))
    state = press(button(a, "rename"))
    check("取消后不改", state == "idle" and name_of(a) == "線形代数 第2回")

    driver.expect(lambda d: (d.nameBox.setText("   "), QTest.mouseClick(d.yesButton, 1)))
    state = press(button(b, "rename"))
    check("清空后保存 → 授業名なし（NULL）", state == "idle" and name_of(b) is None
          and [r[2] for r in table() if r[0] == b] == ["授業名なし"], str(name_of(b)))

    print("\n[2] 真实对话框：删除与级联")
    keep = {sid: snapshot(db, sid) for sid in (a, b)}
    before_c = {t_: len(v) for t_, v in snapshot(db, c).items()}
    msg = {}

    def cancel(d):
        msg["title"], msg["content"] = d.titleLabel.text(), d.content   # contentLabel 是 TextWrap 重排后的文字
        msg["yes"], msg["no"] = d.yesButton.text(), d.cancelButton.text()
        capture_screen(d, "delete_dialog")
        QTest.mouseClick(d.cancelButton, 1)

    driver.expect(cancel)
    before_dialog()
    state = press(button(c, "delete"))
    check_screen("delete_dialog", "删除确认框")
    started = time.strftime("%Y/%m/%d %H:%M", time.localtime(rows(db, "SELECT started_at FROM sessions WHERE id = ?", c)[0]["started_at"]))
    check("删除确认框写明日时与课程名、不可恢复", msg.get("title") == "この記録を削除しますか？"
          and msg.get("content") == f"{started}　統計学\nこの授業の記録をすべて削除します。\n元に戻すことはできません。"
          and (msg.get("yes"), msg.get("no")) == ("削除", "キャンセル"), str(msg))
    check("取消后记录仍在", state == "idle" and name_of(c) == "統計学"
          and {t_: len(v) for t_, v in snapshot(db, c).items()} == before_c, str(before_c))

    driver.expect(lambda d: QTest.mouseClick(d.yesButton, 1))
    state = press(button(c, "delete"))
    left = {t_: rows(db, f"SELECT COUNT(*) AS n FROM {t_} WHERE session_id = ?", c)[0]["n"] for t_ in TABLES}
    check("确认删除后中断的课及其样本 / 事件 / 暂停全部消失",
          state == "idle" and name_of(c) == "<deleted>" and left == {"samples": 0, "events": 0, "pauses": 0}
          and all(n > 0 for n in before_c.values()), f"删除前 {before_c}，删除后 {left}")
    check("其他课逐行不变", all(snapshot(db, sid) == keep[sid] for sid in (a, b)))
    check("外键与完整性检查通过", rows(db, "PRAGMA foreign_key_check") == []
          and rows(db, "PRAGMA integrity_check")[0]["integrity_check"] == "ok")
    check("列表不再含该课", c not in [r[0] for r in table()])

    print("\n[4] 删除 90 分钟课程的耗时")
    n_big = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", big)[0]["n"]
    clicked = {}
    driver.expect(lambda d: (clicked.setdefault("t", time.perf_counter()), QTest.mouseClick(d.yesButton, 1)))
    state = press(button(big, "delete"))
    ms = (time.perf_counter() - clicked.get("t", time.perf_counter())) * 1000
    check("删除 90 分钟课程：确认到列表刷新完成 < 1 秒", state == "idle" and name_of(big) == "<deleted>" and ms < 1000,
          f"{n_big} 秒样本，{ms:.0f} ms")

    print("\n[3] 录制中并发")
    alerts = []
    w.switchTo(m)
    QTest.mouseClick(m.startBtn, 1)
    m.camera_thread.alert_signal.connect(lambda title, text: alerts.append(title))
    QTest.qWait(9000)
    switch_to_report(w)
    wait_state(report.page)
    click(target, point("#tab-manage"))
    QTest.qWait(500)
    live = m.camera_thread.session_id
    t = table()
    live_row = [r for r in t if r[0] == live]
    check("记录中的课在列表最上方、状态为記録中、按钮全部不可用",
          t[0][0] == live and live_row and live_row[0][4] == "記録中" and live_row[0][5] is True, str(live_row))
    n_seen = len(driver.seen)
    state = press(button(live, "delete"), timeout=1.5)
    check("真实点击记录中的课的「削除」：不弹框、不改库", len(driver.seen) == n_seen and state == "stale"
          and rows(db, "SELECT status FROM sessions WHERE id = ?", live)[0]["status"] == "recording")

    n0 = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", live)[0]["n"]
    driver.expect(lambda d: QTest.mouseClick(d.yesButton, 1))
    state = press(button(b, "delete"))
    QTest.qWait(3000)
    n1 = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", live)[0]["n"]
    check("录制中删除另一节课：删除成功、记录继续写入、无「記録エラー」",
          state == "idle" and name_of(b) == "<deleted>" and n1 >= n0 + 2 and "記録エラー" not in alerts,
          f"样本 {n0} → {n1}，警报 {alerts}")

    def fill_dialog():
        for d in w.findChildren(CourseNameDialog):
            if d.isVisible():
                d.nameBox.setText("英語")
                QTest.mouseClick(d.yesButton, 1)

    w.switchTo(m)
    QTimer.singleShot(1500, fill_dialog)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    switch_to_report(w)
    wait_state(report.page)
    click(target, point("#tab-manage"))
    QTest.qWait(300)
    row = [r for r in table() if r[0] == live]
    check("停止并填写课程名后变为可操作", row and row[0][2] == "英語" and row[0][4] == "終了" and row[0][5] is False, str(row))
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    check("没有被拦截的请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    w.grab().save(str(shot_dir / "manage_after.png"))
    driver.timer.stop()
    w.close()
    QTest.qWait(500)

    print("\n[5] 删除后的数据残留（只报告）")
    for path in (db, Path(str(db) + "-wal")):
        if path.exists():
            data = path.read_bytes()
            hits = {name: data.count(name.encode()) for name in ("統計学", "削除テスト用の長い授業")}
            print(f"INFO  {path.name}（{len(data)} 字节）中已删除课程名出现次数：{hits}", flush=True)


if __name__ == "__main__":
    shot_dir = Path(tempfile.mkdtemp(prefix="verify_issue10_"))
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "manage.db", shot_dir)
    print(f"\n截图保存在：{shot_dir}")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
