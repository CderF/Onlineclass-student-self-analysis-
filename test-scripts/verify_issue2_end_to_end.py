"""
#2 自我分析功能第一版的端到端验证：「课中记录 → 课后回看 → 分析 → 文字反馈」整条链路

运行：在仓库根目录执行（会弹出真实窗口，约 2 分钟）
    python test-scripts/verify_issue2_end_to_end.py

程序按 `python main.py` 的真实启动流程运行（runpy 执行 main.py 的 __main__，原生平台），
只是数据库换成临时文件、摄像头换成假摄像头、CSV 导出的文件夹选择对话框换成固定目录。
一次启动内依次操作（按钮用真实 Qt 鼠标点击）：
  1. 课中记录：第 1 节课 开始 → 离席 14 秒 → 暂停 → 继续 → 终了，在真实的课程名对话框里填「線形代数」；
     第 2 节课 同一课程名
  2. 课后回看：点侧栏打开报告页，最新一节课被选中，有图表与反馈卡片（含「前回との比較（線形代数）」）
  3. 分析：在「記録の管理」点第 1 节课的行打开它，离席被判为「注意散漫・離席」问题时段，
     图表中有该行，反馈里写着「席を離れていた可能性があります」
  4. 多节课汇总：「全体の推移」显示 2 节课、课程别汇总里「線形代数」2 节
  5. CSV 导出：「すべて CSV で書き出す」导出 2 节课，问题时段与图表一致
  6. 离线：没有被拦截的网络请求、没有 JS 错误；关闭窗口后退出码 0、锁文件被删除
再启动一次程序：两节课都还在、状态为「終了」，没有被标为 interrupted；退出码 0。
最后在父进程里独立检查数据库：完整性、时间轴连续、不含任何二进制（图像）数据、临时目录里没有图像文件。
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
failures = []
COURSE = "線形代数"


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


# ---------- 子进程：真实的 main.py ----------

def child(db: str, out: str, mode: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SELF.parent))
    os.chdir(ROOT)
    import runpy

    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication

    from app import session_store
    session_store.DEFAULT_DB_PATH = Path(db)
    import verify_issue4_recording as v4
    camera = v4.install_fake_camera()
    real_exec = QApplication.exec

    def exec_(app):
        w = sys._getframe(1).f_globals["w"]          # main.py 的 __main__ 模块里的主窗口
        scenario = record_and_review if mode == "record" else reopen

        def run():
            try:
                scenario(w, Path(db), Path(out), camera)
            except Exception:
                check("场景执行时没有异常", False, traceback.format_exc().replace("\n", " | "))
            w.close()

        QTimer.singleShot(0, run)
        return real_exec()

    QApplication.exec = exec_
    sys.argv = [str(ROOT / "main.py")]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def page_helpers(w):
    from PyQt5.QtTest import QTest
    from verify_issue5_report import run_js
    from verify_issue8_summary import click
    report = w.reportInterface
    target = report.webView.focusProxy() or report.webView

    def point(selector):
        return run_js(report.page, f"""(() => {{
            const el = document.querySelector('{selector}');
            el.scrollIntoView({{block: 'center'}});
            const r = el.getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];
        }})()""")

    def press(selector, flag, done, timeout=15.0):
        """真实鼠标点击页面元素，等到 body[data-<flag>] 变为 done"""
        run_js(report.page, f"document.body.dataset.{flag} = 'stale'")
        click(target, point(selector))
        end = time.time() + timeout
        while time.time() < end and run_js(report.page, f"document.body && document.body.dataset.{flag}") not in done:
            QTest.qWait(20)
        QTest.qWait(300)
        return run_js(report.page, f"document.body.dataset.{flag}")

    def tap(selector):
        """真实鼠标点击页面元素（切换标签页等不改变 data-* 状态的操作）"""
        click(target, point(selector))
        QTest.qWait(400)

    def js(body):
        return json.loads(run_js(report.page, f"JSON.stringify((() => {{ {body} }})())"))

    return report, press, tap, js


SINGLE = """
    const opt = state.chart ? state.chart.getOption() : {series: []};
    const bars = opt.series.filter(s => s.name === '注意散漫・離席' && s.type !== 'line');
    return {
        selected: document.getElementById('session-select').value,
        options: [...document.querySelectorAll('#session-select option')].map(o => o.text),
        status: document.getElementById('stat-status').textContent,
        score: document.getElementById('stat-score').textContent,
        feedbackHidden: document.getElementById('feedback-card').hidden,
        summary: document.getElementById('feedback-summary').textContent,
        segments: [...document.querySelectorAll('#feedback-segments li')].map(li => li.textContent),
        hints: document.querySelectorAll('#feedback-hints li').length,
        comparisonHidden: document.getElementById('feedback-comparison-section').hidden,
        comparisonTitle: document.getElementById('feedback-comparison-title').textContent,
        points: opt.series.length ? opt.series[0].data.length : 0,
        distraction: bars.length ? bars[0].data.map(d => [d[0], d[1]]) : [],
    };"""


def record_and_review(w, db: Path, out: Path, camera) -> None:
    from PyQt5.QtCore import QTimer
    from PyQt5.QtTest import QTest
    from app import report_data, session_store
    from app.view.course_name_dialog import CourseNameDialog
    from verify_issue5_report import switch_to_report, wait_state

    w.resize(1100, 850)
    m = w.monitorInterface
    w.reportInterface.bridge.choose_export_dir = lambda start: str(out)
    QTest.qWait(1500)
    dialogs = []

    def name_dialog(name):
        def fill():
            for d in w.findChildren(CourseNameDialog):
                if d.isVisible():
                    dialogs.append(name)
                    d.nameBox.setText(name)          # QTest.keyClicks 打不了日语
                    QTest.mouseClick(d.yesButton, 1)
        QTimer.singleShot(1500, fill)

    print("\n[1] 课中记录", flush=True)
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(9000)
    camera.show_face = False                         # 离席 14 秒（注意散漫・離席 的最短时长为 10 秒）
    QTest.qWait(14000)
    camera.show_face = True
    QTest.qWait(3000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(3000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(6000)
    name_dialog(COURSE)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(12000)
    name_dialog(COURSE)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    sessions = rows(db, "SELECT id, status, course_name FROM sessions ORDER BY id")
    check("两节课都在真实的课程名对话框里填了课程名", dialogs == [COURSE, COURSE], str(dialogs))
    check("两节课都保存为 completed、课程名为「線形代数」",
          [(s["status"], s["course_name"]) for s in sessions] == [("completed", COURSE)] * 2, str(sessions))
    first, second = sessions[0]["id"], sessions[-1]["id"]
    pauses = rows(db, "SELECT resumed_at FROM pauses WHERE session_id = ?", first)
    away = rows(db, "SELECT t_start, t_end FROM events WHERE session_id = ? AND kind = 'spatial' AND type = 'away'", first)
    check("第 1 节课：1 次暂停已继续、离席被记为 away 事件",
          len(pauses) == 1 and pauses[0]["resumed_at"] is not None and len(away) == 1
          and 12 <= away[0]["t_end"] - away[0]["t_start"] <= 16,
          f"暂停 {len(pauses)} 次，away {[(round(a['t_start'], 1), round(a['t_end'], 1)) for a in away]}")

    print("\n[2] 课后回看", flush=True)
    report, press, tap, js = page_helpers(w)
    switch_to_report(w)                              # 真实点击侧栏
    state = wait_state(report.page)
    QTest.qWait(500)
    d = js(SINGLE)
    check("点侧栏打开报告页，两节课都在列表中，最新一节被选中",
          state == "ready" and len(d["options"]) == 2 and d["selected"] == str(second), f"{state}，{d['options']}")
    n2 = rows(db, "SELECT COUNT(*) AS n, AVG(score) AS a FROM samples WHERE session_id = ?", second)[0]
    check("概要与曲线和数据库一致", d["status"] == "終了" and d["score"] == str(round(n2["a"])) and d["points"] == n2["n"],
          f"{d['status']} / {d['score']} / {d['points']} 点，数据库 {n2['n']} 秒、平均 {n2['a']:.1f}")
    check("显示反馈卡片（总评、提示、与前回比较）",
          not d["feedbackHidden"] and d["summary"] and d["hints"] > 0 and not d["comparisonHidden"]
          and d["comparisonTitle"] == f"前回との比較（{COURSE}）", f"{d['summary']} / {d['comparisonTitle']}")

    print("\n[3] 分析：在「記録の管理」打开第 1 节课", flush=True)
    tap("#tab-manage")
    records = js("return [...document.querySelectorAll('#record-table tbody tr')].map(tr => tr.dataset.id);")
    check("「記録の管理」列出两节课", records == [str(second), str(first)], str(records))
    state = press(f'#record-table tr[data-id="{first}"] td', "state", ("ready", "error"))
    d = js(SINGLE)
    conn = session_store.connect(db)
    data = report_data.load_session(conn, first)
    conn.close()
    want = [[p["start"] / 60, p["end"] / 60] for p in data["problems"] if p["category"] == "distraction"]
    check("点击行后切到「授業ごと」并显示第 1 节课",
          state == "ready" and d["selected"] == str(first)
          and js("return document.getElementById('view-single').hidden;") is False, f"{state}，选中 {d['selected']}")
    check("离席被判为「注意散漫・離席」问题时段，图表中的该行与判定一致",
          len(want) == 1 and d["distraction"] == want, f"图表 {d['distraction']}，判定 {want}")
    check("反馈写出「席を離れていた可能性があります」",
          any("席を離れていた可能性があります" in s for s in d["segments"]), str(d["segments"]))
    check("第 1 节课没有「前回との比較」（同课程名的第一节）", d["comparisonHidden"])

    print("\n[4] 多节课汇总", flush=True)
    state = press("#tab-summary", "summary", ("ready", "error"))
    s = js("""
        const c = echarts.getInstanceByDom(document.getElementById('summary-chart'));
        return {
            columns: c ? c.getOption().xAxis[0].data.length : 0,
            table: [...document.querySelectorAll('#course-table tbody tr')].map(tr => [...tr.cells].slice(0, 2).map(td => td.textContent)),
        };""")
    check("「全体の推移」显示两节课、课程别汇总为「線形代数」2 节",
          state == "ready" and s["columns"] == 2 and s["table"] == [[COURSE, "2"]], f"{state}，{s}")

    print("\n[5] CSV 导出", flush=True)
    tap("#tab-manage")
    state = press("#export-all", "manage", ("idle",), timeout=30)
    folders = sorted(out.iterdir())
    files = sorted(p.name for p in folders[0].iterdir()) if len(folders) == 1 else []
    check("点「すべて CSV で書き出す」导出到一个新文件夹，含 5 个 CSV",
          state == "idle" and files == ["events.csv", "pauses.csv", "problems.csv", "samples.csv", "sessions.csv"],
          f"{state}，{files}")
    if files:
        import pandas as pd
        sess = pd.read_csv(folders[0] / "sessions.csv", encoding="utf-8-sig")
        # pandas 默认的浮点解析不保证读回原值（末位可能不同），按 round_trip 读取才能逐值比对
        probs = pd.read_csv(folders[0] / "problems.csv", encoding="utf-8-sig", float_precision="round_trip")
        exported = [[r.start / 60, r.end / 60] for r in probs.itertuples()
                    if r.session_id == first and r.category == "distraction"]
        check("sessions.csv 含两节课，problems.csv 的离席时段与图表一致",
              sorted(sess["session_id"]) == [first, second] and exported == want, f"导出 {exported}，图表 {want}")

    print("\n[6] 离线与退出前检查", flush=True)
    check("没有被拦截的网络请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))


def reopen(w, db: Path, out: Path, camera) -> None:
    from PyQt5.QtTest import QTest
    from verify_issue5_report import switch_to_report, wait_state
    print("\n[7] 再次启动程序", flush=True)
    w.resize(1100, 850)
    QTest.qWait(1500)
    report, press, tap, js = page_helpers(w)
    switch_to_report(w)
    state = wait_state(report.page)
    QTest.qWait(500)
    d = js(SINGLE)
    check("重启后两节课都还在，最新一节显示反馈卡片",
          state == "ready" and len(d["options"]) == 2 and not d["feedbackHidden"]
          and all(o.endswith(COURSE) for o in d["options"]), f"{state}，{d['options']}")
    check("两节课都没有被标为 interrupted",
          [r["status"] for r in rows(db, "SELECT status FROM sessions ORDER BY id")] == ["completed"] * 2)
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))


# ---------- 父进程 ----------

def launch(db: Path, out: Path, mode: str):
    cmd = [sys.executable, "-X", "faulthandler", str(SELF), "--child", str(db), str(out), mode]
    p = subprocess.Popen(cmd, cwd=ROOT, text=True, errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    for line in p.stdout:
        lines.append(line)
        if line.startswith(("PASS", "FAIL", "[")):
            print(line, end="", flush=True)
            if line.startswith("FAIL"):
                failures.append(line.split("  ")[1].strip())
    code = p.wait(timeout=60)
    if code != 0:
        print("      子进程输出末尾：\n" + "".join(lines[-30:]), flush=True)
    return code, "".join(lines)


def main(tmp: Path):
    db, out = tmp / "data" / "self_analysis.db", tmp / "export"
    out.mkdir()
    lock = db.with_name(db.name + ".lock")
    code, log = launch(db, out, "record")
    check("关闭窗口后程序正常退出（退出码 0，无 Fatal Python error），锁文件被删除",
          code == 0 and "Fatal Python error" not in log and not lock.exists(), f"退出码 {code}")
    if not db.exists():
        return
    code, log = launch(db, out, "reopen")
    check("再次启动后正常退出（退出码 0）", code == 0 and "Fatal Python error" not in log, f"退出码 {code}")

    print("\n[8] 数据库独立检查", flush=True)
    conn = sqlite3.connect(str(db))
    check("数据库完整性检查通过", conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok")
    blobs = {}
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'"):
        for col in [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]:
            n = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE typeof({col}) = 'blob'").fetchone()[0]
            if n:
                blobs[f"{table}.{col}"] = n
    check("数据库里没有任何二进制数据（只存数值，不存图像）", blobs == {}, str(blobs))
    gaps = []
    for (sid,) in conn.execute("SELECT id FROM sessions"):
        ts = [r[0] for r in conn.execute("SELECT t FROM samples WHERE session_id = ? ORDER BY t", (sid,))]
        if ts != list(range(len(ts))):
            gaps.append(sid)
    conn.close()
    check("每节课的听课时间轴从 0 开始连续（跨暂停）", gaps == [], f"不连续的课 {gaps}")
    images = [str(p.relative_to(tmp)) for p in tmp.rglob("*")
              if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".mp4", ".avi")]
    check("临时目录（数据库与导出）里没有图像或视频文件", images == [], str(images))
    size = sum(p.stat().st_size for p in db.parent.iterdir())
    print(f"      数据库文件共 {size / 1024:.0f} KB", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        start = time.monotonic()
        with tempfile.TemporaryDirectory() as d:
            main(Path(d))
        print(f"\n用时 {time.monotonic() - start:.0f} 秒")
        print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
        sys.exit(1 if failures else 0)
