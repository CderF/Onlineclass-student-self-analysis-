"""
#16 报告页读库、判定与 CSV 导出不阻塞界面线程的专项验证

运行：在仓库根目录执行（会短暂弹出真实窗口，约 2 分钟）
    python test-scripts/verify_issue16_ui_responsiveness.py [--baseline]

用 1 ms 的定时器探测界面线程的事件循环：两次触发之间的最大间隔即界面线程单次阻塞的最长时间。
在 60 节 90 分钟课程的数据库上，通过真实窗口里的页面操作测量：
  1. 打开报告页（读取列表并显示最新一节）
  2. 切换到「全体の推移」直到显示完成（每节课跑一遍问题时段判定）
  3. 在「授業ごと」切换到另一节 90 分钟的课
  4. 「すべて CSV で書き出す」全部导出（选择文件夹的对话框替换为固定的临时文件夹）
完成标准：每项单次阻塞 < 100 ms，结果与同步调用 report_data / export_csv 一致。
另检查：读取超过 0.3 秒时显示「読み込み中…」；导出期间管理按钮不可用、不能重复触发。
--baseline 只打印测量值、不判定（用于记录修改前的数值）。
数据库与导出写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)

failures = []
BASELINE = "--baseline" in sys.argv
LIMIT_MS = 100
DAY = 86400


def check(name, cond, detail=""):
    if BASELINE:
        print(f"INFO  {name}" + (f"  ({detail})" if detail else ""), flush=True)
        return
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


class StallProbe:
    """
    1 ms 定时器；记录相邻两次触发的最大间隔（界面线程被占用的最长时间）。
    脚本用 QTest.qWait 等待，它在两次处理事件之间会休眠约 10 ms，因此测量分辨率约 10 ms。
    """

    def __init__(self):
        from PyQt5.QtCore import QElapsedTimer, Qt, QTimer
        self.clock = QElapsedTimer()
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(1)
        self.timer.timeout.connect(self._tick)
        self.last = None
        self.max_gap = 0.0

    def _tick(self):
        now = self.clock.nsecsElapsed() / 1e6
        if self.last is not None:
            self.max_gap = max(self.max_gap, now - self.last)
        self.last = now

    def start(self):
        # 从开始时刻算起：阻塞发生在第一次触发之前也要计入
        self.clock.start()
        self.last, self.max_gap = 0.0, 0.0
        self.timer.start()

    def stop(self):
        self._tick()          # 最后一次触发到结束之间的间隔也要计入
        self.timer.stop()
        return self.max_gap


def main(db: Path, out: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import export_csv, report_data, session_store
    from verify_issue5_report import run_js, switch_to_report
    from verify_issue8_summary import seed

    session_store.DEFAULT_DB_PATH = db
    t = time.time()
    seed(db, 60, 90, random.Random(16), time.time() - 100 * DAY)
    print(f"准备 60 节 90 分钟课程：{time.time() - t:.1f} 秒", flush=True)

    import main as app_main
    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 850)
    w.show()
    report = w.reportInterface
    report.bridge.choose_export_dir = lambda start: str(out)
    probe = StallProbe()
    QTest.qWait(2000)

    def until(expr, value, start, timeout=30.0, sample=None):
        """等页面状态达到 value；耗时从 start（触发操作前）算起；sample(秒) 时记录一次中间状态"""
        seen = None
        while time.time() - start < timeout:
            if sample is not None and seen is None and time.time() - start >= sample:
                seen = run_js(report.page, "JSON.stringify({loading: document.body.dataset.loading || null,"
                                           " shown: !document.getElementById('loading').hidden})")
            if run_js(report.page, expr) == value:
                return time.time() - start, seen
            QTest.qWait(2)
        return None, seen

    conn = session_store.connect(db)
    newest = report_data.list_sessions(conn)[0]["id"]

    print("\n[1] 打开报告页")
    run_js(report.page, "document.body.dataset.state = 'stale'")
    probe.start()
    t0 = time.time()
    switch_to_report(w)
    elapsed, _ = until("document.body.dataset.state", "ready", t0)
    stall = probe.stop()
    check(f"打开报告页：界面线程单次阻塞 < {LIMIT_MS} ms", elapsed is not None and stall < LIMIT_MS,
          f"最大阻塞 {stall:.0f} ms，显示完成 {elapsed if elapsed is None else round(elapsed, 2)} 秒")
    check("显示最新一节，内容与同步读取一致",
          run_js(report.page, "document.getElementById('session-select').value") == str(newest)
          and run_js(report.page, "document.getElementById('stat-score').textContent")
          == str(round(report_data.load_session(conn, newest)["session"]["avg_score"])))

    print("\n[2] 切换到「全体の推移」")
    run_js(report.page, "document.body.dataset.summary = 'stale'")
    probe.start()
    t0 = time.time()
    run_js(report.page, "document.getElementById('tab-summary').click()")
    elapsed, mid = until("document.body.dataset.summary", "ready", t0, sample=0.35)
    stall = probe.stop()
    check(f"切换到「全体の推移」：界面线程单次阻塞 < {LIMIT_MS} ms", elapsed is not None and stall < LIMIT_MS,
          f"最大阻塞 {stall:.0f} ms，显示完成 {elapsed if elapsed is None else round(elapsed, 2)} 秒")
    if not BASELINE and elapsed is not None and elapsed > 0.45:
        check("读取超过 0.3 秒时显示「読み込み中…」", mid is not None and json.loads(mid)["shown"] is True, str(mid))
    want = report_data.summarize_sessions(conn)
    got = json.loads(run_js(report.page, "JSON.stringify(state.summary)"))
    check("汇总结果与同步调用 summarize_sessions 一致", got == json.loads(json.dumps(want)),
          f"{len(got['sessions'])} 节、{len(got['courses'])} 个课程名")
    check("显示完成后不再显示「読み込み中…」", run_js(report.page, "document.getElementById('loading').hidden") in (True, None))

    print("\n[3] 在「授業ごと」切换到另一节课")
    other = report_data.list_sessions(conn)[30]["id"]
    run_js(report.page, "document.getElementById('tab-single').click()")
    QTest.qWait(300)
    run_js(report.page, "document.body.dataset.state = 'stale'")
    probe.start()
    t0 = time.time()
    run_js(report.page, f"""(() => {{ const s = document.getElementById('session-select');
        s.value = '{other}'; s.dispatchEvent(new Event('change')); }})()""")
    elapsed, _ = until("document.body.dataset.state", "ready", t0)
    stall = probe.stop()
    check(f"切换课程：界面线程单次阻塞 < {LIMIT_MS} ms", elapsed is not None and stall < LIMIT_MS,
          f"最大阻塞 {stall:.0f} ms，显示完成 {elapsed if elapsed is None else round(elapsed, 2)} 秒")
    check("图表点数等于该课的每秒样本数",
          run_js(report.page, "echarts.getInstanceByDom(document.getElementById('score-chart')).getOption().series[0].data.length")
          == len(report_data.load_session(conn, other)["samples"]))

    print("\n[4] 全部导出 CSV")
    run_js(report.page, "document.getElementById('tab-manage').click()")
    QTest.qWait(300)
    run_js(report.page, "document.body.dataset.manage = 'stale'")
    probe.start()
    t0 = time.time()
    run_js(report.page, "document.getElementById('export-all').click()")
    QTest.qWait(150)
    during = json.loads(run_js(report.page, """JSON.stringify({
        manage: document.body.dataset.manage,
        disabled: [...document.querySelectorAll('#record-table button'), document.getElementById('export-all')].every(b => b.disabled),
        notice: document.getElementById('manage-message').textContent})"""))
    if not BASELINE:
        run_js(report.page, "document.getElementById('export-all').click()")   # 导出期间再次点击：应被忽略
    elapsed, _ = until("document.body.dataset.manage", "idle", t0, timeout=60)
    stall = probe.stop()
    check(f"全部导出：界面线程单次阻塞 < {LIMIT_MS} ms", elapsed is not None and stall < LIMIT_MS,
          f"最大阻塞 {stall:.0f} ms，完成 {elapsed if elapsed is None else round(elapsed, 2)} 秒")
    check("导出期间显示「書き出し中…」，所有管理按钮不可用",
          during["manage"] == "busy" and during["disabled"] is True and "書き出し中" in during["notice"], str(during))
    folders = sorted(p for p in out.iterdir() if p.is_dir())
    notice = run_js(report.page, "document.getElementById('manage-message').textContent")
    check("只导出一次，完成后显示导出位置、按钮恢复可用",
          len(folders) == 1 and notice == f"60 件の記録を CSV で書き出しました：{folders[0]}"
          and run_js(report.page, "document.getElementById('export-all').disabled") is False, notice)
    if folders:
        ref = out / "ref"
        ref.mkdir()
        expected = Path(export_csv.export(conn, ref)["dir"])
        same = all((folders[0] / f).read_bytes() == (expected / f).read_bytes()
                   for f in ("sessions.csv", "samples.csv", "events.csv", "pauses.csv", "problems.csv"))
        check("后台导出的五个文件与同步调用 export_csv.export 逐字节相同", same)
    conn.close()
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    w.close()
    QTest.qWait(500)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "exports"
        out.mkdir()
        main(Path(d) / "perf.db", out)
    print("\nRESULT:", "BASELINE" if BASELINE else ("ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}"),
          flush=True)
    sys.exit(1 if failures else 0)
