"""
#11 CSV 导出的专项验证（补充 tests/test_export_csv.py）

运行：在仓库根目录执行（会短暂弹出真实窗口，约 1.5 分钟）
    python test-scripts/verify_issue11_export_csv.py

检查项：
  1. 真实记录：原生窗口用假摄像头记录一节课（含暂停 / 继续、离席），在「記録の管理」用真实 Qt 鼠标点「CSV」导出
  2. 独立核对：用 pandas 读回五个 CSV，与直接 SQL 查询逐值比对；问题时段与 problem_segments.detect 一致；
     用每个事件真实记录的墙上时间检验「听课时间 → 墙上时间」换算（跨暂停，误差应在一帧以内）
  3. 文件格式：UTF-8 BOM、CRLF、可按 UTF-8 解码（纯文本，不含图像）
  4. 录制中导出全部：记录中的课不在导出里，记录继续写入、无「記録エラー」
  5. 耗时：60 节 90 分钟课程全部导出的时间与体积（直接同步调用 export_csv.export；页面上的导出在后台线程，见 verify_issue16）
选择文件夹的对话框（macOS 原生）无法自动操作，脚本里替换为固定的临时文件夹；对话框本身需人工确认。
数据库与导出写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
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
DAY = 86400


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


def same(a, b):
    """CSV 读回的值与数据库值是否相同（NaN / 空 视为 None）"""
    import math
    if a is None or (isinstance(a, float) and math.isnan(a)):
        a = None
    if b is None or a is None:
        return a is None and b is None
    if isinstance(b, float) or isinstance(a, float):
        return float(a) == float(b)
    return str(a) == str(b)


def compare_export(folder: Path, db: Path, label: str):
    import pandas as pd
    from app.problem_segments import detect

    def load(name):
        # pandas 默认的浮点解析不保证与写出的十进制完全往返，核对时用 round_trip
        return pd.read_csv(folder / name, encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                           float_precision="round_trip")

    sessions = load("sessions.csv")
    ids = [int(i) for i in sessions["session_id"]]
    want_ids = [r["id"] for r in rows(db, "SELECT id FROM sessions WHERE status != 'recording' ORDER BY started_at, id")]
    check(f"{label}：sessions.csv 为全部非记录中的课（按开始时间）", ids == want_ids, f"{ids}")

    bad = []
    for table, cols, order in (
        ("samples", ["t", "wall_time", "score", "fatigue_index", "cognitive_state", "spatial_state", "face_ratio",
                     "delta_pitch", "delta_yaw", "frame_count"], "t"),
        ("events", ["kind", "type", "t_start", "t_end", "wall_start"], "t_start, id"),
        ("pauses", ["t", "paused_at", "resumed_at"], "t, id"),
    ):
        got = load(f"{table}.csv")
        want = []
        for sid in ids:
            want += [{"session_id": sid, **r} for r in rows(db, f"SELECT {', '.join(cols)} FROM {table} WHERE session_id = ? ORDER BY {order}", sid)]
        if len(got) != len(want):
            bad.append(f"{table} 行数 {len(got)} ≠ {len(want)}")
            continue
        for i, (g, w) in enumerate(zip(got.to_dict("records"), want)):
            diff = [c for c in ["session_id"] + cols if not same(g[c], w[c])]
            if diff:
                bad.append(f"{table} 第 {i} 行 {diff}")
                break
    check(f"{label}：samples / events / pauses 与 SQL 逐值一致", not bad, "; ".join(bad) or
          f"samples {len(load('samples.csv'))} 行，events {len(load('events.csv'))} 行，pauses {len(load('pauses.csv'))} 行")

    problems = load("problems.csv")
    want = []
    for sid in ids:
        samples = rows(db, "SELECT t, cognitive_state, fatigue_index FROM samples WHERE session_id = ? ORDER BY t", sid)
        events = rows(db, "SELECT kind, type, t_start, t_end FROM events WHERE session_id = ?", sid)
        pauses = rows(db, "SELECT t FROM pauses WHERE session_id = ?", sid)
        want += [(sid, p["category"], p["start"], p["end"]) for p in detect(samples, events, pauses)]
    got = [(int(r["session_id"]), r["category"], float(r["start"]), float(r["end"])) for r in problems.to_dict("records")]
    check(f"{label}：problems.csv 与独立调用 detect 一致", got == want, f"{len(got)} 段")

    # 换算检验：事件的 t_start 换成墙上时间，应与事件真实记录的 wall_start 一致
    from app.export_csv import wall_at
    worst = 0.0
    for sid in ids:
        smp = rows(db, "SELECT t, wall_time FROM samples WHERE session_id = ? ORDER BY t", sid)
        ts, walls = [s["t"] for s in smp], [s["wall_time"] for s in smp]
        for e in rows(db, "SELECT t_start, wall_start FROM events WHERE session_id = ?", sid):
            worst = max(worst, abs(wall_at(ts, walls, e["t_start"]) - e["wall_start"]))
    return worst


def file_format(folder: Path):
    raw = {p.name: p.read_bytes() for p in folder.iterdir()}
    ok = all(b.startswith(b"\xef\xbb\xbf") for b in raw.values())
    crlf = all(b"\r\n" in b and b.count(b"\n") == b.count(b"\r\n") for b in raw.values())
    try:
        for b in raw.values():
            b.decode("utf-8-sig")
        text = True
    except UnicodeDecodeError:
        text = False
    nul = any(b"\x00" in b for b in raw.values())
    return ok, crlf, text and not nul, sum(len(b) for b in raw.values())


def main(db: Path, perf_db: Path, out: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtCore import QTimer
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import export_csv, session_store
    from verify_issue5_report import run_js, switch_to_report, wait_state
    from verify_issue8_summary import click
    from verify_issue10_record_management import lesson

    session_store.DEFAULT_DB_PATH = db
    now = time.time()
    lesson(db, now - 3 * DAY, 600, "線形代数")
    lesson(db, now - 2 * DAY, 300, "統計学", interrupted=True)

    import main as app_main
    import verify_issue4_recording as v4
    from app.view.course_name_dialog import CourseNameDialog
    camera = v4.install_fake_camera()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 850)
    w.show()
    report, m = w.reportInterface, w.monitorInterface
    target = report.webView.focusProxy() or report.webView
    report.bridge.choose_export_dir = lambda start: str(out)
    QTest.qWait(1500)

    def name_dialog(name):
        def fill():
            for d in w.findChildren(CourseNameDialog):
                if d.isVisible():
                    d.nameBox.setText(name)
                    QTest.mouseClick(d.yesButton, 1)
        QTimer.singleShot(1500, fill)

    def point(selector):
        return run_js(report.page, f"""(() => {{
            const el = document.querySelector('{selector}');
            el.scrollIntoView({{block: 'center'}});
            const r = el.getBoundingClientRect();
            return [r.left + r.width / 2, r.top + r.height / 2];
        }})()""")

    def press(selector, timeout=30.0):
        run_js(report.page, "document.body.dataset.manage = 'stale'")
        click(target, point(selector))
        end = time.time() + timeout
        while time.time() < end and run_js(report.page, "document.body.dataset.manage") != "idle":
            QTest.qWait(20)
        return run_js(report.page, "document.body.dataset.manage")

    def open_manage():
        switch_to_report(w)
        wait_state(report.page)
        click(target, point("#tab-manage"))
        QTest.qWait(400)

    def notice():
        return run_js(report.page, "[document.getElementById('manage-message').dataset.tone, document.getElementById('manage-message').textContent]")

    print("\n[1] 真实记录一节课（暂停 / 继续、离席）")
    QTest.mouseClick(m.startBtn, 1)
    QTest.qWait(9000)
    camera.show_face = False   # 离席 14 秒（注意散漫・離席 的最短时长为 10 秒）
    QTest.qWait(14000)
    camera.show_face = True
    QTest.qWait(2000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(3000)
    QTest.mouseClick(m.pauseBtn, 1)
    QTest.qWait(6000)
    name_dialog("英語, 第1回")
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    live = rows(db, "SELECT id FROM sessions ORDER BY id DESC LIMIT 1")[0]["id"]
    n_pause = rows(db, "SELECT COUNT(*) AS n FROM pauses WHERE session_id = ?", live)[0]["n"]
    check("记录完成，含暂停", rows(db, "SELECT status, course_name FROM sessions WHERE id = ?", live)[0]
          == {"status": "completed", "course_name": "英語, 第1回"} and n_pause == 1)

    open_manage()
    state = press(f'#record-table tr[data-id="{live}"] button[data-action="export"]')
    folders = sorted(out.iterdir())
    tone, text = notice()
    check("真实点击「CSV」导出这节课，页面显示导出位置",
          state == "idle" and len(folders) == 1 and tone == "ok" and text == f"1 件の記録を CSV で書き出しました：{folders[0]}",
          text)
    single = folders[0]
    import pandas as pd
    s = pd.read_csv(single / "sessions.csv", encoding="utf-8-sig")
    check("单节导出只含这节课，含逗号的课程名原样读回",
          list(s["session_id"]) == [live] and s["course_name"][0] == "英語, 第1回", str(list(s["course_name"])))

    print("\n[2] 全部导出与独立核对")
    state = press("#export-all")
    folders = sorted(out.iterdir())
    full = [f for f in folders if f != single][0]
    tone, text = notice()
    check("真实点击「すべて CSV で書き出す」", state == "idle" and tone == "ok" and text.startswith("3 件の記録"), text)
    worst = compare_export(full, db, "全部导出")
    check("听课时间 → 墙上时间换算与事件真实时刻一致（跨暂停，误差 < 0.1 秒）", worst < 0.1, f"最大误差 {worst * 1000:.1f} ms")
    probs = pd.read_csv(full / "problems.csv", encoding="utf-8-sig")
    live_probs = probs[probs["session_id"] == live]
    check("真实记录的离席被导出为问题时段，并带墙上时间",
          (live_probs["category"] == "distraction").any() and live_probs["wall_start_iso"].notna().all(),
          live_probs[["category", "start", "end", "wall_start_iso"]].to_string(index=False).replace("\n", " | "))

    print("\n[3] 文件格式")
    bom, crlf, text_only, size = file_format(full)
    check("UTF-8 BOM、CRLF 换行（Excel 直接打开不乱码）", bom and crlf)
    check("纯文本（可 UTF-8 解码、无 NUL），不含图像", text_only, f"5 个文件共 {size / 1024:.1f} KB")

    print("\n[4] 录制中导出全部")
    alerts = []
    w.switchTo(m)
    QTest.mouseClick(m.startBtn, 1)
    m.camera_thread.alert_signal.connect(lambda title, msg: alerts.append(title))
    QTest.qWait(8000)
    rec = m.camera_thread.session_id
    open_manage()
    n0 = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", rec)[0]["n"]
    state = press("#export-all")
    newest = sorted(out.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    ids = list(pd.read_csv(newest / "sessions.csv", encoding="utf-8-sig")["session_id"])
    QTest.qWait(3000)
    n1 = rows(db, "SELECT COUNT(*) AS n FROM samples WHERE session_id = ?", rec)[0]["n"]
    check("记录中的课不导出，记录继续写入、无「記録エラー」",
          state == "idle" and rec not in ids and len(ids) == 3 and n1 >= n0 + 2 and "記録エラー" not in alerts,
          f"导出 {ids}，样本 {n0} → {n1}")
    row_disabled = run_js(report.page, f"""document.querySelector('#record-table tr[data-id="{rec}"] button[data-action="export"]').disabled""")
    check("记录中的课的「CSV」按钮不可用", row_disabled is True)
    name_dialog("")
    w.switchTo(m)
    QTest.mouseClick(m.stopBtn, 1)
    QTest.qWait(2500)
    check("页面没有 JS 错误", report.page.js_errors == [], str(report.page.js_errors))
    check("没有被拦截的请求", report.interceptor.blocked == [], str(report.interceptor.blocked))
    w.close()
    QTest.qWait(500)

    print("\n[5] 耗时（60 节 90 分钟课程）")
    from verify_issue8_summary import seed
    seed(perf_db, 60, 90, random.Random(11), time.time() - 100 * DAY)
    conn = session_store.connect(perf_db)
    perf_out = out / "perf"
    perf_out.mkdir()
    start = time.perf_counter()
    res = export_csv.export(conn, perf_out)
    sec = time.perf_counter() - start
    conn.close()
    size = sum(p.stat().st_size for p in Path(res["dir"]).iterdir())
    check("全部导出 60 节 90 分钟课程 < 5 秒", sec < 5.0,
          f"{sec:.2f} 秒，samples {res['rows']['samples']} 行，共 {size / 1024 / 1024:.1f} MB（直接同步调用 export_csv.export；页面上的导出在后台线程，见 verify_issue16）")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "exports"
        out.mkdir()
        main(Path(d) / "export.db", Path(d) / "perf.db", out)
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
