"""报告页读库、判定与 CSV 导出在后台线程执行（#16）：ReportBridge.request / response_signal 与页面"""
import json
import threading
import time
from pathlib import Path

import pytest

from app import export_csv, report_data, session_store
from tests.test_record_management import open_manage
from tests.test_report_interface import conn, db, dom, open_report, run_js, select, wait_for  # noqa: F401
from tests.test_report_summary import lesson
from tests.test_session_store import T0


def pump_until(cond, timeout=10.0):
    from PyQt5.QtTest import QTest
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        QTest.qWait(10)
    return False


@pytest.fixture
def bridge(qapp, db):
    from app.view.report_interface import ReportBridge
    b = ReportBridge(db_path=db)
    b.responses = {}
    b.started = []
    b.response_signal.connect(lambda i, s: b.responses.__setitem__(i, json.loads(s)))
    b.export_started_signal.connect(b.started.append)
    yield b
    b.wait_for_background()


def threads_of(monkeypatch, module, name):
    seen = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        seen.append(threading.current_thread() is threading.main_thread())
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return seen


# ---------- ReportBridge ----------

@pytest.mark.parametrize("method, args, func", [
    ("list_sessions", [], "list_sessions"),
    ("load_session", ["SID"], "load_session"),
    ("summarize_sessions", [], "summarize_sessions"),
])
def test_reads_run_in_background_and_match_sync(bridge, db, conn, monkeypatch, method, args, func):
    sid = lesson(db, conn, T0, 120, name="英語", away=[(10, 40)])
    args = [sid if a == "SID" else a for a in args]
    want = json.loads(getattr(bridge, method)(*args))     # 同步调用（主线程）
    on_main = threads_of(monkeypatch, report_data, func)
    bridge.request(7, method, json.dumps(args))
    assert 7 not in bridge.responses                      # request 立即返回，结果之后送达
    assert pump_until(lambda: 7 in bridge.responses)
    assert on_main == [False]                             # 在后台线程执行
    assert bridge.responses[7] == want and want["error"] is None


def test_responses_matched_by_request_id(bridge, db, conn, monkeypatch):
    a = lesson(db, conn, T0, 60, score=60)
    b = lesson(db, conn, T0 + 100, 60, score=90)
    real = report_data.load_session

    def slow_for_a(c, sid):
        if sid == a:
            time.sleep(0.3)
        return real(c, sid)

    monkeypatch.setattr(report_data, "load_session", slow_for_a)
    bridge.request(1, "load_session", json.dumps([a]))
    bridge.request(2, "load_session", json.dumps([b]))
    assert pump_until(lambda: len(bridge.responses) == 2)
    assert bridge.responses[1]["data"]["session"]["id"] == a
    assert bridge.responses[2]["data"]["session"]["id"] == b


def test_background_error_is_reported(bridge, tmp_path):
    bridge.db_path = tmp_path   # 目录：打不开数据库
    bridge.request(3, "list_sessions", "[]")
    assert pump_until(lambda: 3 in bridge.responses)
    assert bridge.responses[3]["data"] is None and bridge.responses[3]["error"]


def test_unknown_method(bridge):
    bridge.request(4, "drop_everything", "[]")
    assert pump_until(lambda: 4 in bridge.responses)
    assert "未知的方法" in bridge.responses[4]["error"]


@pytest.mark.parametrize("method", ["rename_session", "delete_session"])
def test_dialog_methods_stay_on_ui_thread(bridge, db, conn, method):
    sid = lesson(db, conn, T0, 30, name="英語")
    on_main = []

    def ask(names, current):
        on_main.append(threading.current_thread() is threading.main_thread())
        return True, "線形代数"

    def confirm(session):
        on_main.append(threading.current_thread() is threading.main_thread())
        return True

    bridge.ask_course_name, bridge.confirm_delete = ask, confirm
    bridge.request(5, method, json.dumps([sid]))
    assert on_main == [True]                              # 对话框在界面线程、request 返回前已完成
    assert pump_until(lambda: 5 in bridge.responses)
    assert bridge.responses[5] == {"data": {"done": True}, "error": None}


@pytest.fixture
def gate(monkeypatch):
    """让后台导出停在开始处，直到 gate.set()"""
    event = threading.Event()
    real = export_csv.export
    calls = []

    def gated(*args, **kwargs):
        calls.append(threading.current_thread() is threading.main_thread())
        assert event.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(export_csv, "export", gated)
    event.calls = calls
    yield event
    event.set()


def test_export_runs_in_background_one_at_a_time(bridge, db, conn, tmp_path, gate):
    lesson(db, conn, T0, 30, name="英語")
    out = tmp_path / "out"
    out.mkdir()
    bridge.choose_export_dir = lambda start: str(out)
    bridge.request(10, "export_all", "[]")
    assert bridge.started == [10] and bridge.export_request == 10
    assert pump_until(lambda: gate.calls == [False])      # 写文件在后台线程
    bridge.request(11, "export_all", "[]")                # 进行中再次导出：拒绝
    assert pump_until(lambda: 11 in bridge.responses)
    assert "已有导出正在进行" in bridge.responses[11]["error"] and bridge.started == [10]
    gate.set()
    assert pump_until(lambda: 10 in bridge.responses)
    res = bridge.responses[10]
    assert res["error"] is None and res["data"]["done"] is True and Path(res["data"]["dir"]).parent == out
    assert bridge.export_request is None and len(list(out.iterdir())) == 1


def test_export_cancel_and_invalid_do_not_start(bridge, db, conn, tmp_path):
    sid = lesson(db, conn, T0, 30)
    rec = lesson(db, conn, T0 + 100, 30, finish=False)
    bridge.choose_export_dir = lambda start: None
    bridge.request(20, "export_session", json.dumps([sid]))
    bridge.request(21, "export_session", json.dumps([rec]))
    assert pump_until(lambda: {20, 21} <= set(bridge.responses))
    assert bridge.responses[20]["data"] == {"done": False, "dir": None, "sessions": 0}
    assert bridge.responses[21]["error"] and bridge.started == [] and bridge.export_request is None


def test_export_is_consistent_snapshot(db, conn, tmp_path, monkeypatch):
    """导出途中另一连接删除并改名：导出内容仍是开始时的状态"""
    a = lesson(db, conn, T0, 60, name="英語")
    b = lesson(db, conn, T0 + 100, 60, name="統計学")
    real = export_csv._collect
    done = []

    def collect_then_modify(c, sid, tables):
        real(c, sid, tables)
        if not done:
            done.append(sid)
            other = session_store.connect(db)
            other.execute("PRAGMA busy_timeout = 100")
            session_store.rename_session(other, a, "変更後")
            with other:
                other.execute("DELETE FROM sessions WHERE id = ?", (b,))
            other.close()

    monkeypatch.setattr(export_csv, "_collect", collect_then_modify)
    out = tmp_path / "out"
    out.mkdir()
    res = export_csv.export(conn, out)
    import csv
    with open(Path(res["dir"]) / "sessions.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    with open(Path(res["dir"]) / "samples.csv", encoding="utf-8-sig", newline="") as f:
        samples = list(csv.DictReader(f))
    assert [(int(r["session_id"]), r["course_name"]) for r in rows] == [(a, "英語"), (b, "統計学")]
    assert sum(r["session_id"] == str(b) for r in samples) == 60
    assert not conn.in_transaction
    assert [r[0] for r in conn.execute("SELECT id FROM sessions")] == [a]   # 修改确实发生了


def test_export_inside_callers_transaction_is_left_open(db, conn, tmp_path):
    lesson(db, conn, T0, 10)
    conn.execute("BEGIN")
    export_csv.export(conn, tmp_path)
    assert conn.in_transaction
    conn.rollback()


# ---------- 页面 ----------

def loading(report):
    return json.loads(run_js(report.page, "JSON.stringify({flag: document.body.dataset.loading,"
                                          " shown: !document.getElementById('loading').hidden})"))


def test_slow_read_shows_loading_after_delay(db, conn, open_report, monkeypatch):
    from PyQt5.QtTest import QTest
    lesson(db, conn, T0, 60, name="英語")
    report = open_report()
    assert loading(report) == {"flag": "false", "shown": False}
    real = report_data.summarize_sessions
    monkeypatch.setattr(report_data, "summarize_sessions", lambda c: (time.sleep(1.0), real(c))[1])
    run_js(report.page, "document.body.dataset.summary = 'stale'; document.getElementById('tab-summary').click()")
    QTest.qWait(100)
    assert loading(report) == {"flag": "true", "shown": False}      # 0.3 秒内不显示
    QTest.qWait(400)
    assert loading(report) == {"flag": "true", "shown": True}
    assert run_js(report.page, "document.getElementById('loading').textContent") == "読み込み中…"
    wait_for(report.page, "document.body.dataset.summary", "ready")
    assert loading(report) == {"flag": "false", "shown": False}


def test_fast_read_never_shows_loading(db, conn, open_report):
    from PyQt5.QtTest import QTest
    lesson(db, conn, T0, 60)
    report = open_report()
    run_js(report.page, """window.__shown = false;
        new MutationObserver(() => { if (!document.getElementById('loading').hidden) window.__shown = true; })
            .observe(document.getElementById('loading'), {attributes: true});""")
    run_js(report.page, "document.body.dataset.summary = 'stale'; document.getElementById('tab-summary').click()")
    wait_for(report.page, "document.body.dataset.summary", "ready")
    QTest.qWait(400)
    assert run_js(report.page, "window.__shown") is False


def test_latest_selection_wins_when_responses_arrive_out_of_order(db, conn, open_report, monkeypatch):
    slow = lesson(db, conn, T0, 60, score=60)
    fast = lesson(db, conn, T0 + 100, 60, score=90)
    lesson(db, conn, T0 + 200, 60, score=75)       # 最新的一节（打开时显示）
    report = open_report()
    real = report_data.load_session
    monkeypatch.setattr(report_data, "load_session",
                        lambda c, sid: (time.sleep(0.6) if sid == slow else None, real(c, sid))[1])
    select(report, slow)
    select(report, fast)          # 后发的请求先返回；慢的那个返回后也不应覆盖
    wait_for(report.page, "document.getElementById('stat-score').textContent", "90")
    from PyQt5.QtTest import QTest
    QTest.qWait(900)
    assert dom(report.page)["score"] == "90" and dom(report.page)["selected"] == str(fast)


def test_export_busy_disables_all_management(db, conn, open_report, tmp_path, gate):
    sid = lesson(db, conn, T0, 30, name="英語")
    out = tmp_path / "out"
    out.mkdir()
    report = open_report()
    asked = []
    report.bridge.choose_export_dir = lambda start: asked.append(start) or str(out)
    report.bridge.confirm_delete = lambda s: pytest.fail("导出期间不应能删除")
    open_manage(report)
    run_js(report.page, "document.getElementById('export-all').click()")
    assert pump_until(lambda: gate.calls == [False])
    state = json.loads(run_js(report.page, """JSON.stringify({
        manage: document.body.dataset.manage,
        disabled: [...document.querySelectorAll('#record-table button'), document.getElementById('export-all')].map(b => b.disabled),
        notice: document.getElementById('manage-message').textContent})"""))
    assert state["manage"] == "busy" and all(state["disabled"]) and state["notice"] == "書き出し中…"
    # 导出期间点击：不再弹出选择框、不删除
    run_js(report.page, f"""document.getElementById('export-all').click();
        document.querySelector('#record-table tr[data-id="{sid}"] button[data-action="delete"]').click();
        exportCsv('export_session', {sid}); manage('delete_session', {sid});""")
    from PyQt5.QtTest import QTest
    QTest.qWait(200)
    assert len(asked) == 1
    gate.set()
    wait_for(report.page, "document.body.dataset.manage", "idle")
    folder = next(out.iterdir())
    assert run_js(report.page, "document.getElementById('manage-message').textContent") == \
        f"1 件の記録を CSV で書き出しました：{folder}"
    assert run_js(report.page, "[...document.querySelectorAll('#record-table button')].some(b => b.disabled)") is False
    assert session_store.connect(db).execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


def test_shutdown_waits_for_background_then_destroy(db, conn, qapp, monkeypatch):
    """后台任务进行中关闭：shutdown 等任务结束（不死锁）、之后的请求被忽略，然后可以安全销毁"""
    from PyQt5 import sip
    from app.view.report_interface import ReportInterface
    lesson(db, conn, T0, 60)
    report = ReportInterface()
    report.show()
    wait_for(report.page, "document.body && document.body.dataset.state", "ready")
    real = report_data.summarize_sessions
    monkeypatch.setattr(report_data, "summarize_sessions", lambda c: (time.sleep(0.5), real(c))[1])
    run_js(report.page, "document.getElementById('tab-summary').click()")
    from PyQt5.QtTest import QTest
    QTest.qWait(100)
    assert report.bridge.pool.activeThreadCount() == 1
    start = time.time()
    report.shutdown()
    assert time.time() - start < 2.0 and report.bridge.pool.activeThreadCount() == 0
    report.bridge.request(99, "list_sessions", "[]")
    assert report.bridge.pool.activeThreadCount() == 0
    sip.delete(report)
    QTest.qWait(100)
