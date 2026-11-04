"""报告页「記録の管理」（#10）：列表、改课程名、删除（session_store + ReportBridge + 页面）"""
import json

import pytest

from app import report_data, session_store
from app.session_recorder import SessionRecorder
from tests.test_report_interface import conn, db, dom, open_report, run_js, wait_for  # noqa: F401
from tests.test_report_summary import lesson
from tests.test_session_store import T0, frame


def counts(conn, sid):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t} WHERE session_id = ?", (sid,)).fetchone()[0]
            for t in ("samples", "events", "pauses")}


def with_pause(db, conn, start, name=None):
    """有样本、事件（离席 + 警报）和一次暂停的课"""
    sid = session_store.create_session(conn, start)
    rec = SessionRecorder(sid, start, db)
    for i in range(20):
        rec.add_frame(frame(spatial="away" if 5 <= i < 10 else "normal", has_face=not 5 <= i < 10,
                            alert="absence" if i == 6 else None), start + i)
    rec.pause(start + 20)
    rec = SessionRecorder(sid, start + 30, db)
    for i in range(10):
        rec.add_frame(frame(), start + 30 + i)
    rec.close(start + 40)
    session_store.finish_session(conn, sid, start + 40)
    session_store.set_course_name(conn, sid, name)
    return sid


def status(conn, sid):
    row = conn.execute("SELECT course_name, status FROM sessions WHERE id = ?", (sid,)).fetchone()
    return None if row is None else (row["course_name"], row["status"])


# ---------- session_store ----------

def test_delete_cascades_and_keeps_others(db, conn):
    target = with_pause(db, conn, T0, name="線形代数")
    other = with_pause(db, conn, T0 + 100, name="英語")
    before = counts(conn, target)
    assert before["samples"] == 30 and before["events"] > 0 and before["pauses"] == 1
    assert session_store.delete_session(conn, target) is True
    assert status(conn, target) is None
    assert counts(conn, target) == {"samples": 0, "events": 0, "pauses": 0}
    assert counts(conn, other) == before and status(conn, other) == ("英語", "completed")
    assert session_store.delete_session(conn, target) is False  # 已不存在


def test_recording_session_is_not_deleted_or_renamed(db, conn):
    sid = lesson(db, conn, T0, 30, name="統計学", finish=False)
    assert session_store.delete_session(conn, sid) is False
    assert session_store.rename_session(conn, sid, "別名") is False
    assert status(conn, sid) == ("統計学", "recording") and counts(conn, sid)["samples"] == 30


def test_interrupted_session_can_be_managed(db, conn):
    sid = lesson(db, conn, T0, 30, finish=False)
    session_store.mark_interrupted_sessions(conn)
    assert session_store.rename_session(conn, sid, "英語") is True
    assert status(conn, sid) == ("英語", "interrupted")
    assert session_store.delete_session(conn, sid) is True


@pytest.mark.parametrize("name, stored", [("  線形代数 第3回 ", "線形代数 第3回"), ("", None), ("   ", None), (None, None)])
def test_rename_strips_and_clears(db, conn, name, stored):
    sid = lesson(db, conn, T0, 10, name="英語")
    assert session_store.rename_session(conn, sid, name) is True
    assert status(conn, sid) == (stored, "completed")


# ---------- ReportBridge ----------

@pytest.fixture
def bridge(qapp, db):
    from app.view.report_interface import ReportBridge
    b = ReportBridge(db_path=db)
    b.asked, b.confirmed = [], []
    return b


def answer(bridge, rename=None, delete=None):
    """替换对话框：rename 为 (是否保存, 课程名)，delete 为是否确认"""
    def ask(names, current):
        bridge.asked.append((names, current))
        return rename

    def confirm(session):
        bridge.confirmed.append(session)
        return delete

    bridge.ask_course_name, bridge.confirm_delete = ask, confirm


def test_bridge_rename(bridge, db, conn):
    lesson(db, conn, T0, 10, name="英語")
    sid = lesson(db, conn, T0 + 100, 10, name="線形代数")
    answer(bridge, rename=(True, " 統計学 "))
    assert json.loads(bridge.rename_session(sid)) == {"data": {"done": True}, "error": None}
    assert bridge.asked == [(["線形代数", "英語"], "線形代数")]   # 用过的课程名（最近的在前）与当前名
    assert status(conn, sid) == ("統計学", "completed")


def test_bridge_rename_cancel_and_clear(bridge, db, conn):
    sid = lesson(db, conn, T0, 10, name="英語")
    answer(bridge, rename=(False, None))
    assert json.loads(bridge.rename_session(sid))["data"] == {"done": False}
    assert status(conn, sid) == ("英語", "completed")
    answer(bridge, rename=(True, None))   # 清空后保存
    assert json.loads(bridge.rename_session(sid))["data"] == {"done": True}
    assert status(conn, sid) == (None, "completed")


def test_bridge_delete_confirm_and_cancel(bridge, db, conn):
    sid = with_pause(db, conn, T0, name="英語")
    answer(bridge, delete=False)
    assert json.loads(bridge.delete_session(sid))["data"] == {"done": False}
    assert status(conn, sid) == ("英語", "completed")
    answer(bridge, delete=True)
    assert json.loads(bridge.delete_session(sid)) == {"data": {"done": True}, "error": None}
    assert bridge.confirmed[0]["id"] == sid and bridge.confirmed[0]["course_name"] == "英語"
    assert status(conn, sid) is None and counts(conn, sid) == {"samples": 0, "events": 0, "pauses": 0}


@pytest.mark.parametrize("slot", ["rename_session", "delete_session"])
def test_bridge_refuses_recording_and_missing(bridge, db, conn, slot):
    sid = lesson(db, conn, T0, 10, name="英語", finish=False)
    answer(bridge, rename=(True, "x"), delete=True)
    for target in (sid, sid + 1000):
        res = json.loads(getattr(bridge, slot)(target))
        assert res["data"] is None and res["error"]
    assert bridge.asked == [] and bridge.confirmed == []   # 不弹对话框
    assert status(conn, sid) == ("英語", "recording")


def test_real_dialogs_construct(qapp):
    from PyQt5.QtWidgets import QWidget
    from app.view.course_name_dialog import RenameCourseDialog
    from app.view.report_interface import ReportBridge
    host = QWidget()
    host.resize(800, 600)
    dialog = RenameCourseDialog(["英語", "線形代数"], "線形代数", host)
    assert dialog.course_name() == "線形代数" and dialog.titleLabel.text() == "授業名を変更"
    assert dialog.cancelButton.text() == "キャンセル"
    dialog.nameBox.setText("  ")
    assert dialog.course_name() is None
    b = ReportBridge(parent=host)
    assert b._dialog_parent() is host.window()


# ---------- 页面 ----------

def table(report):
    return json.loads(run_js(report.page, """JSON.stringify([...document.querySelectorAll('#record-table tbody tr')].map(tr => ({
        id: Number(tr.dataset.id),
        cells: [...tr.cells].slice(0, 4).map(td => td.textContent),
        noName: tr.cells[1].classList.contains('no-name'),
        selected: tr.getAttribute('aria-selected'),
        buttons: [...tr.querySelectorAll('button')].map(b => [b.dataset.action, b.textContent, b.disabled, b.title]),
    })))"""))


def open_manage(report):
    run_js(report.page, "document.getElementById('tab-manage').click()")
    wait_for(report.page, "document.getElementById('view-manage').hidden", False)


def click_action(report, sid, action):
    run_js(report.page, f"""document.querySelector('#record-table tr[data-id="{sid}"] button[data-action="{action}"]').click()""")
    wait_for(report.page, "document.body.dataset.manage", "idle")


def test_table_lists_all_sessions(db, conn, open_report):
    old = lesson(db, conn, T0, 65, name="線形代数")
    mid = lesson(db, conn, T0 + 3600, 3725)
    rec = lesson(db, conn, T0 + 90000, 5, name="英語", finish=False)
    empty = session_store.create_session(conn, T0 + 95000)  # 刚开始、还没有数据
    report = open_report()
    open_manage(report)
    d = dom(report.page)
    assert d["singleHidden"] is True and d["summaryHidden"] is True
    rows = table(report)
    assert [r["id"] for r in rows] == [empty, rec, mid, old]
    labels = dict(dom(report.page)["options"])
    by_id = {r["id"]: r for r in rows}
    assert by_id[old]["cells"][1:] == ["線形代数", "1分5秒", "終了"]
    assert by_id[mid]["cells"][1:] == ["授業名なし", "1時間2分", "終了"] and by_id[mid]["noName"] is True
    assert by_id[rec]["cells"][3] == "記録中" and by_id[empty]["cells"][2] == "0分0秒"
    for sid in (old, mid):
        assert labels[str(sid)].startswith(by_id[sid]["cells"][0])  # 日時与下拉框一致
        assert by_id[sid]["buttons"] == [["export", "CSV", False, ""], ["rename", "名前を変更", False, ""], ["delete", "削除", False, ""]]
    for sid in (rec, empty):
        assert all(b[2] is True and b[3] == "記録中の授業は変更・削除・書き出しできません" for b in by_id[sid]["buttons"])
    assert by_id[empty]["selected"] == "true"   # 默认选中最新的


def test_rename_from_page(db, conn, open_report):
    sid = lesson(db, conn, T0, 30, name="英語")
    report = open_report()
    answer_page = []
    report.bridge.ask_course_name = lambda names, current: answer_page.append(current) or (True, "線形代数")
    open_manage(report)
    click_action(report, sid, "rename")
    assert answer_page == ["英語"] and status(conn, sid) == ("線形代数", "completed")
    assert table(report)[0]["cells"][1] == "線形代数"
    assert "線形代数" in dom(report.page)["options"][0][1]          # 「授業ごと」的下拉框也更新
    assert run_js(report.page, "document.getElementById('manage-message').hidden") is True
    assert run_js(report.page, "document.getElementById('view-manage').hidden") is False  # 留在本页


def test_delete_selected_falls_back_to_newest(db, conn, open_report):
    old = lesson(db, conn, T0, 30, name="線形代数", score=60)
    new = with_pause(db, conn, T0 + 100, name="英語")
    report = open_report()
    report.bridge.confirm_delete = lambda s: True
    open_manage(report)
    assert table(report)[0]["selected"] == "true"
    click_action(report, new, "delete")
    assert status(conn, new) is None and counts(conn, new) == {"samples": 0, "events": 0, "pauses": 0}
    rows = table(report)
    assert [r["id"] for r in rows] == [old] and rows[0]["selected"] == "true"
    d = dom(report.page)
    assert d["options"] == [[str(old), d["options"][0][1]]] and d["score"] == "60" and d["count"] == "1"


def test_delete_other_keeps_selection(db, conn, open_report):
    old = lesson(db, conn, T0, 30, name="線形代数", score=60)
    new = lesson(db, conn, T0 + 100, 30, name="英語", score=90)
    report = open_report()
    report.bridge.confirm_delete = lambda s: True
    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select'); s.value = '{old}'; s.dispatchEvent(new Event('change'));
    }})()""")
    wait_for(report.page, "document.getElementById('stat-score').textContent", "60")
    open_manage(report)
    click_action(report, new, "delete")
    assert [r["id"] for r in table(report)] == [old] and dom(report.page)["selected"] == str(old)


def test_cancel_changes_nothing(db, conn, open_report):
    sid = lesson(db, conn, T0, 30, name="英語")
    report = open_report()
    report.bridge.confirm_delete = lambda s: False
    report.bridge.ask_course_name = lambda names, current: (False, None)
    open_manage(report)
    before = table(report)
    click_action(report, sid, "delete")
    click_action(report, sid, "rename")
    assert status(conn, sid) == ("英語", "completed") and table(report) == before


def test_delete_last_record_shows_empty_state(db, conn, open_report):
    sid = lesson(db, conn, T0, 30)
    report = open_report()
    report.bridge.confirm_delete = lambda s: True
    open_manage(report)
    click_action(report, sid, "delete")
    d = dom(report.page)
    assert d["state"] == "empty" and "まだ記録がありません" in d["message"] and table(report) == []


def test_error_shows_notice_and_reloads(db, conn, open_report):
    """列表显示之后记录变为记录中（如另一个程序实例）：拒绝操作、提示并刷新为不可操作"""
    sid = lesson(db, conn, T0, 30, name="英語")
    report = open_report()
    report.bridge.confirm_delete = lambda s: pytest.fail("不应弹出确认")
    open_manage(report)
    with conn:
        conn.execute("UPDATE sessions SET status = 'recording' WHERE id = ?", (sid,))
    click_action(report, sid, "delete")
    assert status(conn, sid) == ("英語", "recording")
    assert run_js(report.page, "document.getElementById('manage-message').hidden") is False
    assert "操作できませんでした" in run_js(report.page, "document.getElementById('manage-message').textContent")
    assert all(b[2] for b in table(report)[0]["buttons"])
    # 之后成功的操作会清除提示
    with conn:
        conn.execute("UPDATE sessions SET status = 'completed' WHERE id = ?", (sid,))
    report.bridge.ask_course_name = lambda names, current: (True, "線形代数")
    run_js(report.page, "document.body.dataset.state = 'stale'")
    report.hide()
    report.show()
    wait_for(report.page, "document.body.dataset.state", "ready")
    click_action(report, sid, "rename")
    assert run_js(report.page, "document.getElementById('manage-message').hidden") is True


def test_row_click_opens_session(db, conn, open_report):
    old = lesson(db, conn, T0, 30, name="線形代数", score=60)
    lesson(db, conn, T0 + 100, 30, name="英語", score=90)
    report = open_report()
    open_manage(report)
    run_js(report.page, f"""document.querySelector('#record-table tr[data-id="{old}"] td').click()""")
    wait_for(report.page, "document.getElementById('stat-score').textContent", "60")
    d = dom(report.page)
    assert d["singleHidden"] is False and d["selected"] == str(old)
    assert run_js(report.page, "document.getElementById('tab-single').getAttribute('aria-selected')") == "true"


def test_button_click_does_not_open_session(db, conn, open_report):
    sid = lesson(db, conn, T0, 30, name="英語")
    report = open_report()
    report.bridge.ask_course_name = lambda names, current: (False, None)
    open_manage(report)
    click_action(report, sid, "rename")
    assert run_js(report.page, "document.getElementById('view-manage').hidden") is False


def test_summary_reflects_deletion(db, conn, open_report):
    a = lesson(db, conn, T0, 30, name="線形代数")
    lesson(db, conn, T0 + 100, 30, name="英語")
    report = open_report()
    report.bridge.confirm_delete = lambda s: True
    open_manage(report)
    click_action(report, a, "delete")
    run_js(report.page, "document.body.dataset.summary = 'stale'; document.getElementById('tab-summary').click()")
    wait_for(report.page, "document.body.dataset.summary", "ready")
    names = json.loads(run_js(report.page, "JSON.stringify([...document.querySelectorAll('#course-table tbody tr')].map(tr => tr.cells[0].textContent))"))
    assert names == ["英語"]
    assert report_data.summarize_sessions(conn)["courses"][0]["course_name"] == "英語"


# ---------- 对话框显示期间用截图代替网页（#18） ----------

def page_shown(report):
    return report.webView.isVisible(), report.snapshot.isVisible()


def test_page_frozen_swaps_in_snapshot(db, conn, open_report):
    lesson(db, conn, T0, 30, name="英語")
    report = open_report()
    open_manage(report)
    assert page_shown(report) == (True, False)
    from PyQt5.QtTest import QTest
    area = report.webView.geometry()
    with report.page_frozen():
        QTest.qWait(50)   # 让布局处理完
        assert page_shown(report) == (False, True)
        pix = report.snapshot.pixmap()
        assert not pix.isNull()
        assert (pix.width() / pix.devicePixelRatio(), pix.height() / pix.devicePixelRatio()) == (area.width(), area.height())
        assert report.snapshot.geometry() == area   # 占据网页原来的位置
    QTest.qWait(50)
    assert page_shown(report) == (True, False) and report.webView.geometry() == area
    assert report.snapshot.pixmap() is None or report.snapshot.pixmap().isNull()


def test_page_frozen_restores_on_error(db, conn, open_report):
    report = open_report()
    with pytest.raises(RuntimeError):
        with report.page_frozen():
            raise RuntimeError("dialog failed")
    assert page_shown(report) == (True, False)


@pytest.mark.parametrize("kind", ["delete", "rename"])
def test_real_dialog_methods_freeze_page(open_report, monkeypatch, kind):
    """真实的 confirm_delete / ask_course_name 在对话框 exec 期间网页被截图代替，之后恢复"""
    from app.view import report_interface
    report = open_report()
    seen = []

    class FakeDialog:
        def __init__(self, *args, **kwargs):
            self.yesButton = self.cancelButton = self

        def setText(self, text):
            pass

        def exec(self):
            seen.append(page_shown(report))
            return 1

        def course_name(self):
            return "線形代数"

    monkeypatch.setattr(report_interface, "MessageBox", FakeDialog)
    monkeypatch.setattr(report_interface, "RenameCourseDialog", FakeDialog)
    if kind == "delete":
        assert report.bridge.confirm_delete({"started_at": T0, "course_name": None}) is True
    else:
        assert report.bridge.ask_course_name(["英語"], "英語") == (True, "線形代数")
    assert seen == [(False, True)] and page_shown(report) == (True, False)


def test_bridge_without_report_parent_does_not_freeze(qapp, monkeypatch):
    from PyQt5.QtWidgets import QWidget
    from app.view import report_interface
    host = QWidget()
    bridge = report_interface.ReportBridge(parent=host)

    class FakeBox:
        def __init__(self, *args):
            self.yesButton = self.cancelButton = self

        def setText(self, text):
            pass

        def exec(self):
            return 0

    monkeypatch.setattr(report_interface, "MessageBox", FakeBox)
    assert bridge.confirm_delete({"started_at": T0, "course_name": "英語"}) is False
