"""报告页（QWebEngineView + 本地 ECharts）在 offscreen 平台下的加载、数据显示与离线限制"""
import json
import time

import pytest

from app import session_store
from tests.test_report_data import record
from tests.test_session_store import T0

FINAL_STATES = ("ready", "empty", "error")


def run_js(page, script, timeout=5.0):
    from PyQt5.QtTest import QTest
    box = []
    page.runJavaScript(script, box.append)
    end = time.time() + timeout
    while not box and time.time() < end:
        QTest.qWait(20)
    assert box, f"runJavaScript 超时：{script}"
    return box[0]


def wait_for(page, script, expected, timeout=10.0):
    from PyQt5.QtTest import QTest
    end = time.time() + timeout
    value = None
    while time.time() < end:
        value = run_js(page, script)
        if value == expected or (callable(expected) and expected(value)):
            return value
        QTest.qWait(50)
    pytest.fail(f"{script} 未变为期望值，最后为 {value!r}")


def dom(page):
    return json.loads(run_js(page, """JSON.stringify({
        state: document.body.dataset.state,
        message: document.getElementById('message').textContent,
        options: [...document.querySelectorAll('#session-select option')].map(o => [o.value, o.text]),
        selected: document.getElementById('session-select').value,
        date: document.getElementById('stat-date').textContent,
        duration: document.getElementById('stat-duration').textContent,
        score: document.getElementById('stat-score').textContent,
        status: document.getElementById('stat-status').textContent,
        count: document.getElementById('session-count').textContent,
        singleHidden: document.getElementById('view-single').hidden,
        summaryHidden: document.getElementById('view-summary').hidden,
    })"""))


def chart_points(page):
    return run_js(page, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        return c ? c.getOption().series[0].data.length : -1;
    })()""")


def reshow(report):
    """模拟切换到别的页面再切回来；先把状态标为 stale，等到的 ready 就是这次刷新的结果"""
    run_js(report.page, "document.body.dataset.state = 'stale'")
    report.hide()
    report.show()
    wait_for(report.page, "document.body.dataset.state", lambda s: s in FINAL_STATES)


def select(report, session_id):
    run_js(report.page, f"""(() => {{
        const s = document.getElementById('session-select');
        s.value = '{session_id}';
        s.dispatchEvent(new Event('change'));
    }})()""")


@pytest.fixture
def db(isolated_db):
    return isolated_db


@pytest.fixture
def conn(db):
    c = session_store.connect(db)
    yield c
    c.close()


@pytest.fixture
def open_report(qapp):
    """创建并显示报告页，等页面读完数据；测试结束时检查没有 JS 错误和被拦截的请求"""
    from app.view.report_interface import ReportInterface
    pages = []

    def _open():
        report = ReportInterface()
        report.resize(1000, 700)
        report.show()
        pages.append(report)
        # 页面可能还没解析到 <body>
        wait_for(report.page, "document.body && document.body.dataset.state", lambda s: s in FINAL_STATES)
        return report

    yield _open
    from PyQt5 import sip
    for report in pages:
        assert report.page.js_errors == []
        sip.delete(report)  # 立即释放：留到进程退出时才销毁 WebEngine 页面会崩溃


def test_empty_db_shows_empty_state(open_report):
    report = open_report()
    d = dom(report.page)
    assert d["state"] == "empty"
    assert "まだ記録がありません" in d["message"]
    assert d["options"] == [] and d["count"] == "0"
    assert report.interceptor.blocked == []


def test_page_loads_local_echarts_and_channel(open_report):
    report = open_report()
    assert run_js(report.page, "echarts.version") == "6.1.0"
    assert run_js(report.page, "typeof qt.webChannelTransport") == "object"
    assert run_js(report.page, "document.documentElement.lang") == "ja"
    assert "Hiragino Sans" in run_js(report.page, "getComputedStyle(document.body).fontFamily")


def test_sessions_listed_and_newest_shown(db, conn, open_report):
    old = record(db, conn, T0, [60] * 90, name="線形代数")
    new = record(db, conn, T0 + 86400, [40, 50, 60], finish=False)
    session_store.mark_interrupted_sessions(conn)

    report = open_report()
    d = dom(report.page)
    assert d["state"] == "ready" and d["count"] == "2"
    assert [o[0] for o in d["options"]] == [str(new), str(old)]
    assert d["options"][0][1].endswith("授業名なし（中断）")
    assert d["options"][1][1].endswith("線形代数")
    assert d["selected"] == str(new)
    assert (d["duration"], d["score"], d["status"]) == ("0分3秒", "50", "中断")
    assert chart_points(report.page) == 3


def test_select_other_session(db, conn, open_report):
    old = record(db, conn, T0, [60] * 3700, name="線形代数")
    record(db, conn, T0 + 86400, [40, 50, 60])
    report = open_report()

    select(report, old)
    wait_for(report.page, "document.getElementById('stat-duration').textContent", "1時間1分")
    d = dom(report.page)
    assert (d["score"], d["status"]) == ("60", "終了")
    assert chart_points(report.page) == 3700


def test_refresh_when_page_shown_again(db, conn, open_report):
    report = open_report()
    assert dom(report.page)["state"] == "empty"

    a = record(db, conn, T0, [70, 80], name="英語")
    reshow(report)  # 切换回报告页时重新读取
    d = dom(report.page)
    assert d["state"] == "ready" and d["selected"] == str(a) and d["count"] == "1"

    # 有新记录（刚结束的课）时显示最新的一条
    b = record(db, conn, T0 + 86400, [10])
    reshow(report)
    d = dom(report.page)
    assert d["count"] == "2" and d["selected"] == str(b) and d["duration"] == "0分1秒"

    # 没有新记录时保持用户的选择
    select(report, a)
    wait_for(report.page, "document.getElementById('stat-duration').textContent", "0分2秒")
    reshow(report)
    assert dom(report.page)["selected"] == str(a)


def test_deleted_selection_falls_back_to_newest(db, conn, open_report):
    a = record(db, conn, T0, [70])
    b = record(db, conn, T0 + 100, [80])
    c = record(db, conn, T0 + 200, [90])
    report = open_report()
    select(report, b)
    wait_for(report.page, "document.getElementById('session-select').value", str(b))

    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (b,))
    reshow(report)
    d = dom(report.page)
    assert d["count"] == "2" and d["selected"] == str(c) and [o[0] for o in d["options"]] == [str(c), str(a)]


def test_tabs_switch_views(db, conn, open_report):
    record(db, conn, T0, [70])
    report = open_report()
    run_js(report.page, "document.getElementById('tab-summary').click()")
    d = dom(report.page)
    assert d["singleHidden"] is True and d["summaryHidden"] is False and d["count"] == "1"
    assert run_js(report.page, "document.getElementById('tab-summary').getAttribute('aria-selected')") == "true"
    run_js(report.page, "document.getElementById('tab-single').click()")
    d = dom(report.page)
    assert d["singleHidden"] is False and d["summaryHidden"] is True


def test_db_error_shows_message(tmp_path, open_report):
    report = open_report()
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    report.bridge.db_path = bad
    reshow(report)
    d = dom(report.page)
    assert (d["state"], d["message"]) == ("error", "データを読み込めませんでした。")


def test_network_requests_blocked(open_report):
    report = open_report()
    run_js(report.page, """(() => {
        const script = document.createElement('script');
        script.src = 'https://example.com/x.js';
        document.head.appendChild(script);
        new Image().src = 'http://example.com/a.png';
    })()""")
    from PyQt5.QtTest import QTest
    end = time.time() + 5
    while len(report.interceptor.blocked) < 2 and time.time() < end:
        QTest.qWait(50)
    assert sorted(report.interceptor.blocked) == ["http://example.com/a.png", "https://example.com/x.js"]
    report.page.js_errors.clear()  # 被拦截的加载本身会在 console 报错，属于预期
