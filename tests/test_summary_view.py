"""报告页「全体の推移」（#8）：summary_chart.js 的数据转换、由真实记录生成的图表、筛选 / 汇总表 / 点击跳转"""
import json

import pytest

from app import report_data, session_store
from tests.test_report_interface import conn, db, open_report, reshow, run_js, wait_for  # noqa: F401
from tests.test_report_summary import lesson
from tests.test_session_store import T0

DAY = 86400


def js(report, expr):
    return json.loads(run_js(report.page, f"JSON.stringify({expr})"))


def open_summary(report):
    run_js(report.page, "document.getElementById('tab-summary').click()")
    wait_for(report.page, "document.body.dataset.summary", "ready")


def summary_option(report):
    return js(report, "echarts.getInstanceByDom(document.getElementById('summary-chart')).getOption()")


def table(report):
    return js(report, """[...document.querySelectorAll('#course-table tbody tr')].map(tr => ({
        key: tr.dataset.key, selected: tr.getAttribute('aria-selected'),
        cells: [...tr.cells].map(td => td.textContent)}))""")


@pytest.fixture
def three_courses(db, conn):
    """線形代数 ×2（第 2 回中断）、英語 ×1、授業名なし ×1；按时间：線形代数1、英語、線形代数2、なし"""
    ids = [
        lesson(db, conn, T0, 200, name="線形代数", score=60, doubt=[(0, 120)]),
        lesson(db, conn, T0 + DAY, 120, name="英語", score=80, away=[(10, 30)]),
        lesson(db, conn, T0 + 2 * DAY, 300, name="線形代数", score=90, finish=False),
        lesson(db, conn, T0 + 3 * DAY, 60, name=None, score=70),
    ]
    session_store.mark_interrupted_sessions(conn)
    return ids


@pytest.fixture
def page(open_report):
    return open_report()


# ---------- 纯函数 ----------

def test_format_and_labels(page):
    assert js(page, "[59, 60 * 45, 60 * 95].map(SummaryChart.formatMinutes)") == ["1分", "45分", "1時間35分"]
    labels = js(page, """[
        {started_at: 0, course_name: '英語', status: 'completed'},
        {started_at: 0, course_name: null, status: 'interrupted'},
        {started_at: 0, course_name: 'A', status: 'recording'},
    ].map(SummaryChart.axisLabel).map(l => l.split('\\n').slice(1))""")
    assert labels == [["英語"], ["授業名なし", "中断"], ["A", "記録中"]]


def test_filter_keys(page):
    sessions = [{"course_name": "A"}, {"course_name": None}, {"course_name": "A"}, {"course_name": "B"}]
    got = js(page, f"""['all', 'none', 'name:A', 'name:B', 'name:C']
        .map(k => SummaryChart.filterSessions({json.dumps(sessions)}, k).length)""")
    assert got == [4, 1, 2, 1, 0]
    assert js(page, "[SummaryChart.filterKey(null), SummaryChart.filterKey('none')]") == ["none", "name:none"]


def test_course_row_text(page):
    course = {"course_name": None, "sessions": 3, "interrupted": 1, "seconds": 3600, "avg_score": 71.6,
              "problems": {"doubt": {"count": 2, "seconds": 360}, "fatigue": {"count": 0, "seconds": 0},
                           "distraction": {"count": 1, "seconds": 90}}}
    assert js(page, f"SummaryChart.courseRow({json.dumps(course)})") == {
        "key": "none", "name": "授業名なし", "sessions": "3（うち中断 1）", "score": "72",
        "duration": "1時間0分", "problems": ["10%", "0%", "3%"],
    }


# ---------- 由真实记录生成的图表与汇总表 ----------

def test_chart_matches_summary(db, conn, three_courses, open_report):
    data = report_data.summarize_sessions(conn)
    report = open_report()
    open_summary(report)
    opt = summary_option(report)

    by_name = {s["name"]: s for s in opt["series"]}
    sessions = data["sessions"]
    assert [d["value"] for d in by_name["平均エンゲージメント"]["data"]] == [round(s["avg_score"], 1) for s in sessions]
    assert [d["symbol"] for d in by_name["平均エンゲージメント"]["data"]] == ["circle", "circle", "emptyCircle", "circle"]
    assert by_name["受講時間"]["data"] == [round(s["seconds"] / 60, 1) for s in sessions]
    for key, label in (("doubt", "理解不足の可能性"), ("fatigue", "疲労"), ("distraction", "注意散漫・離席")):
        want = [round(s["problems"][key]["seconds"] / s["seconds"] * 100, 1) for s in sessions]
        assert by_name[label]["data"] == want and "stack" not in by_name[label]  # 可能重叠，并列不堆叠
    assert by_name["理解不足の可能性"]["data"][0] > 0 and by_name["注意散漫・離席"]["data"][1] > 0
    assert opt["dataZoom"] == []  # 不多于 15 节课时不出现滑条
    assert opt["yAxis"][2]["max"] == 100


def test_course_table(db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    rows = table(report)
    assert [r["cells"][0] for r in rows] == ["授業名なし", "線形代数", "英語"]  # 最近上课的在前
    la = rows[1]["cells"]
    assert la[1] == "2（うち中断 1）" and la[3] == "8分"
    heads = js(report, "[...document.querySelectorAll('#course-table thead th')].map(th => th.textContent)")
    assert heads == ["授業名", "回数", "平均エンゲージメント", "合計受講時間", "理解不足の可能性", "疲労", "注意散漫・離席"]
    assert run_js(report.page, "document.getElementById('session-count').textContent") == "4"


def test_filter_by_course(db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    options = js(report, "[...document.querySelectorAll('#course-filter option')].map(o => [o.value, o.text])")
    assert options == [["all", "すべて"], ["none", "授業名なし"], ["name:線形代数", "線形代数"], ["name:英語", "英語"]]

    run_js(report.page, """(() => {
        const s = document.getElementById('course-filter');
        s.value = 'name:線形代数';
        s.dispatchEvent(new Event('change'));
    })()""")
    opt = summary_option(report)
    assert len(opt["series"][0]["data"]) == 2
    assert [r["selected"] for r in table(report)] == ["false", "true", "false"]

    # 点击汇总表的行：切换筛选；再点同一行取消
    run_js(report.page, "document.querySelector('#course-table tbody tr[data-key=\"none\"]').click()")
    assert len(summary_option(report)["series"][0]["data"]) == 1
    assert run_js(report.page, "document.getElementById('course-filter').value") == "none"
    run_js(report.page, "document.querySelector('#course-table tbody tr[data-key=\"none\"]').click()")
    assert len(summary_option(report)["series"][0]["data"]) == 4


def test_click_session_opens_single_view(db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    english = three_courses[1]
    for grid in (0, 2):  # 点分数行或问题时段行都可以
        run_js(report.page, f"""(() => {{
            const c = echarts.getInstanceByDom(document.getElementById('summary-chart'));
            const [x, y] = c.convertToPixel({{xAxisIndex: {grid}, yAxisIndex: {grid}}}, ['1', 10]);
            c.getZr().trigger('click', {{offsetX: x, offsetY: y}});
        }})()""")
        wait_for(report.page, "document.getElementById('session-select').value", str(english))
        assert run_js(report.page, "document.getElementById('view-single').hidden") is False
        wait_for(report.page, "document.getElementById('stat-duration').textContent", "2分0秒")
        open_summary(report)


def test_many_sessions_show_latest_with_slider(db, conn, open_report):
    for i in range(20):
        lesson(db, conn, T0 + i * DAY, 20, name="統計", score=50 + i)
    report = open_report()
    open_summary(report)
    zoom = summary_option(report)["dataZoom"]
    assert len(zoom) == 2 and zoom[0]["startValue"] == 5 and zoom[0]["endValue"] == 19


def test_summary_refreshes_when_page_shown(db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    lesson(db, conn, T0 + 10 * DAY, 30, name="英語")
    run_js(report.page, "document.body.dataset.summary = 'stale'")
    reshow(report)
    wait_for(report.page, "document.body.dataset.summary", "ready")
    assert len(summary_option(report)["series"][0]["data"]) == 5
    assert run_js(report.page, "document.getElementById('view-summary').hidden") is False


def test_tooltip_describes_session(db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    text = run_js(report.page, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('summary-chart'));
        const [x, y] = c.convertToPixel({xAxisIndex: 0, yAxisIndex: 0}, ['2', 50]);
        c.dispatchAction({type: 'showTip', x, y});
        return document.getElementById('summary-chart').innerText;
    })()""")
    assert "線形代数（中断）" in text and "平均エンゲージメント：90" in text and "受講時間：5分" in text
    assert "理解不足の可能性：0%（0回・0分0秒）" in text and "クリックでこの授業を表示" in text


def test_summary_error_message(tmp_path, db, conn, three_courses, open_report):
    report = open_report()
    open_summary(report)
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    report.bridge.db_path = bad
    run_js(report.page, "document.getElementById('tab-single').click(); document.getElementById('tab-summary').click()")
    wait_for(report.page, "document.body.dataset.summary", "error")
    assert run_js(report.page, "document.getElementById('summary-message').textContent") == "データを読み込めませんでした。"
