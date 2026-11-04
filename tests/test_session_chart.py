"""单节课图表（app/view/web/session_chart.js）：数据转换、由真实记录生成的图表配置、悬停 / 缩放 / 图例交互"""
import json

import pytest

from app import report_data, session_store
from app.session_recorder import SessionRecorder
from tests.test_report_interface import conn, db, open_report, run_js, select, wait_for  # noqa: F401
from tests.test_session_store import T0, frame

SPATIAL_GROUP = {
    "head_down": "notes", "taking_notes": "notes", "hidden_notes": "notes",
    "distracted": "distracted", "hidden_abnormal": "distracted",
    "away": "away", "absent": "away",
}


def js(report, expr):
    """在页面里求值并以 JSON 取回"""
    return json.loads(run_js(report.page, f"JSON.stringify({expr})"))


def option(report):
    return js(report, "echarts.getInstanceByDom(document.getElementById('score-chart')).getOption()")


def series(opt, name, type_=None):
    return [s for s in opt["series"] if s["name"] == name and (type_ is None or s["type"] == type_)]


@pytest.fixture
def page(open_report):
    return open_report()


# ---------- 纯函数 ----------

def test_format_time(page):
    assert js(page, "[0, 59.9, 61, 3725].map(SessionChart.formatTime)") == ["0分0秒", "0分59秒", "1分1秒", "1時間2分5秒"]


def test_cognitive_runs_merge_and_break(page):
    samples = [
        {"t": 0, "cognitive_state": "Understand"}, {"t": 1, "cognitive_state": "Understand"},
        {"t": 2, "cognitive_state": None},
        {"t": 3, "cognitive_state": "Doubt"}, {"t": 4, "cognitive_state": "Doubt"}, {"t": 5, "cognitive_state": "Doubt"},
        {"t": 7, "cognitive_state": "Doubt"},  # 暂停造成的时间轴空缺：不跨空缺合并
        {"t": 8, "cognitive_state": "Neutral"},
    ]
    assert js(page, f"SessionChart.cognitiveRuns({json.dumps(samples)})") == [
        {"state": "Understand", "start": 0, "end": 2},
        {"state": "Doubt", "start": 3, "end": 6},
        {"state": "Doubt", "start": 7, "end": 8},
        {"state": "Neutral", "start": 8, "end": 9},
    ]


def test_cognitive_runs_all_empty(page):
    assert js(page, "SessionChart.cognitiveRuns([{t: 0, cognitive_state: null}])") == []
    assert js(page, "SessionChart.cognitiveRuns([])") == []


@pytest.mark.parametrize("code, group", SPATIAL_GROUP.items())
def test_spatial_groups(page, code, group):
    events = [{"kind": "spatial", "type": code, "t_start": 1.5, "t_end": 4.0}]
    iv = js(page, f"SessionChart.spatialIntervals({json.dumps(events)}, 10)")
    assert [(i["group"], i["type"], i["start"], i["end"]) for i in iv] == [(group, code, 1.5, 4.0)]
    assert iv[0]["label"]


def test_spatial_intervals_open_event_and_filtering(page):
    events = [
        {"kind": "alert", "type": "absence", "t_start": 3.0, "t_end": 3.0},
        {"kind": "spatial", "type": "away", "t_start": 4.2, "t_end": None},  # 记录中仍在进行
        {"kind": "spatial", "type": "unknown_code", "t_start": 1.0, "t_end": 2.0},
    ]
    iv = js(page, f"SessionChart.spatialIntervals({json.dumps(events)}, 8)")
    assert [(i["type"], i["start"], i["end"]) for i in iv] == [("away", 4.2, 8)]


def test_alert_and_pause_marks(page):
    events = [{"kind": "alert", "type": "fatigue", "t_start": 12.5, "t_end": 12.5},
              {"kind": "alert", "type": "new_type", "t_start": 20.0, "t_end": 20.0}]
    assert js(page, f"SessionChart.alertMarks({json.dumps(events)})") == [
        {"type": "fatigue", "label": "疲労", "t": 12.5}, {"type": "new_type", "label": "new_type", "t": 20.0}]
    pauses = [{"t": 30.4, "paused_at": 100.0, "resumed_at": 400.0}, {"t": 50.0, "paused_at": 500.0, "resumed_at": None}]
    assert js(page, f"SessionChart.pauseMarks({json.dumps(pauses)})") == [{"t": 30.4, "seconds": 300.0}, {"t": 50.0, "seconds": None}]


def test_describe_second(page):
    ctx = """{
        byT: new Map([[100, {t: 100, score: 72.4, fatigue_index: 0.314, cognitive_state: 'Doubt'}]]),
        intervals: [{label: '離席中', start: 99.5, end: 120.0}],
        alerts: [{label: '離席継続', t: 104.2}],
        pauses: [{t: 100.3, seconds: 65}],
        problems: [{category: 'doubt', start: 90, end: 130}],
    }"""
    text = js(page, f"SessionChart.describeSecond({ctx}, 100)")
    assert text.split("<br>") == [
        "<b>1分40秒</b>", "エンゲージメント：72", "疲労指数：0.31", "認知状態：疑問",
        "離席中（1分39秒〜2分0秒）", "問題：理解不足の可能性（1分30秒〜2分10秒）", "一時停止（1分5秒）",
    ]
    # 容差内的警报也显示（缩小时一像素有好几秒）
    assert "アラート：離席継続" in js(page, f"SessionChart.describeSecond({ctx}, 100, 5)")
    assert js(page, f"SessionChart.describeSecond({ctx}, 130)").split("<br>") == ["<b>2分10秒</b>", "データなし"]


# ---------- 由真实记录生成的图表 ----------

def record_lesson(db, conn):
    """0–9 秒：前 3 秒理解，3–6 秒离席（无脸），6 秒警报；最后一帧 9 秒后暂停 300 秒；
    继续后 10–14 秒疑问、11 秒起低头直到结束（止于最后一帧 14 秒）"""
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(10):
        away = 3 <= i < 6
        rec.add_frame(frame(score=20 if away else 80, cognitive=None if away else "Understand",
                            spatial="away" if away else "normal", has_face=not away,
                            alert=("absence", "x", "y") if i == 6 else None), T0 + i)
    rec.pause(T0 + 9.5)
    rec = SessionRecorder(sid, T0 + 309.5, db)
    for i in range(5):
        rec.add_frame(frame(score=60, cognitive="Doubt", spatial="head_down" if i >= 1 else "normal"), T0 + 309.5 + i)
    rec.close(T0 + 314.5)
    session_store.finish_session(conn, sid, T0 + 315)
    return sid


def test_option_matches_record(db, conn, open_report):
    sid = record_lesson(db, conn)
    data = report_data.load_session(conn, sid)
    report = open_report()
    opt = option(report)

    ts = [s["t"] for s in data["samples"]]
    assert ts == list(range(15))
    end_min = (ts[-1] + 1) / 60
    assert all(abs(x["max"] - end_min) < 1e-9 and x["min"] == 0 for x in opt["xAxis"])

    assert opt["series"][0]["name"] == "エンゲージメント"
    assert [p[1] for p in opt["series"][0]["data"]] == [s["score"] for s in data["samples"]]
    assert [p[1] for p in series(opt, "疲労指数")[0]["data"]] == [s["fatigue_index"] for s in data["samples"]]

    def bars(name):
        return [[round(v * 60, 6) for v in d[:2]] for d in series(opt, name, "custom")[0]["data"]]

    assert bars("理解") == [[0, 3], [6, 10]]
    assert bars("疑問") == [[10, 15]]
    assert bars("中立") == [] and bars("嫌悪") == []

    db_events = [e for e in data["events"] if e["kind"] == "spatial"]
    assert [(e["type"], e["t_start"], e["t_end"]) for e in db_events] == [("away", 3.0, 6.0), ("head_down", 11.0, 14.0)]
    assert bars("離席") == [[3.0, 6.0]]
    assert bars("下向き・ノート") == [[11.0, 14.0]]
    assert bars("注意散漫") == []

    alerts = series(opt, "アラート", "scatter")[0]["data"]
    assert [round(a[0] * 60, 6) for a in alerts] == [6.0]

    pause_lines = series(opt, "一時停止")
    assert len(pause_lines) == 5  # 每行都有暂停竖线
    for s in pause_lines:
        assert [round(d["xAxis"] * 60, 6) for d in s["markLine"]["data"]] == [9.0]
    assert pause_lines[0]["markLine"]["data"][0]["text"] == "一時停止 5分0秒"

    assert opt["legend"][0]["data"] == ["理解不足の可能性", "疲労", "注意散漫・離席",
                                         "エンゲージメント", "疲労指数", "理解", "疑問", "中立", "嫌悪",
                                         "下向き・ノート", "注意散漫", "離席", "アラート", "一時停止"]


def test_rows_use_category_labels(db, conn, open_report):
    record_lesson(db, conn)
    opt = option(open_report())
    assert opt["yAxis"][2]["data"] == ["認知状態"]
    assert opt["yAxis"][3]["data"] == ["アラート", "離席", "注意散漫", "下向き・ノート"]
    row = opt["yAxis"][3]["data"]
    assert {s["name"]: {d[2] for d in s["data"]} for s in opt["series"]
            if s["type"] == "custom" and s["yAxisIndex"] == 3 and s["data"]} == {
        "離席": {row.index("離席")}, "下向き・ノート": {row.index("下向き・ノート")}}


def test_open_event_drawn_to_end_while_recording(db, conn, open_report):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(6):
        rec.add_frame(frame(spatial="distracted" if i >= 2 else "normal"), T0 + i)
    rec.conn.commit()  # 模拟正在记录：最后一秒尚未写入，事件未关闭
    opt = option(open_report())
    bar = series(opt, "注意散漫", "custom")[0]["data"]
    assert [[round(v * 60, 6) for v in d[:2]] for d in bar] == [[2.0, 5.0]]
    rec.close(T0 + 6)


def test_session_without_events_or_cognition(db, conn, open_report):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(3):
        rec.add_frame(frame(cognitive=None), T0 + i)
    rec.close(T0 + 3)
    session_store.finish_session(conn, sid, T0 + 3)
    opt = option(open_report())
    assert all(s["data"] == [] for s in opt["series"] if s["type"] in ("custom", "scatter"))
    assert all(s["markLine"]["data"] == [] for s in series(opt, "一時停止"))


# ---------- 交互 ----------

def hover(report, grid, minute, row):
    """在第 grid 行、横轴 minute 分处触发悬停提示，返回提示文字"""
    return run_js(report.page, f"""(() => {{
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        const [x, y] = c.convertToPixel({{xAxisIndex: {grid}, yAxisIndex: {grid}}}, [{minute}, {json.dumps(row)}]);
        c.dispatchAction({{type: 'showTip', x, y}});
        return document.getElementById('score-chart').innerText;
    }})()""")


@pytest.mark.parametrize("grid, row", [(0, 50), (1, 0.5), (2, "認知状態"), (3, "離席"), (3, "アラート"),
                                       (4, "注意散漫・離席")])
def test_tooltip_same_second_on_every_row(db, conn, open_report, grid, row):
    record_lesson(db, conn)
    text = hover(open_report(), grid, 4 / 60, row)
    assert "0分4秒" in text and "離席中（0分3秒〜0分6秒）" in text and "認知状態：なし" in text
    assert "NaN" not in text


def test_tooltip_inside_band_bar_uses_pointer_time(db, conn, open_report):
    record_lesson(db, conn)
    # 指向「疑問」矩形（10–15 秒）的中间：时间应为 12 秒，而不是被吸附到矩形起点 10 秒
    text = hover(open_report(), 2, 12 / 60, "認知状態")
    assert "0分12秒" in text and "認知状態：疑問" in text and "下向き（0分11秒〜0分14秒）" in text


def test_tooltip_when_curves_hidden(db, conn, open_report):
    record_lesson(db, conn)
    report = open_report()
    run_js(report.page, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        c.dispatchAction({type: 'legendUnSelect', name: 'エンゲージメント'});
        c.dispatchAction({type: 'legendUnSelect', name: '疲労指数'});
    })()""")
    assert "0分4秒" in hover(report, 3, 4 / 60, "離席")


def test_zoom_is_linked_across_rows(db, conn, open_report):
    record_lesson(db, conn)
    report = open_report()
    extents = js(report, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        c.dispatchAction({type: 'dataZoom', start: 50, end: 100});
        return [0, 1, 2, 3, 4].map((i) => c.getModel().getComponent('xAxis', i).axis.scale.getExtent());
    })()""")
    assert all(e == pytest.approx(extents[0]) for e in extents)
    assert extents[0] == pytest.approx([7.5 / 60, 15 / 60])


def test_legend_toggle(db, conn, open_report):
    record_lesson(db, conn)
    report = open_report()
    selected = js(report, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        c.dispatchAction({type: 'legendUnSelect', name: '一時停止'});
        return c.getOption().legend[0].selected;
    })()""")
    assert selected["一時停止"] is False and selected["エンゲージメント"] is True


def test_switching_session_replaces_chart(db, conn, open_report):
    a = record_lesson(db, conn)
    sid = session_store.create_session(conn, T0 + 1000)
    rec = SessionRecorder(sid, T0 + 1000, db)
    rec.add_frame(frame(score=50), T0 + 1000)
    rec.close(T0 + 1000.5)
    session_store.finish_session(conn, sid, T0 + 1001)
    report = open_report()
    assert len(option(report)["series"][0]["data"]) == 1
    select(report, a)
    wait_for(report.page, "document.getElementById('stat-duration').textContent", "0分15秒")
    opt = option(report)
    assert len(opt["series"][0]["data"]) == 15 and series(opt, "離席", "custom")[0]["data"]


def test_crosshair_distinct_from_pause_lines(db, conn, open_report):
    record_lesson(db, conn)
    opt = option(open_report())
    assert all(x["axisPointer"]["lineStyle"]["type"] == "solid" for x in opt["xAxis"])
    assert all(s["markLine"]["lineStyle"]["type"] == "dashed" for s in series(opt, "一時停止"))


# ---------- 问题时段（#7） ----------

def record_problem_lesson(db, conn):
    """200 秒：0–100 秒疑問、60–75 秒离席、120 秒起疲劳指数 0.9"""
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(200):
        away = 60 <= i < 75
        rec.add_frame(frame(cognitive=None if away else ("Doubt" if i < 100 else "Neutral"),
                            fatigue=0.9 if i >= 120 else 0.1, spatial="away" if away else "normal",
                            has_face=not away), T0 + i)
    rec.close(T0 + 200)
    session_store.finish_session(conn, sid, T0 + 200)
    return sid


def test_problem_row_and_shading_match_detection(db, conn, open_report):
    sid = record_problem_lesson(db, conn)
    problems = report_data.load_session(conn, sid)["problems"]
    assert {p["category"] for p in problems} == {"doubt", "fatigue", "distraction"}
    opt = option(open_report())

    assert opt["yAxis"][4]["data"] == ["注意散漫・離席", "疲労", "理解不足の可能性"]
    labels = {"doubt": "理解不足の可能性", "fatigue": "疲労", "distraction": "注意散漫・離席"}
    for key, label in labels.items():
        want = [(p["start"], p["end"]) for p in problems if p["category"] == key]
        row = opt["yAxis"][4]["data"].index(label)
        bar = series(opt, label, "custom")[0]
        assert [(round(d[0] * 60, 6), round(d[1] * 60, 6)) for d in bar["data"]] == want
        assert all(d[2] == row for d in bar["data"]) and bar["xAxisIndex"] == 4
        shade = series(opt, label, "line")[0]
        assert shade["xAxisIndex"] == 0
        assert [(round(a["xAxis"] * 60, 6), round(b["xAxis"] * 60, 6)) for a, b in shade["markArea"]["data"]] == want


def test_problem_in_tooltip(db, conn, open_report):
    record_problem_lesson(db, conn)
    text = hover(open_report(), 0, 30 / 60, 50)
    assert "問題：理解不足の可能性（0分0秒〜" in text


def test_problem_legend_toggles_row_and_shading(db, conn, open_report):
    record_problem_lesson(db, conn)
    report = open_report()
    selected = js(report, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        c.dispatchAction({type: 'legendUnSelect', name: '疲労'});
        return c.getOption().legend[0].selected;
    })()""")
    assert selected["疲労"] is False and selected["疲労指数"] is True
