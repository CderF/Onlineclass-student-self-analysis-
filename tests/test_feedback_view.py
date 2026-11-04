"""报告页「授業ごと」的文字反馈卡片（#9）"""
import json

import pytest

from app import report_data, session_store
from tests.test_report_interface import conn, db, open_report, run_js, select, wait_for  # noqa: F401
from tests.test_report_summary import lesson
from tests.test_session_store import T0


def js(report, expr):
    return json.loads(run_js(report.page, f"JSON.stringify({expr})"))


def card(report):
    return js(report, """{
        hidden: document.getElementById('feedback-card').hidden,
        summary: document.getElementById('feedback-summary').textContent,
        segmentsHidden: document.getElementById('feedback-segments-section').hidden,
        segments: [...document.querySelectorAll('#feedback-segments li')].map(li => li.textContent),
        dots: [...document.querySelectorAll('#feedback-segments .segment-dot')].map(d => getComputedStyle(d).backgroundColor),
        more: document.getElementById('feedback-more').hidden ? null : document.getElementById('feedback-more').textContent,
        hints: [...document.querySelectorAll('#feedback-hints li')].map(li => li.textContent),
        comparisonHidden: document.getElementById('feedback-comparison-section').hidden,
        comparisonTitle: document.getElementById('feedback-comparison-title').textContent,
        comparison: document.getElementById('feedback-comparison').textContent,
    }""")


def test_card_shows_generated_feedback(db, conn, open_report):
    sid = lesson(db, conn, T0, 300, name="線形代数", score=80, away=[(100, 160)], doubt=[(0, 90)])
    fb = report_data.load_session(conn, sid)["feedback"]
    c = card(open_report())
    assert c["hidden"] is False and c["summary"] == fb["summary"]
    assert c["segments"] == [s["text"] for s in fb["segments"]] and len(c["segments"]) == 2
    assert c["dots"] == ["rgb(235, 104, 52)", "rgb(27, 175, 122)"]  # 理解不足（橙）、注意散漫・離席（青绿）
    assert c["hints"] == fb["hints"] and c["more"] is None
    assert c["comparisonHidden"] is True  # 该课程名的第一节课


def test_no_problems_hides_segment_section(db, conn, open_report):
    lesson(db, conn, T0, 120, name="英語", score=80)
    c = card(open_report())
    assert c["segmentsHidden"] is True and c["summary"].endswith("気になる時間帯は見つかりませんでした。")
    assert c["hints"] == ["この調子で続けましょう。"]


def test_comparison_and_switching_sessions(db, conn, open_report):
    first = lesson(db, conn, T0, 60, name="線形代数", score=60)
    lesson(db, conn, T0 + 100, 60, name="線形代数", score=75)
    report = open_report()
    c = card(report)
    assert c["comparisonHidden"] is False and c["comparisonTitle"] == "前回との比較（線形代数）"
    assert c["comparison"] == "平均エンゲージメントは前回（60）より 15 高くなりました。"
    select(report, first)
    wait_for(report.page, "document.getElementById('feedback-comparison-section').hidden", True)
    assert card(report)["summary"].startswith("おおむね集中できていました（平均エンゲージメント 60）")


def test_more_segments_note(db, conn, open_report):
    away = [(i * 100, i * 100 + 20) for i in range(5)]
    lesson(db, conn, T0, 600, score=80, away=away)
    c = card(open_report())
    assert len(c["segments"]) == 3 and c["more"] == "ほかに 2 か所あります（グラフで確認できます）。"


def test_click_segment_zooms_chart(db, conn, open_report):
    sid = lesson(db, conn, T0, 600, score=80, away=[(300, 360)])
    (seg,) = report_data.load_session(conn, sid)["feedback"]["segments"]
    report = open_report()
    run_js(report.page, "document.querySelector('#feedback-segments .segment-link').click()")
    extents = js(report, """(() => {
        const c = echarts.getInstanceByDom(document.getElementById('score-chart'));
        return c.getOption().xAxis.map((_, i) => c.getModel().getComponent('xAxis', i).axis.scale.getExtent());
    })()""")
    want = [(seg["start"] - 30) / 60, (seg["end"] + 30) / 60]
    assert all(e == pytest.approx(want) for e in extents)


def test_card_hidden_for_session_without_data(db, conn, open_report):
    session_store.create_session(conn, T0 + 999)   # 刚开始记录、还没有数据
    lesson(db, conn, T0, 60, score=80)
    report = open_report()
    assert card(report)["hidden"] is True        # 默认显示最新的（没有数据的）那条
    assert report.page.js_errors == []
