import json
from dataclasses import replace

import pytest

from app.problem_segments import (
    CATEGORIES, DEFAULT_CONFIG, DISTRACTION, DOUBT, FATIGUE, detect,
)


def samples(n, start=0, cognitive="Neutral", fatigue=0.0, overrides=None):
    """t 从 start 起连续 n 秒；overrides: {t: {字段: 值}}"""
    rows = []
    for t in range(start, start + n):
        row = {"t": t, "cognitive_state": cognitive, "fatigue_index": fatigue, "spatial_state": "normal"}
        row.update((overrides or {}).get(t, {}))
        rows.append(row)
    return rows


def spatial(type_, start, end):
    return {"kind": "spatial", "type": type_, "t_start": start, "t_end": end, "wall_start": 0.0}


def pause(t):
    return {"t": t, "paused_at": 0.0, "resumed_at": 1.0}


def only(result, category):
    return [(p["start"], p["end"]) for p in result if p["category"] == category]


def covers_interior(intervals, lo=30.0, hi=270.0):
    """课程开头 / 结尾的窗口会被截断（有效秒数与占比随之变化），只断言中间部分被连成一段"""
    return len(intervals) == 1 and intervals[0][0] <= lo and intervals[0][1] >= hi


def test_empty_session():
    assert detect([], [], []) == []


def test_quiet_lesson_has_no_problems():
    assert detect(samples(600), [], []) == []


# ---------- 可能没听懂 ----------

def test_doubt_whole_lesson():
    assert only(detect(samples(120, cognitive="Doubt"), [], []), DOUBT) == [(0.0, 120.0)]


def test_doubt_burst_extent_follows_centered_window():
    # 疑問 [100,140)：以 i 为中心的 60 秒窗口 [i-30, i+30) 中疑問 ≥ 24 秒（40%）的 i 为 94..146
    rows = samples(300, overrides={t: {"cognitive_state": "Doubt"} for t in range(100, 140)})
    assert only(detect(rows, [], []), DOUBT) == [(94.0, 147.0)]


def test_short_doubt_burst_is_ignored():
    rows = samples(300, overrides={t: {"cognitive_state": "Doubt"} for t in range(100, 120)})
    assert only(detect(rows, [], []), DOUBT) == []  # 20/60 < 40%


@pytest.mark.parametrize("doubt_per_5, hit", [(2, True), (1, False)])
def test_doubt_ratio_threshold_inclusive(doubt_per_5, hit):
    # 每 5 秒中 2 秒疑問 = 完整窗口内恰好 40%（含边界）；1 秒 = 20%
    rows = samples(300, overrides={t: {"cognitive_state": "Doubt"} for t in range(300) if t % 5 < doubt_per_5})
    marked = only(detect(rows, [], []), DOUBT)
    assert covers_interior(marked) if hit else marked == []


@pytest.mark.parametrize("none_every, expected", [(2, [(29.0, 271.0)]), (3, [])])
def test_doubt_needs_enough_seconds_with_cognitive_state(none_every, expected):
    # 每 2 秒一次无结果 → 完整窗口内恰好 30 秒有效（够）；课程首尾窗口被截断、有效不足 30 秒，
    # 所以是 [29, 271)。每 3 秒中 2 秒无结果 → 20 秒（不够）
    if none_every == 2:
        overrides = {t: {"cognitive_state": None} for t in range(300) if t % 2}
    else:
        overrides = {t: {"cognitive_state": None} for t in range(300) if t % 3}
    rows = samples(300, cognitive="Doubt", overrides=overrides)
    assert only(detect(rows, [], []), DOUBT) == expected


def test_seconds_without_face_do_not_count_against_doubt():
    # 离席时认知状态为空：占比只按有认知状态的秒计算
    overrides = {t: {"cognitive_state": None} for t in range(0, 300) if t % 4 == 0}
    rows = samples(300, cognitive="Doubt", overrides=overrides)
    assert covers_interior(only(detect(rows, [], []), DOUBT))


# ---------- 疲劳 ----------

def test_fatigue_constant_high():
    assert only(detect(samples(100, fatigue=0.5), [], []), FATIGUE) == [(0.0, 100.0)]


@pytest.mark.parametrize("value, hit", [(0.38, True), (0.379, False)])
def test_fatigue_threshold_inclusive(value, hit):
    assert bool(only(detect(samples(100, fatigue=value), [], []), FATIGUE)) is hit


def test_fatigue_spike_below_min_duration_dropped():
    # 疲劳 1.0 持续 20 秒：30 秒均值 ≥0.38 的范围是 [97,124)，共 27 秒 < 30 秒 → 丢弃
    rows = samples(300, overrides={t: {"fatigue_index": 1.0} for t in range(100, 120)})
    assert only(detect(rows, [], []), FATIGUE) == []


def test_fatigue_spike_long_enough_kept():
    # 持续 25 秒：范围 [97,129)，共 32 秒
    rows = samples(300, overrides={t: {"fatigue_index": 1.0} for t in range(100, 125)})
    assert only(detect(rows, [], []), FATIGUE) == [(97.0, 129.0)]


@pytest.mark.parametrize("dip, expected", [
    (18, [(0.0, 154.0)]),                  # 30 秒均值把短低谷填平
    (20, [(0.0, 64.0), (77.0, 154.0)]),    # 判定后仍相隔 13 秒 > 10 秒，不合并
])
def test_fatigue_dip_between_high_periods(dip, expected):
    high = {t: {"fatigue_index": 1.0} for t in list(range(0, 60)) + list(range(60 + dip, 150))}
    assert only(detect(samples(200, overrides=high), [], []), FATIGUE) == expected


# ---------- 分心 / 离席 ----------

@pytest.mark.parametrize("state", ["distracted", "hidden_abnormal", "away", "absent"])
def test_distraction_states_counted(state):
    assert only(detect(samples(100), [spatial(state, 20.5, 40.0)], []), DISTRACTION) == [(20.5, 40.0)]


@pytest.mark.parametrize("state", ["head_down", "taking_notes", "hidden_notes"])
def test_head_down_and_notes_not_counted(state):
    assert detect(samples(100), [spatial(state, 20.0, 60.0)], []) == []


@pytest.mark.parametrize("length, kept", [(9.9, False), (10.0, True)])
def test_distraction_min_duration(length, kept):
    result = detect(samples(100), [spatial("away", 20.0, 20.0 + length)], [])
    assert bool(only(result, DISTRACTION)) is kept


@pytest.mark.parametrize("gap, expected", [
    (10.0, [(10.0, 45.0)]),
    (10.5, [(10.0, 20.0), (30.5, 45.5)]),
])
def test_distraction_merge_gap_inclusive(gap, expected):
    events = [spatial("away", 10.0, 20.0), spatial("distracted", 20.0 + gap, 35.0 + gap)]
    assert only(detect(samples(100), events, []), DISTRACTION) == expected


def test_short_events_merged_then_kept():
    # 各 6 秒、间隔 3 秒的三段离席：单独都短于 10 秒，合并后 24 秒
    events = [spatial("away", 10, 16), spatial("away", 19, 25), spatial("absent", 28, 34)]
    assert only(detect(samples(100), events, []), DISTRACTION) == [(10.0, 34.0)]


def test_open_event_runs_to_end_of_timeline():
    assert only(detect(samples(50), [spatial("away", 30.0, None)], []), DISTRACTION) == [(30.0, 50.0)]


def test_alert_events_ignored():
    alert = {"kind": "alert", "type": "absence", "t_start": 10.0, "t_end": 10.0, "wall_start": 0.0}
    assert detect(samples(100), [alert], []) == []


# ---------- 暂停 ----------

def test_windows_do_not_cross_pause():
    # 暂停（t=99.6，继续后从 100 开始）前不疲劳、后疲劳：判定不应提前到暂停之前
    rows = samples(100) + samples(100, start=100, fatigue=1.0)
    assert only(detect(rows, [], [pause(99.6)]), FATIGUE) == [(100.0, 200.0)]
    # 没有暂停时窗口会跨过 100，开始时间提前
    assert only(detect(rows, [], []), FATIGUE)[0][0] < 100.0


def test_doubt_window_does_not_cross_pause():
    rows = samples(100) + samples(100, start=100, cognitive="Doubt")
    assert only(detect(rows, [], [pause(99.2)]), DOUBT) == [(100.0, 200.0)]


def test_intervals_not_merged_across_pause():
    events = [spatial("away", 80.0, 99.5), spatial("away", 100.0, 120.0)]
    assert only(detect(samples(200), events, [pause(99.5)]), DISTRACTION) == [(80.0, 99.5), (100.0, 120.0)]
    assert only(detect(samples(200), events, []), DISTRACTION) == [(80.0, 120.0)]


def test_short_segments_after_pause_handled():
    # 暂停后只有 10 秒数据：窗口截断在段内，不报错
    rows = samples(100, cognitive="Doubt") + samples(10, start=100, cognitive="Doubt")
    assert only(detect(rows, [], [pause(99.9)]), DOUBT) == [(0.0, 100.0)]


def test_gap_in_timeline_splits_windows():
    rows = samples(50, cognitive="Doubt") + samples(50, start=60)
    assert only(detect(rows, [], []), DOUBT) == [(0.0, 50.0)]


# ---------- 输出与配置 ----------

def test_multiple_categories_sorted_and_json_ready():
    rows = samples(400, overrides={
        **{t: {"cognitive_state": "Doubt"} for t in range(200, 300)},
        **{t: {"fatigue_index": 0.9} for t in range(0, 100)},
    })
    result = detect(rows, [spatial("away", 150.0, 170.0)], [])
    assert [p["category"] for p in result] == [FATIGUE, DISTRACTION, DOUBT]
    assert [p["start"] for p in result] == sorted(p["start"] for p in result)
    assert set(p["category"] for p in result) <= set(CATEGORIES)
    json.dumps(result, allow_nan=False)


def test_overlapping_categories_reported_independently():
    rows = samples(200, cognitive="Doubt", fatigue=0.9)
    result = detect(rows, [spatial("distracted", 50.0, 100.0)], [])
    assert {p["category"] for p in result} == set(CATEGORIES)


def test_config_is_the_single_place_to_tune():
    rows = samples(300, overrides={t: {"cognitive_state": "Doubt"} for t in range(100, 120)})
    assert only(detect(rows, [], []), DOUBT) == []
    loose = replace(DEFAULT_CONFIG, doubt_ratio=0.3, min_duration={DOUBT: 0, FATIGUE: 30, DISTRACTION: 10})
    assert only(detect(rows, [], [], loose), DOUBT) != []
    strict = replace(DEFAULT_CONFIG, distraction_states=frozenset({"absent"}))
    assert detect(samples(100), [spatial("away", 0.0, 50.0)], [], strict) == []
