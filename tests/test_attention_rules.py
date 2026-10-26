import numpy as np
import pytest

from app import attention_rules
from app.attention_rules import AttentionAnalyzer

NEUTRAL = np.zeros(7)
NEUTRAL[4] = 1.0
NO_FACE = (None, None, None, None, None)


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(attention_rules.time, "time", c)
    return c


def feed(analyzer, clock, seconds, *frame, fps=10):
    for _ in range(int(seconds * fps)):
        analyzer.process_frame(*frame)
        clock.now += 1.0 / fps


def at(analyzer, clock, t, frame=NO_FACE):
    clock.now = t
    return analyzer.process_frame(*frame)


def test_away_decays_then_absent_after_120s(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 10, NEUTRAL, 0.3, 0.1, 0.0, 0.0)
    t0 = clock.now
    first = at(a, clock, t0)
    assert first.spatial_state == "away"

    # 回归：宏观窗口清空（>60 秒）后曾回跳为 100 / 初始化中
    r61 = at(a, clock, t0 + 61)
    assert r61.spatial_state == "away" and 0 < r61.score < first.score

    r119 = at(a, clock, t0 + 119)
    assert r119.spatial_state == "away" and r119.score < 10

    r121 = at(a, clock, t0 + 121)
    assert r121.spatial_state == "absent" and r121.score == 0
    assert r121.alert is not None and r121.alert[0] == "absence"


def test_face_return_resets_absence_timer(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 10, NEUTRAL, 0.3, 0.1, 0.0, 0.0)
    at(a, clock, clock.now + 30)
    r = at(a, clock, clock.now + 1, (NEUTRAL, 0.3, 0.1, 0.0, 0.0))
    assert r.spatial_state == "normal" and r.cognitive_state == "Neutral"
    assert a.face_not_detected_time is None


def test_rules_still_run_without_model(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 16, None, 0.3, 0.1, -30.0, 0.0)
    r = a.process_frame(None, 0.3, 0.1, -30.0, 0.0)
    assert r.spatial_state == "taking_notes"
    assert r.cognitive_state is None

    b = AttentionAnalyzer()
    feed(b, clock, 5, None, 0.3, 0.1, 0.0, 0.0)
    t0 = clock.now
    at(b, clock, t0)
    assert at(b, clock, t0 + 121).spatial_state == "absent"


def test_head_down_under_15s_is_head_down(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 5, NEUTRAL, 0.3, 0.1, 0.0, 0.0)
    r = a.process_frame(NEUTRAL, 0.3, 0.1, -30.0, 0.0)
    assert r.spatial_state == "head_down"
    assert r.cognitive_state == "Neutral"


def test_hidden_with_low_score_stays_abnormal(clock):
    a = AttentionAnalyzer()
    # 闭眼低头 10 秒：疲劳指数升高，分数降到 60 以下
    feed(a, clock, 10, NEUTRAL, 0.1, 0.1, -30.0, 0.0)
    before = a.process_frame(NEUTRAL, 0.1, 0.1, -30.0, 0.0)
    assert before.score < 60
    t0 = clock.now + 0.1
    assert at(a, clock, t0).spatial_state == "hidden_notes"
    # 回归：窗口清空后曾被误判为记笔记
    for sec in (10, 70, 130):
        r = at(a, clock, t0 + sec)
        assert r.spatial_state == "hidden_abnormal"
        assert r.score <= before.score
    assert at(a, clock, t0 + 130).score == 0


def test_hidden_with_high_score_stays_taking_notes(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 10, NEUTRAL, 0.3, 0.1, -30.0, 0.0)
    t0 = clock.now
    for sec in (0, 10, 70, 130):
        r = at(a, clock, t0 + sec)
        assert r.spatial_state == "hidden_notes" and r.score >= 60


def test_no_face_frames_have_no_cognitive_state(clock):
    a = AttentionAnalyzer()
    feed(a, clock, 3, NEUTRAL, 0.3, 0.1, 0.0, 0.0)
    r = at(a, clock, clock.now)
    assert not r.has_face and r.cognitive_state is None


def test_cognitive_state_uses_disgust_label(clock):
    a = AttentionAnalyzer()
    disgust = np.zeros(7)
    disgust[a.EMOTION_IDX["Disgust"]] = 1.0
    r = a.process_frame(disgust, 0.3, 0.1, 0.0, 0.0)
    assert r.cognitive_state == "Disgust"
