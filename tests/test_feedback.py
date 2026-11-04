"""规则生成的日语文字反馈（#9）：app/feedback.py"""
import pytest

from app import feedback, report_data, session_store
from app.feedback import FeedbackContext, RuleFeedbackGenerator, duration_text, minute_range
from tests.test_report_summary import lesson
from tests.test_session_store import T0

GEN = RuleFeedbackGenerator()


def ctx(score=70.0, seconds=3600, problems=(), events=(), history=(), name="線形代数", status="completed"):
    return FeedbackContext(course_name=name, status=status, seconds=seconds, avg_score=score,
                           problems=list(problems), spatial_events=list(events), history_scores=list(history))


def prob(category, start, end):
    return {"category": category, "start": float(start), "end": float(end)}


def spatial(type_, start, end):
    return {"kind": "spatial", "type": type_, "t_start": float(start), "t_end": end, "wall_start": 0.0}


# ---------- 文字格式 ----------

@pytest.mark.parametrize("start, end, text", [
    (725, 1045, "12分〜17分"),      # 12:05–17:25 → 四舍五入
    (750, 1050, "13分〜18分"),      # 12:30–17:30（.5 进位）
    (3310, 3340, "55分〜56分"),
    (3310, 3320, "55分〜56分"),     # 落在同一分钟：至少跨 1 分钟
    (0, 60, "0分〜1分"),
    (3600, 3700, "1時間0分〜1時間2分"),
])
def test_minute_range(start, end, text):
    assert minute_range(start, end) == text


@pytest.mark.parametrize("seconds, text", [(0.3, "1秒"), (40, "40秒"), (59.4, "59秒"), (60, "1分"),
                                           (320, "5分"), (3700, "1時間2分")])
def test_duration_text(seconds, text):
    assert duration_text(seconds) == text


# ---------- 総評 ----------

@pytest.mark.parametrize("score, head", [
    (80, "全体としてよく集中できていました"), (75, "全体としてよく集中できていました"),
    (74.9, "おおむね集中できていました"), (60, "おおむね集中できていました"),
    (59.9, "集中が続きにくい授業だったようです"),
])
def test_summary_levels(score, head):
    text = GEN.generate(ctx(score=score))["summary"]
    assert text.startswith(head) and f"（平均エンゲージメント {round(score)}）" in text
    assert text.endswith("気になる時間帯は見つかりませんでした。")


@pytest.mark.parametrize("problems, head", [
    ([prob("doubt", 0, 1080)], "おおむね"),                         # 30%：降一档
    ([prob("doubt", 0, 1070)], "全体としてよく"),                   # 29.7%：不降
    # 重叠部分只算一次：0–700 与 400–1100 的并集是 1100 秒（30.6%）
    ([prob("doubt", 0, 700), prob("fatigue", 400, 1100)], "おおむね"),
    ([prob("doubt", 0, 500), prob("fatigue", 400, 900)], "全体としてよく"),  # 并集 900 秒（25%）
])
def test_heavy_problems_downgrade(problems, head):
    assert GEN.generate(ctx(score=80, problems=problems))["summary"].startswith(head)


def test_lowest_level_not_downgraded_and_suggests_review():
    text = GEN.generate(ctx(score=50, problems=[prob("doubt", 0, 3000)]))["summary"]
    assert text.startswith("集中が続きにくい") and text.endswith("気になった時間帯を振り返ってみましょう。")


def test_middle_level_with_problems_has_no_extra_sentence():
    text = GEN.generate(ctx(score=70, problems=[prob("doubt", 0, 60)]))["summary"]
    assert text == "おおむね集中できていました（平均エンゲージメント 70）。"


@pytest.mark.parametrize("status, suffix", [
    ("interrupted", "。この記録は途中で終了しています。"), ("recording", "。記録中のため、途中までの結果です。"),
])
def test_status_suffix(status, suffix):
    assert GEN.generate(ctx(status=status))["summary"].endswith(suffix)


# ---------- 気になった時間帯 ----------

def test_segment_texts_by_category():
    problems = [prob("doubt", 725, 1045), prob("fatigue", 2400, 2760), prob("distraction", 3310, 3370)]
    events = [spatial("away", 3310, 3370)]
    seg = GEN.generate(ctx(problems=problems, events=events))["segments"]
    assert [s["text"] for s in seg] == [
        "12分〜17分：理解が追いついていなかった可能性があります（5分）",
        "40分〜46分：疲れが見られました（6分）",
        "55分〜56分：席を離れていた可能性があります（1分）",
    ]
    assert [s["category"] for s in seg] == ["doubt", "fatigue", "distraction"]


@pytest.mark.parametrize("events, text", [
    ([spatial("away", 100, 130), spatial("distracted", 130, 140)], "席を離れていた"),
    ([spatial("absent", 100, 110), spatial("hidden_abnormal", 110, 140)], "注意がそれていた"),
    ([spatial("distracted", 100, None)], "注意がそれていた"),   # 进行中的事件画到末尾
    ([spatial("away", 100, None)], "席を離れていた"),
])
def test_distraction_split_by_dominant_state(events, text):
    seg = GEN.generate(ctx(seconds=200, problems=[prob("distraction", 100, 140)], events=events))["segments"]
    assert text in seg[0]["text"]


def test_only_longest_segments_listed_in_time_order():
    problems = [prob("doubt", 0, 100), prob("fatigue", 200, 260), prob("doubt", 400, 700),
                prob("distraction", 900, 920), prob("fatigue", 1000, 1200)]
    out = GEN.generate(ctx(problems=problems, events=[spatial("away", 900, 920)]))
    assert [s["start"] for s in out["segments"]] == [0.0, 400.0, 1000.0]
    assert out["more"] == 2


def test_no_segments_when_no_problems():
    out = GEN.generate(ctx())
    assert out["segments"] == [] and out["more"] == 0


# ---------- 次回へのヒント ----------

def test_hints_ordered_by_total_duration():
    problems = [prob("doubt", 0, 100), prob("fatigue", 200, 600), prob("distraction", 700, 720)]
    hints = GEN.generate(ctx(seconds=3600, problems=problems, events=[spatial("distracted", 700, 720)]))["hints"]
    assert hints == [feedback.HINT_TEXT["fatigue"], feedback.HINT_TEXT[feedback.DOUBT], feedback.HINT_TEXT["distracted"]]


@pytest.mark.parametrize("start, end, key", [
    (2200, 3400, "fatigue_late"),   # 全在后半
    (1500, 2500, "fatigue_late"),   # 后半 1800–2500 = 70% ≥ 60%
    (1600, 2600, "fatigue_late"),   # 后半 800/1000 = 80%
    (1300, 2300, "fatigue"),        # 后半 500/1000 = 50%
    (1000, 2000, "fatigue"),        # 后半 200/1000 = 20%
])
def test_fatigue_hint_late_or_not(start, end, key):
    hints = GEN.generate(ctx(seconds=3600, problems=[prob("fatigue", start, end)]))["hints"]
    assert hints == [feedback.HINT_TEXT[key]]


def test_away_hint_mentions_pause_button():
    hints = GEN.generate(ctx(problems=[prob("distraction", 0, 60)], events=[spatial("away", 0, 60)]))["hints"]
    assert hints == ["席を離れるときは「一時停止」を使うと、記録がより正確になります。"]


@pytest.mark.parametrize("score, key", [(80, "keep"), (60, "keep"), (55, "low_no_problem")])
def test_hints_without_problems(score, key):
    assert GEN.generate(ctx(score=score))["hints"] == [feedback.HINT_TEXT[key]]


# ---------- 前回との比較 ----------

@pytest.mark.parametrize("history, text", [
    ([66], "平均エンゲージメントは前回（66）より 6 高くなりました。"),
    ([60, 70, 68], "平均エンゲージメントはこれまでの 3 回の平均（66）より 6 高くなりました。"),
    ([80], "平均エンゲージメントは前回（80）より 8 低くなりました。"),
    ([67], "平均エンゲージメントは前回（67）と同程度でした。"),
    ([77], "平均エンゲージメントは前回（77）と同程度でした。"),   # 差 -5（含边界）
])
def test_comparison(history, text):
    out = GEN.generate(ctx(score=72, history=history))
    assert out["comparison"] == {"course_name": "線形代数", "text": text}


@pytest.mark.parametrize("name, history", [(None, [60]), ("線形代数", [])])
def test_no_comparison(name, history):
    assert GEN.generate(ctx(name=name, history=history))["comparison"] is None


def test_custom_config():
    strict = RuleFeedbackGenerator(feedback.FeedbackConfig(good_score=90, max_segments=1, same_margin=10))
    out = strict.generate(ctx(score=80, problems=[prob("doubt", 0, 60), prob("fatigue", 100, 400)], history=[72]))
    assert out["summary"].startswith("おおむね") and len(out["segments"]) == 1 and out["more"] == 1
    assert "同程度" in out["comparison"]["text"]


# ---------- 从数据库组装、可替换 ----------

@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.db"


@pytest.fixture
def conn(db):
    c = session_store.connect(db)
    yield c
    c.close()


def test_history_is_same_course_and_earlier_only(db, conn):
    lesson(db, conn, T0, 30, name="線形代数", score=60)
    lesson(db, conn, T0 + 10, 30, name="英語", score=10)
    lesson(db, conn, T0 + 20, 30, name="線形代数", score=70)
    target = lesson(db, conn, T0 + 30, 30, name="線形代数", score=90)
    lesson(db, conn, T0 + 40, 30, name="線形代数", score=20)      # 之后的课不算
    session_store.create_session(conn, T0 + 25)                   # 没有数据的不算
    data = report_data.load_session(conn, target)
    assert data["feedback"]["comparison"]["text"] == "平均エンゲージメントはこれまでの 2 回の平均（65）より 25 高くなりました。"


def test_first_lesson_of_course_has_no_comparison(db, conn):
    sid = lesson(db, conn, T0, 30, name="統計学")
    assert report_data.load_session(conn, sid)["feedback"]["comparison"] is None


def test_feedback_none_without_samples(conn):
    sid = session_store.create_session(conn, T0)
    assert report_data.load_session(conn, sid)["feedback"] is None


def test_generator_is_replaceable(db, conn, monkeypatch):
    class Fake:
        name = "fake"

        def generate(self, c):
            return {"summary": f"{c.course_name}:{c.seconds}:{len(c.problems)}", "segments": [], "more": 0,
                    "hints": [], "comparison": None, "generator": self.name}

    sid = lesson(db, conn, T0, 120, name="英語", away=[(10, 40)])
    monkeypatch.setattr(feedback, "GENERATOR", Fake())
    assert report_data.load_session(conn, sid)["feedback"]["summary"] == "英語:120:1"
    session = report_data.load_session(conn, sid)["session"]
    assert feedback.generate(conn, session, [], [], generator=RuleFeedbackGenerator())["generator"] == "rules"


def test_real_recording_feedback(db, conn):
    sid = lesson(db, conn, T0, 300, name="線形代数", score=80, away=[(100, 160)], doubt=[(0, 90)])
    out = report_data.load_session(conn, sid)["feedback"]
    # 分数 80 但问题时段（疑問约 2 分 + 离席 1 分）超过听课时间的 30%：降为「おおむね」
    assert out["summary"].startswith("おおむね集中できていました（平均エンゲージメント 80）")
    texts = [s["text"] for s in out["segments"]]
    assert any("席を離れていた可能性があります（1分）" in t for t in texts)
    assert any("理解が追いついていなかった" in t for t in texts)
