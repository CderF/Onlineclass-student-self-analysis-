"""多节课汇总（#8）：report_data.summarize_sessions / course_summary"""
import json

import pytest

from app import report_data, session_store
from app.session_recorder import SessionRecorder
from tests.test_session_store import T0, frame


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.db"


@pytest.fixture
def conn(db):
    c = session_store.connect(db)
    yield c
    c.close()


def lesson(db, conn, start, seconds, name=None, score=80, away=(), doubt=(), finish=True):
    """每秒一帧；away / doubt 为 (开始秒, 结束秒) 区间"""
    sid = session_store.create_session(conn, start)
    rec = SessionRecorder(sid, start, db)
    for i in range(seconds):
        is_away = any(a <= i < b for a, b in away)
        is_doubt = any(a <= i < b for a, b in doubt)
        rec.add_frame(frame(score=score, spatial="away" if is_away else "normal", has_face=not is_away,
                            cognitive=None if is_away else ("Doubt" if is_doubt else "Neutral")), start + i)
    rec.close(start + seconds)
    if finish:
        session_store.finish_session(conn, sid, start + seconds)
    session_store.set_course_name(conn, sid, name)
    return sid


def test_empty(conn):
    assert report_data.summarize_sessions(conn) == {"sessions": [], "courses": []}


def test_sessions_ascending_with_problem_stats(db, conn):
    later = lesson(db, conn, T0 + 86400, 120, name="英語", away=[(10, 30), (60, 75)])
    first = lesson(db, conn, T0, 200, name="線形代数", score=60, doubt=[(0, 120)])
    session_store.create_session(conn, T0 + 99999)  # 刚开始、还没有数据：不计入

    data = report_data.summarize_sessions(conn)
    assert [s["id"] for s in data["sessions"]] == [first, later]
    a, b = data["sessions"]
    assert (a["course_name"], a["status"], a["seconds"], a["avg_score"]) == ("線形代数", "completed", 200, 60.0)
    assert b["problems"]["distraction"] == {"count": 2, "seconds": pytest.approx(20.0 + 15.0)}
    assert b["problems"]["doubt"] == {"count": 0, "seconds": 0.0}
    # 与单节课的判定结果一致
    for s in data["sessions"]:
        detected = report_data.load_session(conn, s["id"])["problems"]
        for cat, st in s["problems"].items():
            got = [p for p in detected if p["category"] == cat]
            assert st["count"] == len(got)
            assert st["seconds"] == pytest.approx(sum(p["end"] - p["start"] for p in got))
    json.dumps(data, allow_nan=False)


def test_course_summary_grouping_and_weighting(db, conn):
    lesson(db, conn, T0, 100, name="線形代数", score=50)
    lesson(db, conn, T0 + 100, 300, name="線形代数", score=90, away=[(0, 20)], finish=False)
    lesson(db, conn, T0 + 200, 60, name=None, score=70)
    lesson(db, conn, T0 + 50, 60, name="英語", score=40)
    session_store.mark_interrupted_sessions(conn)

    courses = report_data.summarize_sessions(conn)["courses"]
    # 最近上课的在前
    assert [c["course_name"] for c in courses] == [None, "線形代数", "英語"]
    la = courses[1]
    assert (la["sessions"], la["interrupted"], la["seconds"]) == (2, 1, 400)
    # 按听课秒数加权：(50*100 + 平均分*300) / 400
    second_avg = report_data.list_sessions(conn)[1]["avg_score"]
    assert la["avg_score"] == pytest.approx((50 * 100 + second_avg * 300) / 400)
    assert la["problems"]["distraction"] == {"count": 1, "seconds": pytest.approx(20.0)}
    assert courses[0]["sessions"] == 1 and courses[0]["interrupted"] == 0


def test_course_summary_pure_function():
    sessions = [
        {"course_name": "A", "status": "completed", "seconds": 10, "avg_score": 100.0, "started_at": 1.0,
         "problems": {"doubt": {"count": 1, "seconds": 5.0}, "fatigue": {"count": 0, "seconds": 0.0},
                      "distraction": {"count": 0, "seconds": 0.0}}},
        {"course_name": "A", "status": "interrupted", "seconds": 30, "avg_score": 60.0, "started_at": 2.0,
         "problems": {"doubt": {"count": 2, "seconds": 7.0}, "fatigue": {"count": 1, "seconds": 30.0},
                      "distraction": {"count": 0, "seconds": 0.0}}},
    ]
    (c,) = report_data.course_summary(sessions)
    assert c == {
        "course_name": "A", "sessions": 2, "interrupted": 1, "seconds": 40, "avg_score": 70.0,
        "problems": {"doubt": {"count": 3, "seconds": 12.0}, "fatigue": {"count": 1, "seconds": 30.0},
                     "distraction": {"count": 0, "seconds": 0.0}},
        "last_started_at": 2.0,
    }


def test_bridge_summary(qapp, db, conn):
    from app.view.report_interface import ReportBridge
    lesson(db, conn, T0, 30, name="線形代数")
    res = json.loads(ReportBridge(db_path=db).summarize_sessions())
    assert res["error"] is None and res["data"]["courses"][0]["course_name"] == "線形代数"
