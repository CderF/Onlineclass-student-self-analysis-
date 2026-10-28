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


def record(db, conn, started_at, scores, name=None, finish=True):
    """以 started_at 开始，每秒一帧写入 scores，返回课程记录 ID"""
    sid = session_store.create_session(conn, started_at)
    rec = SessionRecorder(sid, started_at, db)
    for i, score in enumerate(scores):
        rec.add_frame(frame(score=score), started_at + i)
    rec.close(started_at + len(scores))
    if finish:
        session_store.finish_session(conn, sid, started_at + len(scores))
    session_store.set_course_name(conn, sid, name)
    return sid


def test_list_sessions_empty(conn):
    assert report_data.list_sessions(conn) == []


def test_list_sessions_newest_first_with_summary(db, conn):
    old = record(db, conn, T0, [60, 80], name="線形代数")
    new = record(db, conn, T0 + 3600, [50, 70, 90], finish=False)
    session_store.mark_interrupted_sessions(conn)

    sessions = report_data.list_sessions(conn)
    assert [s["id"] for s in sessions] == [new, old]
    assert sessions[0] == {
        "id": new, "course_name": None, "started_at": T0 + 3600, "ended_at": T0 + 3602,
        "status": "interrupted", "seconds": 3, "avg_score": 70.0,
    }
    assert (sessions[1]["course_name"], sessions[1]["status"], sessions[1]["seconds"], sessions[1]["avg_score"]) == (
        "線形代数", "completed", 2, 70.0)


def test_list_sessions_includes_recording_without_samples(conn):
    sid = session_store.create_session(conn, T0)
    assert report_data.list_sessions(conn) == [{
        "id": sid, "course_name": None, "started_at": T0, "ended_at": None,
        "status": "recording", "seconds": 0, "avg_score": None,
    }]


def test_load_session_full_record(db, conn):
    sid = session_store.create_session(conn, T0)
    session_store.set_baseline(conn, sid, -4.5, 2.0)
    rec = SessionRecorder(sid, T0, db)
    rec.add_frame(frame(score=80), T0)
    rec.add_frame(frame(score=40, spatial="away", has_face=False, cognitive=None, pitch=None, yaw=None,
                        alert=("離席", "x")), T0 + 1)
    rec.add_frame(frame(score=60), T0 + 2)
    rec.pause(T0 + 2.5)
    rec = SessionRecorder(sid, T0 + 10, db)
    rec.add_frame(frame(score=90), T0 + 10)
    rec.close(T0 + 10.5)
    session_store.finish_session(conn, sid, T0 + 11)

    data = report_data.load_session(conn, sid)
    assert set(data) == {"session", "samples", "events", "pauses"}
    assert data["session"]["baseline_pitch"] == -4.5 and data["session"]["seconds"] == 4
    assert [s["t"] for s in data["samples"]] == [0, 1, 2, 3]
    assert [s["score"] for s in data["samples"]] == [80, 40, 60, 90]
    assert data["samples"][1]["spatial_state"] == "away" and data["samples"][1]["cognitive_state"] is None
    assert [(e["kind"], e["type"], e["t_start"], e["t_end"]) for e in data["events"]] == [
        ("spatial", "away", 1.0, 2.0), ("alert", "離席", 1.0, 1.0)]
    assert data["pauses"] == [{"t": 2.5, "paused_at": T0 + 2.5, "resumed_at": T0 + 10}]
    json.dumps(data, allow_nan=False)  # 能直接转成 JSON 交给页面


def test_load_session_missing(conn):
    assert report_data.load_session(conn, 999) is None


def test_load_session_isolated_from_other_sessions(db, conn):
    a = record(db, conn, T0, [10, 20])
    record(db, conn, T0 + 100, [30, 40, 50])
    data = report_data.load_session(conn, a)
    assert [s["score"] for s in data["samples"]] == [10, 20]


# ---------- 数据通道（ReportBridge） ----------

def test_bridge_returns_json(qapp, db, conn):
    from app.view.report_interface import ReportBridge
    sid = record(db, conn, T0, [70], name="線形代数 第3回")
    bridge = ReportBridge(db_path=db)

    listed = json.loads(bridge.list_sessions())
    assert listed["error"] is None and listed["data"][0]["course_name"] == "線形代数 第3回"
    assert "線形代数" in bridge.list_sessions()  # 日语原样输出，不转义
    loaded = json.loads(bridge.load_session(sid))
    assert loaded["error"] is None and loaded["data"]["samples"][0]["score"] == 70
    assert json.loads(bridge.load_session(999)) == {"data": None, "error": None}


def test_bridge_reports_db_error(qapp, tmp_path):
    from app.view.report_interface import ReportBridge
    bad = tmp_path / "is_a_dir"
    bad.mkdir()
    res = json.loads(ReportBridge(db_path=bad).list_sessions())
    assert res["data"] is None and res["error"]


def test_bridge_default_db_is_isolated(qapp, isolated_db):
    from app.view.report_interface import ReportBridge
    assert json.loads(ReportBridge().list_sessions()) == {"data": [], "error": None}
    assert isolated_db.exists()
