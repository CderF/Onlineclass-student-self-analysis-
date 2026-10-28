import pytest

from app import session_store
from app.attention_rules import AnalysisResult
from app.session_recorder import SessionRecorder

T0 = 1_000_000.0  # 墙上时间起点


def frame(score=80, fatigue=0.1, cognitive="Neutral", spatial="normal", has_face=True,
          pitch=0.0, yaw=0.0, alert=None) -> AnalysisResult:
    return AnalysisResult(
        score=score, status_text="", alert=alert, has_face=has_face, cognitive_state=cognitive,
        fatigue_index=fatigue, spatial_state=spatial, delta_pitch=pitch, delta_yaw=yaw,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.db"


@pytest.fixture
def conn(db):
    c = session_store.connect(db)
    yield c
    c.close()


def samples(conn, session_id):
    return [dict(r) for r in conn.execute("SELECT * FROM samples WHERE session_id = ? ORDER BY t", (session_id,))]


def events(conn, session_id, kind=None):
    sql = "SELECT * FROM events WHERE session_id = ?" + (" AND kind = ?" if kind else "") + " ORDER BY t_start, id"
    return [dict(r) for r in conn.execute(sql, (session_id, kind) if kind else (session_id,))]


# ---------- 表结构 ----------

def test_connect_creates_schema_once(db):
    c = session_store.connect(db)
    assert c.execute("PRAGMA user_version").fetchone()[0] == session_store.SCHEMA_VERSION
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"sessions", "samples", "events", "pauses"} <= tables
    c.close()
    session_store.connect(db).close()  # 再次打开不应重复建表


def test_connect_rejects_newer_schema(db):
    c = session_store.connect(db)
    c.execute("PRAGMA user_version = 99")
    c.close()
    with pytest.raises(RuntimeError):
        session_store.connect(db)


def record_one_second(db, sid, wall=T0):
    rec = SessionRecorder(sid, wall, db)
    rec.add_frame(frame(), wall)
    rec.close(wall + 0.5)


def test_session_lifecycle(conn, db):
    sid = session_store.create_session(conn, T0)
    session_store.set_baseline(conn, sid, -4.5, 2.0)
    record_one_second(db, sid)
    assert session_store.finish_session(conn, sid, T0 + 60) is True
    row = conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert (row["status"], row["ended_at"], row["baseline_pitch"], row["baseline_yaw"]) == ("completed", T0 + 60, -4.5, 2.0)
    assert row["course_name"] is None


# ---------- 每秒汇总 ----------

def test_per_second_aggregation_rules(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    # 第 0 秒：4 帧
    rec.add_frame(frame(score=80, fatigue=0.1, cognitive="Doubt", pitch=-2.0, yaw=1.0), T0 + 0.0)
    rec.add_frame(frame(score=60, fatigue=0.3, cognitive="Neutral", pitch=None, yaw=None, has_face=False), T0 + 0.25)
    rec.add_frame(frame(score=70, fatigue=0.2, cognitive="Doubt", pitch=-4.0, yaw=3.0), T0 + 0.5)
    rec.add_frame(frame(score=90, fatigue=0.2, cognitive=None, spatial="head_down", pitch=-20.0, yaw=2.0), T0 + 0.75)
    # 第 1 秒：认知状态全为空
    rec.add_frame(frame(cognitive=None, has_face=False, pitch=None, yaw=None, spatial="away"), T0 + 1.2)
    rec.close(T0 + 1.5)

    s0, s1 = samples(conn, sid)
    assert s0["t"] == 0 and s0["frame_count"] == 4 and s0["wall_time"] == T0
    assert s0["score"] == pytest.approx(75.0)
    assert s0["fatigue_index"] == pytest.approx(0.2)
    assert s0["cognitive_state"] == "Doubt"  # 众数，忽略空值
    assert s0["spatial_state"] == "head_down"  # 最后一帧
    assert s0["face_ratio"] == pytest.approx(0.75)
    assert s0["delta_pitch"] == pytest.approx((-2.0 - 4.0 - 20.0) / 3)  # 只平均有值的帧
    assert s0["delta_yaw"] == pytest.approx(2.0)

    assert s1["t"] == 1 and s1["cognitive_state"] is None
    assert s1["delta_pitch"] is None and s1["face_ratio"] == 0.0


def test_cognitive_tie_takes_first_seen(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i, state in enumerate(["Understand", "Doubt", "Doubt", "Understand"]):
        rec.add_frame(frame(cognitive=state), T0 + i * 0.2)
    rec.close(T0 + 0.9)
    assert samples(conn, sid)[0]["cognitive_state"] == "Understand"


def test_each_full_second_is_committed_immediately(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(25):  # 0.0 ~ 2.4 秒，10 fps
        rec.add_frame(frame(), T0 + i * 0.1)
    # 另一个连接只能看到已提交的数据：第 0、1 秒已写入，第 2 秒还在汇总中
    other = session_store.connect(db)
    assert [s["t"] for s in samples(other, sid)] == [0, 1]
    other.close()
    rec.close(T0 + 2.5)
    assert [s["t"] for s in samples(conn, sid)] == [0, 1, 2]


# ---------- 暂停与听课时间轴 ----------

def test_pause_resume_keeps_timeline_continuous(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(35):  # 0.0 ~ 3.4 秒
        rec.add_frame(frame(spatial="head_down" if i >= 30 else "normal"), T0 + i * 0.1)
    rec.pause(T0 + 3.45)

    pause = dict(conn.execute("SELECT * FROM pauses WHERE session_id = ?", (sid,)).fetchone())
    assert pause["t"] == pytest.approx(3.45) and pause["paused_at"] == T0 + 3.45 and pause["resumed_at"] is None
    # 暂停时关闭进行中的事件
    (ev,) = events(conn, sid, "spatial")
    assert (ev["type"], ev["t_start"], ev["t_end"]) == ("head_down", pytest.approx(3.0), pytest.approx(3.45))

    # 暂停 10 分钟后继续：时间轴从暂停点后的下一整秒接着走
    resume_wall = T0 + 600
    rec2 = SessionRecorder(sid, resume_wall, db)
    assert rec2.t_base == 4.0
    for i in range(15):
        rec2.add_frame(frame(), resume_wall + i * 0.1)
    rec2.close(resume_wall + 1.5)

    ts = [s["t"] for s in samples(conn, sid)]
    assert ts == [0, 1, 2, 3, 4, 5]
    pause = conn.execute("SELECT resumed_at FROM pauses WHERE session_id = ?", (sid,)).fetchone()
    assert pause["resumed_at"] == resume_wall
    # 墙上时间保留真实时刻，便于与外部问卷对齐
    assert samples(conn, sid)[4]["wall_time"] == resume_wall


def test_pause_without_frames_does_not_advance_timeline(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    rec.pause(T0 + 30)  # 例如摄像头打不开
    assert conn.execute("SELECT t FROM pauses").fetchone()["t"] == 0.0


# ---------- 事件 ----------

def test_spatial_events_open_and_close_on_state_change(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    sequence = ["normal"] * 5 + ["head_down"] * 5 + ["distracted"] * 5 + ["normal"] * 5 + ["away"] * 3
    for i, state in enumerate(sequence):
        rec.add_frame(frame(spatial=state), T0 + i * 0.1)
    rec.close(T0 + 2.3)

    got = [(e["type"], round(e["t_start"], 2), round(e["t_end"], 2)) for e in events(conn, sid, "spatial")]
    assert got == [("head_down", 0.5, 1.0), ("distracted", 1.0, 1.5), ("away", 2.0, 2.3)]
    assert all(e["wall_start"] == pytest.approx(T0 + e["t_start"]) for e in events(conn, sid, "spatial"))


def test_alerts_are_point_events(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    rec.add_frame(frame(), T0)
    rec.add_frame(frame(alert=("fatigue", "疲労アラート", "...")), T0 + 0.5)
    rec.close(T0 + 1)
    (ev,) = events(conn, sid, "alert")
    assert (ev["type"], ev["t_start"], ev["t_end"]) == ("fatigue", 0.5, 0.5)


# ---------- 崩溃遗留 ----------

def test_crash_keeps_committed_data_and_marks_interrupted(db):
    c = session_store.connect(db)
    sid = session_store.create_session(c, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(30):  # 0.0 ~ 2.9 秒，之后崩溃：不调用 close / pause
        rec.add_frame(frame(spatial="away" if i >= 15 else "normal"), T0 + i * 0.1)
    del rec  # 模拟进程崩溃，未提交的第 2 秒丢失
    done = session_store.create_session(c, T0)
    record_one_second(db, done)
    session_store.finish_session(c, done, T0 + 5)
    empty_leftover = session_store.create_session(c, T0)  # 崩溃时还在校准期：没有任何数据
    c.close()

    c = session_store.connect(db)  # 下次启动
    assert session_store.mark_interrupted_sessions(c) == 1
    row = c.execute("SELECT status, ended_at FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert row["status"] == "interrupted" and row["ended_at"] == pytest.approx(T0 + 1.0)
    assert [s["t"] for s in samples(c, sid)] == [0, 1]
    (ev,) = events(c, sid, "spatial")
    assert ev["t_end"] == 2.0  # 在最后一条样本末尾关闭
    assert c.execute("SELECT status FROM sessions WHERE id = ?", (done,)).fetchone()["status"] == "completed"
    assert c.execute("SELECT 1 FROM sessions WHERE id = ?", (empty_leftover,)).fetchone() is None  # 空的遗留记录被删除
    assert session_store.mark_interrupted_sessions(c) == 0
    c.close()


def test_deleting_session_cascades(conn, db):
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    rec.add_frame(frame(spatial="away", alert=("absence", "a", "b")), T0)
    rec.pause(T0 + 0.5)
    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    for table in ("samples", "events", "pauses"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0



def test_finish_deletes_empty_session(conn, db):
    sid = session_store.create_session(conn, T0)
    session_store.set_baseline(conn, sid, 0.0, 0.0)
    rec = SessionRecorder(sid, T0, db)
    rec.pause(T0 + 1)  # 只有暂停点，没有任何每秒数据
    assert session_store.finish_session(conn, sid, T0 + 2) is False
    for table in ("sessions", "pauses"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
