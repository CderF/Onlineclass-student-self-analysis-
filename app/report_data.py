"""把课程记录读成可直接转 JSON 的字典，供报告页使用（表结构见 docs/adr/0002-session-storage-schema.md）"""
import sqlite3
from typing import Any, Dict, List, Optional

SESSION_COLUMNS = "s.id, s.course_name, s.started_at, s.ended_at, s.status"
SUMMARY_COLUMNS = "COUNT(x.t) AS seconds, AVG(x.score) AS avg_score"


def list_sessions(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """全部课程记录，最近开始的在前；seconds 为听课时间轴上记录的秒数"""
    rows = conn.execute(
        f"SELECT {SESSION_COLUMNS}, {SUMMARY_COLUMNS}"
        " FROM sessions s LEFT JOIN samples x ON x.session_id = s.id"
        " GROUP BY s.id ORDER BY s.started_at DESC, s.id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def load_session(conn: sqlite3.Connection, session_id: int) -> Optional[Dict[str, Any]]:
    """一条课程记录的概要、每秒样本、事件与暂停；记录不存在时返回 None"""
    session = conn.execute(
        f"SELECT {SESSION_COLUMNS}, s.baseline_pitch, s.baseline_yaw, {SUMMARY_COLUMNS}"
        " FROM sessions s LEFT JOIN samples x ON x.session_id = s.id"
        " WHERE s.id = ? GROUP BY s.id",
        (session_id,),
    ).fetchone()
    if session is None:
        return None

    def fetch(sql: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, (session_id,))]

    return {
        "session": dict(session),
        "samples": fetch(
            "SELECT t, wall_time, score, fatigue_index, cognitive_state, spatial_state,"
            " face_ratio, delta_pitch, delta_yaw, frame_count"
            " FROM samples WHERE session_id = ? ORDER BY t"
        ),
        "events": fetch(
            "SELECT kind, type, t_start, t_end, wall_start FROM events WHERE session_id = ? ORDER BY t_start, id"
        ),
        "pauses": fetch("SELECT t, paused_at, resumed_at FROM pauses WHERE session_id = ? ORDER BY t, id"),
    }
