"""把课程记录读成可直接转 JSON 的字典，供报告页使用（表结构见 docs/adr/0002-session-storage-schema.md）"""
import sqlite3
from typing import Any, Dict, List, Optional

from app import feedback, problem_segments

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
    """
    一条课程记录的概要、每秒样本、事件、暂停、问题时段（app/problem_segments.py）
    与文字反馈（app/feedback.py；没有数据时为 None）；记录不存在时返回 None
    """
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

    samples = fetch(
        "SELECT t, wall_time, score, fatigue_index, cognitive_state, spatial_state,"
        " face_ratio, delta_pitch, delta_yaw, frame_count"
        " FROM samples WHERE session_id = ? ORDER BY t"
    )
    events = fetch("SELECT kind, type, t_start, t_end, wall_start FROM events WHERE session_id = ? ORDER BY t_start, id")
    pauses = fetch("SELECT t, paused_at, resumed_at FROM pauses WHERE session_id = ? ORDER BY t, id")
    problems = problem_segments.detect(samples, events, pauses)
    return {
        "session": dict(session),
        "samples": samples,
        "events": events,
        "pauses": pauses,
        "problems": problems,
        "feedback": feedback.generate(conn, dict(session), events, problems) if samples else None,
    }


def summarize_sessions(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    多节课汇总（#8）：sessions 为有听课数据的课程记录（按开始时间升序），附各类问题时段的次数与秒数；
    courses 为按课程名分组的汇总（最近上课的在前）。
    """
    rows = conn.execute(
        f"SELECT {SESSION_COLUMNS}, {SUMMARY_COLUMNS}"
        " FROM sessions s JOIN samples x ON x.session_id = s.id"
        " GROUP BY s.id ORDER BY s.started_at, s.id"
    ).fetchall()
    sessions = []
    for r in rows:
        problems = _problems(conn, r["id"])
        stats = {c: {"count": 0, "seconds": 0.0} for c in problem_segments.CATEGORIES}
        for p in problems:
            stats[p["category"]]["count"] += 1
            stats[p["category"]]["seconds"] += p["end"] - p["start"]
        sessions.append({**dict(r), "problems": stats})
    return {"sessions": sessions, "courses": course_summary(sessions)}


def course_summary(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按课程名分组（未填写的为一组 None）；平均参与度按听课秒数加权，问题时段为合计秒数"""
    groups: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for s in sessions:
        groups.setdefault(s["course_name"], []).append(s)
    courses = []
    for name, items in groups.items():
        seconds = sum(s["seconds"] for s in items)
        courses.append({
            "course_name": name,
            "sessions": len(items),
            "interrupted": sum(1 for s in items if s["status"] == "interrupted"),
            "seconds": seconds,
            "avg_score": sum(s["avg_score"] * s["seconds"] for s in items) / seconds,
            "problems": {
                c: {"count": sum(s["problems"][c]["count"] for s in items),
                    "seconds": sum(s["problems"][c]["seconds"] for s in items)}
                for c in problem_segments.CATEGORIES
            },
            "last_started_at": max(s["started_at"] for s in items),
        })
    courses.sort(key=lambda c: c["last_started_at"], reverse=True)
    return courses


def _problems(conn: sqlite3.Connection, session_id: int) -> List[Dict[str, Any]]:
    """只读判定需要的列（汇总时每节课都要判定一次）"""
    def fetch(sql: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, (session_id,))]

    return problem_segments.detect(
        fetch("SELECT t, cognitive_state, fatigue_index FROM samples WHERE session_id = ? ORDER BY t"),
        fetch("SELECT kind, type, t_start, t_end FROM events WHERE session_id = ? AND kind = 'spatial'"),
        fetch("SELECT t FROM pauses WHERE session_id = ?"),
    )

