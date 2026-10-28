"""课程记录的 SQLite 存储（表结构见 docs/adr/0002-session-storage-schema.md）"""
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "self_analysis.db"
SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    course_name    TEXT,
    started_at     REAL NOT NULL,
    ended_at       REAL,
    status         TEXT NOT NULL CHECK (status IN ('recording', 'completed', 'interrupted')),
    baseline_pitch REAL,
    baseline_yaw   REAL
);

CREATE TABLE samples (
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t               INTEGER NOT NULL,
    wall_time       REAL NOT NULL,
    score           REAL NOT NULL,
    fatigue_index   REAL NOT NULL,
    cognitive_state TEXT,
    spatial_state   TEXT NOT NULL,
    face_ratio      REAL NOT NULL,
    delta_pitch     REAL,
    delta_yaw       REAL,
    frame_count     INTEGER NOT NULL,
    PRIMARY KEY (session_id, t)
);

CREATE TABLE events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL CHECK (kind IN ('spatial', 'alert')),
    type       TEXT NOT NULL,
    t_start    REAL NOT NULL,
    t_end      REAL,
    wall_start REAL NOT NULL
);
CREATE INDEX idx_events_session ON events(session_id, t_start);

CREATE TABLE pauses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    t          REAL NOT NULL,
    paused_at  REAL NOT NULL,
    resumed_at REAL
);
CREATE INDEX idx_pauses_session ON pauses(session_id, t);
"""

PathLike = Union[str, Path]


def connect(db_path: Optional[PathLike] = None) -> sqlite3.Connection:
    """打开数据库（默认 DEFAULT_DB_PATH，调用时解析，便于测试替换）；首次使用时建表。连接只能在创建它的线程内使用。"""
    db_path = Path(db_path if db_path is not None else DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        with conn:
            conn.executescript(SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif version > SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"数据库表结构版本 {version} 高于程序支持的 {SCHEMA_VERSION}：{db_path}")
    return conn


def create_session(conn: sqlite3.Connection, started_at: float) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO sessions (started_at, status) VALUES (?, 'recording')", (started_at,)
        )
    return cur.lastrowid


def set_baseline(conn: sqlite3.Connection, session_id: int, pitch: float, yaw: float) -> None:
    with conn:
        conn.execute(
            "UPDATE sessions SET baseline_pitch = ?, baseline_yaw = ? WHERE id = ?",
            (float(pitch), float(yaw), session_id),
        )


def set_course_name(conn: sqlite3.Connection, session_id: int, name: Optional[str]) -> None:
    """课程名去除首尾空白；空字符串视为未填写"""
    name = (name or "").strip() or None
    with conn:
        conn.execute("UPDATE sessions SET course_name = ? WHERE id = ?", (name, session_id))


def list_course_names(conn: sqlite3.Connection) -> List[str]:
    """已使用过的课程名，最近使用的在前"""
    rows = conn.execute(
        "SELECT course_name FROM sessions WHERE course_name IS NOT NULL"
        " GROUP BY course_name ORDER BY MAX(started_at) DESC"
    ).fetchall()
    return [r[0] for r in rows]


def finish_session(conn: sqlite3.Connection, session_id: int, ended_at: float) -> bool:
    """
    正常结束（Stop 或关闭窗口）。一秒数据都没有的记录（如校准期内就终了）直接删除。
    返回 True 表示记录被保留，False 表示因为是空记录而被删除。
    """
    with conn:
        if _is_empty(conn, session_id):
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return False
        conn.execute(
            "UPDATE sessions SET status = 'completed', ended_at = ? WHERE id = ? AND status = 'recording'",
            (ended_at, session_id),
        )
    return True


def _is_empty(conn: sqlite3.Connection, session_id: int) -> bool:
    return conn.execute("SELECT 1 FROM samples WHERE session_id = ? LIMIT 1", (session_id,)).fetchone() is None


def mark_interrupted_sessions(conn: sqlite3.Connection) -> int:
    """
    程序启动时调用：仍为 recording 的记录是崩溃遗留，改为 interrupted。
    遗留的未关闭事件在最后一条每秒样本的末尾关闭；ended_at 取最后一条样本的墙上时间。
    没有任何数据的遗留记录直接删除。返回标记为 interrupted 的记录数。
    """
    rows = conn.execute("SELECT id FROM sessions WHERE status = 'recording'").fetchall()
    marked = 0
    with conn:
        for (session_id,) in rows:
            if _is_empty(conn, session_id):
                conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
                continue
            marked += 1
            last = conn.execute(
                "SELECT t, wall_time FROM samples WHERE session_id = ? ORDER BY t DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            last_end: Optional[float] = last["t"] + 1 if last else None
            last_wall: Optional[float] = last["wall_time"] if last else None
            conn.execute(
                "UPDATE events SET t_end = MAX(t_start, COALESCE(?, t_start)) WHERE session_id = ? AND t_end IS NULL",
                (last_end, session_id),
            )
            conn.execute(
                "UPDATE sessions SET status = 'interrupted', ended_at = COALESCE(?, started_at) WHERE id = ?",
                (last_wall, session_id),
            )
    return marked
