"""
课程记录的 CSV 导出（#11），供实证实验分析。
一次导出新建一个文件夹，按表分为 sessions / samples / events / pauses / problems 五个 CSV（长格式，每行带 session_id）。
- 只含数值与标签，不含任何图像
- UTF-8 带 BOM（Excel 直接打开日语不乱码；pandas 用 encoding="utf-8-sig"）
- 墙上时间同时给 Unix 秒与带时区的 ISO 8601，便于与外部问卷对齐
- 记录中（status = recording）的课不导出：数据还不完整
列的说明见 README.md「CSV 导出」。
"""
import bisect
import csv
import os
import sqlite3
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from app import problem_segments

ENCODING = "utf-8-sig"

SESSION_COLUMNS = ["session_id", "course_name", "status", "started_at", "started_at_iso", "ended_at", "ended_at_iso",
                   "listening_seconds", "avg_score", "baseline_pitch", "baseline_yaw"]
SAMPLE_COLUMNS = ["session_id", "course_name", "t", "wall_time", "wall_time_iso", "score", "fatigue_index",
                  "cognitive_state", "spatial_state", "face_ratio", "delta_pitch", "delta_yaw", "frame_count"]
EVENT_COLUMNS = ["session_id", "course_name", "kind", "type", "t_start", "t_end", "wall_start", "wall_start_iso"]
PAUSE_COLUMNS = ["session_id", "course_name", "t", "paused_at", "paused_at_iso", "resumed_at", "resumed_at_iso"]
PROBLEM_COLUMNS = ["session_id", "course_name", "category", "start", "end", "duration",
                   "wall_start", "wall_start_iso", "wall_end", "wall_end_iso"]


def iso(ts: Optional[float]) -> Optional[str]:
    """Unix 秒 → 本机时区的 ISO 8601（毫秒，带时差）"""
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec="milliseconds")


def wall_at(ts: Sequence[int], walls: Sequence[float], t: float) -> Optional[float]:
    """
    听课时间 t → 墙上时间：取 t 所在（或之前最近）的一秒的 wall_time（该秒第一帧的时刻）再加上秒内偏移。
    误差不超过一帧间隔。没有样本时为 None。
    """
    i = bisect.bisect_right(ts, t) - 1
    if i < 0:
        return walls[0] - (ts[0] - t) if ts else None
    return walls[i] + (t - ts[i])


class ExportError(Exception):
    pass


def exportable_ids(conn: sqlite3.Connection) -> List[int]:
    """可导出的课程记录（非记录中），按开始时间升序"""
    return [r[0] for r in conn.execute(
        "SELECT id FROM sessions WHERE status != 'recording' ORDER BY started_at, id")]


def export(conn: sqlite3.Connection, parent_dir: Path, session_ids: Optional[Iterable[int]] = None,
           now: Optional[float] = None) -> Dict[str, Any]:
    """
    在 parent_dir 下新建 export 文件夹并写入 CSV。session_ids 为 None 时导出全部可导出的课。
    先写到临时文件夹，全部写完后再改名，失败时不留下写了一半的导出。
    返回 {"dir": 导出文件夹, "sessions": 课数, "rows": {表名: 行数}}。
    """
    parent_dir = Path(parent_dir)
    if session_ids is None:
        ids = exportable_ids(conn)
    else:
        ids = list(dict.fromkeys(session_ids))
        found = {r["id"]: r["status"] for r in conn.execute(
            f"SELECT id, status FROM sessions WHERE id IN ({','.join('?' * len(ids))})", ids)} if ids else {}
        missing = [i for i in ids if i not in found]
        if missing:
            raise ExportError(f"课程记录不存在：{missing}")
        recording = [i for i in ids if found[i] == "recording"]
        if recording:
            raise ExportError(f"课程记录正在记录中，不能导出：{recording}")
    if not ids:
        raise ExportError("没有可导出的课程记录")

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now if now is not None else time.time()))
    name = f"self_analysis_export_{stamp}" + (f"_session{ids[0]}" if session_ids is not None and len(ids) == 1 else "")
    target = _unique(parent_dir / name)

    tables: Dict[str, List[Dict[str, Any]]] = {"sessions": [], "samples": [], "events": [], "pauses": [], "problems": []}
    for sid in ids:
        _collect(conn, sid, tables)

    tmp = Path(tempfile.mkdtemp(prefix=".export_tmp_", dir=parent_dir))
    try:
        for table, columns in (("sessions", SESSION_COLUMNS), ("samples", SAMPLE_COLUMNS), ("events", EVENT_COLUMNS),
                               ("pauses", PAUSE_COLUMNS), ("problems", PROBLEM_COLUMNS)):
            _write(tmp / f"{table}.csv", columns, tables[table])
        os.rename(tmp, target)
    except BaseException:
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
        raise
    return {"dir": str(target), "sessions": len(ids), "rows": {k: len(v) for k, v in tables.items()}}


def _unique(path: Path) -> Path:
    candidate, n = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{n}")
        n += 1
    return candidate


def _collect(conn: sqlite3.Connection, sid: int, tables: Dict[str, List[Dict[str, Any]]]) -> None:
    s = dict(conn.execute(
        "SELECT s.id, s.course_name, s.status, s.started_at, s.ended_at, s.baseline_pitch, s.baseline_yaw,"
        " COUNT(x.t) AS n, AVG(x.score) AS avg_score"
        " FROM sessions s LEFT JOIN samples x ON x.session_id = s.id WHERE s.id = ? GROUP BY s.id", (sid,)).fetchone())
    name = s["course_name"]
    tables["sessions"].append({
        "session_id": sid, "course_name": name, "status": s["status"],
        "started_at": s["started_at"], "started_at_iso": iso(s["started_at"]),
        "ended_at": s["ended_at"], "ended_at_iso": iso(s["ended_at"]),
        "listening_seconds": s["n"], "avg_score": s["avg_score"],
        "baseline_pitch": s["baseline_pitch"], "baseline_yaw": s["baseline_yaw"],
    })

    def fetch(sql: str) -> List[Dict[str, Any]]:
        return [dict(r) for r in conn.execute(sql, (sid,))]

    samples = fetch("SELECT t, wall_time, score, fatigue_index, cognitive_state, spatial_state, face_ratio,"
                    " delta_pitch, delta_yaw, frame_count FROM samples WHERE session_id = ? ORDER BY t")
    events = fetch("SELECT kind, type, t_start, t_end, wall_start FROM events WHERE session_id = ? ORDER BY t_start, id")
    pauses = fetch("SELECT t, paused_at, resumed_at FROM pauses WHERE session_id = ? ORDER BY t, id")
    head = {"session_id": sid, "course_name": name}
    tables["samples"].extend({**head, **x, "wall_time_iso": iso(x["wall_time"])} for x in samples)
    tables["events"].extend({**head, **e, "wall_start_iso": iso(e["wall_start"])} for e in events)
    tables["pauses"].extend({**head, **p, "paused_at_iso": iso(p["paused_at"]), "resumed_at_iso": iso(p["resumed_at"])}
                            for p in pauses)

    ts = [x["t"] for x in samples]
    walls = [x["wall_time"] for x in samples]
    for p in problem_segments.detect(samples, events, pauses):
        ws, we = wall_at(ts, walls, p["start"]), wall_at(ts, walls, p["end"])
        tables["problems"].append({
            **head, "category": p["category"], "start": p["start"], "end": p["end"], "duration": p["end"] - p["start"],
            "wall_start": ws, "wall_start_iso": iso(ws), "wall_end": we, "wall_end_iso": iso(we),
        })


def _write(path: Path, columns: List[str], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row[c]) for c in columns})
