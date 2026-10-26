"""把每帧的 AnalysisResult 按听课时间轴每秒汇总写入数据库（规则见 docs/adr/0002-session-storage-schema.md）"""
import math
from collections import Counter
from typing import List, Optional

from app import session_store
from app.attention_rules import AnalysisResult
from app.session_store import PathLike


class SessionRecorder:
    """
    一次采集（Start 后或继续后，到暂停或结束为止）对应一个实例。
    必须在 CameraThread.run() 内创建和使用：连接不跨线程。
    """

    def __init__(self, session_id: int, start_wall: float, db_path: PathLike = session_store.DEFAULT_DB_PATH):
        self.session_id = session_id
        self.conn = session_store.connect(db_path)
        self.start_wall = start_wall
        self.t_base = self._resume_point(start_wall)

        self._bucket_t: Optional[int] = None
        self._bucket: List[AnalysisResult] = []
        self._bucket_wall: float = start_wall
        self._open_event_id: Optional[int] = None
        self._open_event_type: Optional[str] = None
        self._has_frames = False

    def _resume_point(self, start_wall: float) -> float:
        """若上次以暂停结束，则从暂停点之后的下一整秒继续，并记下继续时刻"""
        pause = self.conn.execute(
            "SELECT id, t FROM pauses WHERE session_id = ? AND resumed_at IS NULL ORDER BY id DESC LIMIT 1",
            (self.session_id,),
        ).fetchone()
        if pause is None:
            return 0.0
        with self.conn:
            self.conn.execute("UPDATE pauses SET resumed_at = ? WHERE id = ?", (start_wall, pause["id"]))
        # 暂停前不足一秒的部分已写成一条样本，向上取整避免与它的秒数重复
        return float(math.ceil(pause["t"]))

    def active_time(self, wall: float) -> float:
        """墙上时间 → 听课时间轴（秒）"""
        return self.t_base + max(0.0, wall - self.start_wall)

    def add_frame(self, result: AnalysisResult, wall: float) -> None:
        t = self.active_time(wall)
        self._has_frames = True
        second = int(t)
        if self._bucket_t is not None and second != self._bucket_t:
            self._flush_bucket()
            self.conn.commit()
        if self._bucket_t is None:
            self._bucket_t = second
            self._bucket_wall = wall
        self._bucket.append(result)

        self._track_spatial_event(result.spatial_state, t, wall)
        if result.alert is not None:
            self.conn.execute(
                "INSERT INTO events (session_id, kind, type, t_start, t_end, wall_start) VALUES (?, 'alert', ?, ?, ?, ?)",
                (self.session_id, result.alert[0], t, t, wall),
            )

    def pause(self, wall: float) -> None:
        """暂停：写入剩余数据、关闭进行中的事件、记录暂停点，然后关闭连接"""
        t = self._finish_segment(wall)
        self.conn.execute(
            "INSERT INTO pauses (session_id, t, paused_at) VALUES (?, ?, ?)", (self.session_id, t, wall)
        )
        self.conn.commit()
        self.conn.close()

    def close(self, wall: float) -> None:
        """结束采集（不改变课程记录状态；正常结束请再调用 session_store.finish_session）"""
        self._finish_segment(wall)
        self.conn.commit()
        self.conn.close()

    def _finish_segment(self, wall: float) -> float:
        # 本段一帧都没有（如摄像头打不开）时听课时间轴不前进
        t = self.active_time(wall) if self._has_frames else self.t_base
        self._flush_bucket()
        self._close_spatial_event(t)
        return t

    def _track_spatial_event(self, state: str, t: float, wall: float) -> None:
        if state == self._open_event_type:
            return
        self._close_spatial_event(t)
        if state != "normal":
            cur = self.conn.execute(
                "INSERT INTO events (session_id, kind, type, t_start, wall_start) VALUES (?, 'spatial', ?, ?, ?)",
                (self.session_id, state, t, wall),
            )
            self._open_event_id = cur.lastrowid
            self._open_event_type = state

    def _close_spatial_event(self, t: float) -> None:
        if self._open_event_id is not None:
            self.conn.execute("UPDATE events SET t_end = ? WHERE id = ?", (t, self._open_event_id))
        self._open_event_id = None
        self._open_event_type = None

    def _flush_bucket(self) -> None:
        if self._bucket_t is None or not self._bucket:
            return
        frames = self._bucket
        n = len(frames)
        states = [f.cognitive_state for f in frames if f.cognitive_state is not None]
        pitches = [f.delta_pitch for f in frames if f.delta_pitch is not None]
        yaws = [f.delta_yaw for f in frames if f.delta_yaw is not None]
        self.conn.execute(
            "INSERT INTO samples (session_id, t, wall_time, score, fatigue_index, cognitive_state, spatial_state,"
            " face_ratio, delta_pitch, delta_yaw, frame_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.session_id,
                self._bucket_t,
                self._bucket_wall,
                sum(f.score for f in frames) / n,
                sum(f.fatigue_index for f in frames) / n,
                Counter(states).most_common(1)[0][0] if states else None,  # 并列时取该秒内先出现的
                frames[-1].spatial_state,
                sum(1 for f in frames if f.has_face) / n,
                sum(pitches) / len(pitches) if pitches else None,
                sum(yaws) / len(yaws) if yaws else None,
                n,
            ),
        )
        self._bucket_t = None
        self._bucket = []
