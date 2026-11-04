"""
单节课的日语文字反馈（#9）：用规则模板生成，不接入大模型。
生成部分可替换：FeedbackContext 是全部输入（纯数据），任何实现 generate(ctx) -> dict 的对象都能换上
（例如以后的大模型生成器），输出结构见 RuleFeedbackGenerator.generate。
阈值集中在 FeedbackConfig，实验后调整。
"""
import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from app.problem_segments import DISTRACTION, DOUBT, FATIGUE

Row = Dict[str, Any]
AWAY_STATES = frozenset({"away", "absent"})


@dataclass(frozen=True)
class FeedbackConfig:
    # 総評：平均参与度 ≥ good 为「よく」，≥ fair 为「おおむね」，以下为「続きにくい」
    good_score: float = 75.0
    fair_score: float = 60.0
    # 问题时段（三类取并集）占听课时间 ≥ 此比例时，前两档各降一档
    heavy_problem_share: float = 0.30
    # 「気になった時間帯」最多列出几条（按时长取最长的，再按时间排序）
    max_segments: int = 3
    # 与同课程名以往平均的差在 ±same_margin 以内视为「同程度」
    same_margin: float = 5.0
    # 疲劳时段的秒数中，后半节占比 ≥ 此值时提示「後半に疲れが出やすい」
    late_fatigue_share: float = 0.6


DEFAULT_CONFIG = FeedbackConfig()


@dataclass
class FeedbackContext:
    """生成反馈所需的全部输入"""
    course_name: Optional[str]
    status: str
    seconds: int                       # 听课秒数
    avg_score: float
    problems: List[Row]                # problem_segments.detect 的结果
    spatial_events: List[Row] = field(default_factory=list)  # 用于区分「注意がそれた」与「席を離れた」
    history_scores: List[float] = field(default_factory=list)  # 同课程名、更早的课的平均参与度


class FeedbackGenerator(Protocol):
    name: str

    def generate(self, ctx: FeedbackContext) -> Dict[str, Any]:
        ...


# ---------- 文字格式 ----------

def minute_range(start: float, end: float) -> str:
    """时段的位置以分钟表示，两端四舍五入（与时长的约略一致）；落在同一分钟时至少跨 1 分钟"""
    a = math.floor(start / 60 + 0.5)
    b = max(a + 1, math.floor(end / 60 + 0.5))
    return f"{_clock(a)}〜{_clock(b)}"


def _clock(minutes: int) -> str:
    return f"{minutes // 60}時間{minutes % 60}分" if minutes >= 60 else f"{minutes}分"


def duration_text(seconds: float) -> str:
    """不足 1 分钟以秒表示，否则约略到分钟"""
    if seconds < 60:
        return f"{max(1, round(seconds))}秒"
    minutes = round(seconds / 60)
    return _clock(minutes)


def _union_seconds(problems: List[Row]) -> float:
    total, cur_start, cur_end = 0.0, None, None
    for p in sorted(problems, key=lambda p: p["start"]):
        if cur_end is None or p["start"] > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = p["start"], p["end"]
        else:
            cur_end = max(cur_end, p["end"])
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _mostly_away(segment: Row, events: List[Row], end: float) -> bool:
    """分心 / 离席时段内，离席（away / absent）的时间是否占多数"""
    away = other = 0.0
    for e in events:
        t_end = e["t_end"] if e["t_end"] is not None else end
        o = _overlap(segment["start"], segment["end"], e["t_start"], t_end)
        if e["type"] in AWAY_STATES:
            away += o
        else:
            other += o
    return away > other


# ---------- 规则生成器 ----------

SEGMENT_TEXT = {
    DOUBT: "理解が追いついていなかった可能性があります",
    FATIGUE: "疲れが見られました",
    "away": "席を離れていた可能性があります",
    "distracted": "注意がそれていた可能性があります",
}

HINT_TEXT = {
    DOUBT: "理解が追いついていなかった可能性のある時間帯の内容を復習し、わからない点は質問してみましょう。",
    "fatigue_late": "授業の後半に疲れが出やすいようです。短い休憩や姿勢の見直しをしてみましょう。",
    "fatigue": "疲れが見られた時間帯がありました。睡眠や休憩のとり方を見直してみましょう。",
    "away": "席を離れるときは「一時停止」を使うと、記録がより正確になります。",
    "distracted": "注意がそれやすい時間帯がありました。通知を切るなど、学習環境を整えてみましょう。",
    "low_no_problem": "難しかった部分をメモしておき、復習してみましょう。",
    "keep": "この調子で続けましょう。",
}


class RuleFeedbackGenerator:
    """
    输出：
      summary     総評（一句）
      segments    気になった時間帯 [{category, start, end, text}]（最多 max_segments 条，按时间排序）
      more        未列出的时段数
      hints       次回へのヒント [str]
      comparison  前回との比較 {course_name, text}；没有同课程名的以往记录时为 None
      generator   生成器名称
    """
    name = "rules"

    def __init__(self, config: FeedbackConfig = DEFAULT_CONFIG):
        self.config = config

    def generate(self, ctx: FeedbackContext) -> Dict[str, Any]:
        segments, kinds = self._segments(ctx)
        longest = sorted(segments, key=lambda s: s["end"] - s["start"], reverse=True)[: self.config.max_segments]
        return {
            "summary": self._summary(ctx),
            "segments": sorted(longest, key=lambda s: s["start"]),
            "more": max(0, len(segments) - self.config.max_segments),
            "hints": self._hints(ctx, kinds),
            "comparison": self._comparison(ctx),
            "generator": self.name,
        }

    def _summary(self, ctx: FeedbackContext) -> str:
        c = self.config
        level = 2 if ctx.avg_score >= c.good_score else (1 if ctx.avg_score >= c.fair_score else 0)
        share = _union_seconds(ctx.problems) / ctx.seconds if ctx.seconds else 0.0
        if share >= c.heavy_problem_share and level > 0:
            level -= 1
        score = round(ctx.avg_score)
        text = {
            2: f"全体としてよく集中できていました（平均エンゲージメント {score}）。",
            1: f"おおむね集中できていました（平均エンゲージメント {score}）。",
            0: f"集中が続きにくい授業だったようです（平均エンゲージメント {score}）。",
        }[level]
        if not ctx.problems:
            text += "気になる時間帯は見つかりませんでした。"
        elif level == 0:
            text += "気になった時間帯を振り返ってみましょう。"
        if ctx.status == "interrupted":
            text += "この記録は途中で終了しています。"
        elif ctx.status == "recording":
            text += "記録中のため、途中までの結果です。"
        return text

    def _segments(self, ctx: FeedbackContext):
        """每个问题时段的文字；kinds 为出现过的细分类别 → 合计秒数（用于提示）"""
        out, kinds = [], {}
        for p in ctx.problems:
            kind = p["category"]
            if kind == DISTRACTION:
                kind = "away" if _mostly_away(p, ctx.spatial_events, ctx.seconds) else "distracted"
            length = p["end"] - p["start"]
            kinds[kind] = kinds.get(kind, 0.0) + length
            out.append({
                "category": p["category"], "start": p["start"], "end": p["end"],
                "text": f"{minute_range(p['start'], p['end'])}：{SEGMENT_TEXT[kind]}（{duration_text(length)}）",
            })
        return out, kinds

    def _hints(self, ctx: FeedbackContext, kinds: Dict[str, float]) -> List[str]:
        if not kinds:
            low = ctx.avg_score < self.config.fair_score
            return [HINT_TEXT["low_no_problem" if low else "keep"]]
        hints = []
        for kind, _ in sorted(kinds.items(), key=lambda kv: kv[1], reverse=True):
            if kind == FATIGUE:
                late = sum(_overlap(p["start"], p["end"], ctx.seconds / 2, ctx.seconds)
                           for p in ctx.problems if p["category"] == FATIGUE)
                kind = "fatigue_late" if late / kinds[FATIGUE] >= self.config.late_fatigue_share else "fatigue"
            hints.append(HINT_TEXT[kind])
        return hints

    def _comparison(self, ctx: FeedbackContext) -> Optional[Dict[str, str]]:
        if not ctx.course_name or not ctx.history_scores:
            return None
        before = sum(ctx.history_scores) / len(ctx.history_scores)
        diff = ctx.avg_score - before
        n = len(ctx.history_scores)
        base = "前回" if n == 1 else f"これまでの {n} 回の平均"
        if abs(diff) <= self.config.same_margin:
            text = f"平均エンゲージメントは{base}（{round(before)}）と同程度でした。"
        else:
            word = "高く" if diff > 0 else "低く"
            text = f"平均エンゲージメントは{base}（{round(before)}）より {round(abs(diff))} {word}なりました。"
        return {"course_name": ctx.course_name, "text": text}


GENERATOR: FeedbackGenerator = RuleFeedbackGenerator()


# ---------- 从数据库组装输入 ----------

def build_context(conn: sqlite3.Connection, session: Row, events: List[Row], problems: List[Row]) -> FeedbackContext:
    """session 为 report_data.load_session 的 session 部分；历史为同课程名、开始更早且有数据的课"""
    history: List[float] = []
    if session["course_name"]:
        history = [r[0] for r in conn.execute(
            "SELECT AVG(x.score) FROM sessions s JOIN samples x ON x.session_id = s.id"
            " WHERE s.course_name = ? AND (s.started_at < ? OR (s.started_at = ? AND s.id < ?))"
            " GROUP BY s.id ORDER BY s.started_at",
            (session["course_name"], session["started_at"], session["started_at"], session["id"]),
        )]
    return FeedbackContext(
        course_name=session["course_name"], status=session["status"], seconds=session["seconds"],
        avg_score=session["avg_score"] if session["avg_score"] is not None else 0.0, problems=problems,
        spatial_events=[e for e in events if e["kind"] == "spatial"], history_scores=history,
    )


def generate(conn: sqlite3.Connection, session: Row, events: List[Row], problems: List[Row],
             generator: Optional[FeedbackGenerator] = None) -> Dict[str, Any]:
    return (generator or GENERATOR).generate(build_context(conn, session, events, problems))

