"""
三类问题时段的判定（术语见 GLOSSARY.md「问题时段」）：可能没听懂（Doubt 集中）、疲劳、分心 / 离席。
阈值与窗口长度是初始值，集中在 ProblemConfig，实验后调整（docs/self-analysis-requirements.md 待定项）。
所有判定都在暂停处断开：窗口不跨暂停，时段也不跨暂停合并。
"""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

DOUBT = "doubt"
FATIGUE = "fatigue"
DISTRACTION = "distraction"
CATEGORIES = (DOUBT, FATIGUE, DISTRACTION)

Row = Dict[str, Any]


@dataclass(frozen=True)
class ProblemConfig:
    # 可能没听懂：以每秒为中心的窗口内，有认知状态的秒数够多，且其中「疑問」占比达到阈值
    doubt_window: int = 60
    doubt_min_valid: int = 30
    doubt_ratio: float = 0.4
    # 疲劳：以每秒为中心的窗口内疲劳指数的平均值达到阈值（与 attention_rules 的疲劳警报阈值相同）
    fatigue_window: int = 30
    fatigue_threshold: float = 0.38
    # 分心 / 离席：这些空间状态的事件区间（低头、记笔记类不算）
    distraction_states: FrozenSet[str] = frozenset({"distracted", "hidden_abnormal", "away", "absent"})
    # 同类时段间隔不超过 merge_gap 秒则合并；合并后短于 min_duration 的丢弃
    merge_gap: float = 10.0
    min_duration: Dict[str, float] = field(default_factory=lambda: {DOUBT: 30.0, FATIGUE: 30.0, DISTRACTION: 10.0})


DEFAULT_CONFIG = ProblemConfig()


def detect(samples: Sequence[Row], events: Sequence[Row], pauses: Sequence[Row],
           config: ProblemConfig = DEFAULT_CONFIG) -> List[Row]:
    """
    输入一节课按 t 排序的每秒样本、事件、暂停（report_data.load_session 的格式），
    输出 [{"category", "start", "end"}]（听课时间轴上的秒，左闭右开），按开始时间排序。
    """
    # 继续后从暂停前最后一秒的下一秒开始（ADR 0002）
    boundaries = sorted({float(math.floor(p["t"]) + 1) for p in pauses})
    end = samples[-1]["t"] + 1 if samples else 0
    found: List[Row] = []
    for category, intervals in (
        (DOUBT, _marked_intervals(samples, boundaries, lambda seg: _doubt_marks(seg, config))),
        (FATIGUE, _marked_intervals(samples, boundaries, lambda seg: _fatigue_marks(seg, config))),
        (DISTRACTION, _distraction_intervals(events, end, config)),
    ):
        merged = _merge(intervals, boundaries, config.merge_gap)
        found += [{"category": category, "start": a, "end": b} for a, b in merged
                  if b - a >= config.min_duration[category]]
    found.sort(key=lambda p: (p["start"], CATEGORIES.index(p["category"])))
    return found


def _segments(samples: Sequence[Row], boundaries: List[float]) -> List[List[Row]]:
    """按暂停与时间轴空缺切成若干段，每段内 t 连续"""
    segments: List[List[Row]] = []
    for s in samples:
        cur = segments[-1] if segments else None
        if cur is None or s["t"] != cur[-1]["t"] + 1 or any(cur[-1]["t"] < b <= s["t"] for b in boundaries):
            segments.append([s])
        else:
            cur.append(s)
    return segments


def _centered_counts(values: List[float], window: int) -> List[tuple]:
    """每个位置以它为中心、长 window 秒的窗口（截断在段内）的 (总和, 个数)"""
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    half = window // 2
    n = len(values)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i - half + window)
        out.append((prefix[hi] - prefix[lo], hi - lo))
    return out


def _doubt_marks(seg: List[Row], config: ProblemConfig) -> List[bool]:
    valid = _centered_counts([1.0 if s["cognitive_state"] is not None else 0.0 for s in seg], config.doubt_window)
    doubt = _centered_counts([1.0 if s["cognitive_state"] == "Doubt" else 0.0 for s in seg], config.doubt_window)
    return [v >= config.doubt_min_valid and d / v >= config.doubt_ratio for (v, _), (d, _) in zip(valid, doubt)]


def _fatigue_marks(seg: List[Row], config: ProblemConfig) -> List[bool]:
    sums = _centered_counts([s["fatigue_index"] for s in seg], config.fatigue_window)
    return [total / n >= config.fatigue_threshold for total, n in sums]


def _marked_intervals(samples, boundaries, mark) -> List[tuple]:
    """段内连续被标记的秒 → [开始, 结束)"""
    intervals = []
    for seg in _segments(samples, boundaries):
        start: Optional[float] = None
        for s, hit in zip(seg, mark(seg)):
            if hit and start is None:
                start = float(s["t"])
            elif not hit and start is not None:
                intervals.append((start, float(s["t"])))
                start = None
        if start is not None:
            intervals.append((start, float(seg[-1]["t"] + 1)))
    return intervals


def _distraction_intervals(events: Sequence[Row], end: float, config: ProblemConfig) -> List[tuple]:
    """空间状态事件；仍在进行（t_end 为空）的画到听课时间轴末尾"""
    return sorted(
        (float(e["t_start"]), float(e["t_end"] if e["t_end"] is not None else max(end, e["t_start"])))
        for e in events if e["kind"] == "spatial" and e["type"] in config.distraction_states
    )


def _merge(intervals: List[tuple], boundaries: List[float], gap: float) -> List[tuple]:
    """间隔不超过 gap 且中间没有暂停的区间合并"""
    merged: List[list] = []
    for a, b in sorted(intervals):
        if merged and a - merged[-1][1] <= gap and not any(merged[-1][1] <= x <= a for x in boundaries):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]
