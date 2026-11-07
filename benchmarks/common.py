"""计算量评估的公共部分：路径、机器信息、进程资源采样、分步计时与统计（需求见 benchmarks/proposal.md）"""
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
DATA_DIR = BENCH_DIR / "data"          # 测试视频与模型输出缓存（含人脸信息，已加入 .gitignore）
RESULTS_DIR = BENCH_DIR / "results"    # 不含画面的测量数据（提交到 git）

VIDEO_PATH = DATA_DIR / "session.mp4"
VIDEO_META = DATA_DIR / "session.json"
CACHE_DIR = DATA_DIR / "cache"

IDEAL_FRAME_MS = 1000 / 30   # 理想线：每帧处理 ≤ 33 ms（跟得上 30 fps）
FRAME_GAP_LIMIT_MS = 250     # 帧间隔底线：attention_rules.py 的 0.25 秒连续性规则


def setup_path() -> None:
    """脚本从任何位置运行都能导入 app.*；ExpressionClassifier 以相对路径加载 weights/best.pt，因此切到仓库根目录"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


# ---------------------------------------------------------------- 机器信息

def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if sys.platform == "win32":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _version(dist: str) -> Optional[str]:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def machine_info() -> Dict:
    """记录测量环境（只读取版本号，不导入 torch 等大库，避免影响「都排除」口径的内存）"""
    freq = psutil.cpu_freq()
    return {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_mhz": round(freq.max) if freq and freq.max else None,
        "memory_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
        "python": platform.python_version(),
        "packages": {name: _version(name) for name in (
            "PyQt5", "PyQt5-Qt5", "PyQtWebEngine", "PyQt-Fluent-Widgets", "numpy", "opencv-python",
            "mediapipe", "mediapipe-silicon", "torch", "ultralytics", "psutil")},
    }


def default_machine_tag() -> str:
    """结果子目录名，如 Darwin-Apple_M4；可用 --machine 覆盖"""
    cpu = re.sub(r"[^A-Za-z0-9]+", "_", _cpu_name()).strip("_")[:40]
    return f"{platform.system()}-{cpu}"


# ---------------------------------------------------------------- 资源采样

class ResourceSampler:
    """
    后台线程定时采样本进程及其子进程（报告页的 QtWebEngineProcess）的 CPU 时间与常驻内存（RSS）。
    CPU 占用率以「单核 = 100%」表示（与活动监视器 / psutil 一致）。
    """

    def __init__(self, interval: float = 0.25, children: bool = False):
        self.interval = interval
        self.proc = psutil.Process()
        self.children = children
        self._kids: List[psutil.Process] = []
        self._kids_at = 0.0
        self.samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _family(self, refresh: bool = False) -> List[psutil.Process]:
        # 枚举子进程很慢（macOS 上约 8 ms，要遍历全部进程），每 0.5 秒才刷新一次，避免采样本身占用 CPU
        if self.children and (refresh or time.perf_counter() - self._kids_at > 0.5):
            try:
                self._kids = self.proc.children(recursive=True)
            except psutil.Error:
                pass
            self._kids_at = time.perf_counter()
        return [self.proc] + self._kids

    def snapshot(self, refresh: bool = False) -> Dict:
        """refresh=True：先重新枚举子进程（测量区间的起止点用，确保新启动的子进程被计入）"""
        cpu, rss = 0.0, 0
        for p in self._family(refresh):
            try:
                t = p.cpu_times()
                cpu += t.user + t.system
                rss += p.memory_info().rss
            except psutil.Error:
                pass  # 子进程恰好退出
        return {"t": time.perf_counter(), "cpu_s": cpu, "rss_mb": rss / 2 ** 20}

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append(self.snapshot())
            self._stop.wait(self.interval)

    def start(self) -> "ResourceSampler":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self.samples.append(self.snapshot())

    def window(self, t0: float, t1: float) -> Dict:
        """[t0, t1] 区间内的 CPU 时间、平均占用率与内存峰值"""
        inside = [s for s in self.samples if t0 <= s["t"] <= t1]
        if len(inside) < 2:
            return {"cpu_s": None, "cpu_pct": None, "rss_peak_mb": None, "rss_mean_mb": None}
        cpu = inside[-1]["cpu_s"] - inside[0]["cpu_s"]
        wall = inside[-1]["t"] - inside[0]["t"]
        rss = [s["rss_mb"] for s in inside]
        return {"cpu_s": cpu, "wall_s": wall, "cpu_pct": 100 * cpu / wall if wall > 0 else None,
                "rss_peak_mb": max(rss), "rss_mean_mb": sum(rss) / len(rss)}


# ---------------------------------------------------------------- 分步计时

class StepTimer:
    """
    给函数套上计时：每次调用累计墙上时间（perf_counter）与本线程 CPU 时间（thread_time）到 sink()[step]。
    Windows 上 thread_time 的分辨率约 15.6 ms，单步 CPU 时间只在长时间累计后才有意义。
    """

    def __init__(self, sink: Callable[[], Optional[Dict]]):
        self.sink = sink

    def wrap(self, func: Callable, step: str) -> Callable:
        def timed(*args, **kwargs):
            w0, c0 = time.perf_counter(), time.thread_time()
            try:
                return func(*args, **kwargs)
            finally:
                rec = self.sink()
                if rec is not None:
                    w, c = rec.setdefault(step, [0.0, 0.0])
                    rec[step] = [w + time.perf_counter() - w0, c + time.thread_time() - c0]
        timed.__wrapped__ = func
        return timed


# ---------------------------------------------------------------- 统计

def percentile(values: List[float], q: float) -> Optional[float]:
    """线性插值分位数（q 取 0–100）"""
    vals = sorted(v for v in values if v is not None and not math.isnan(v))
    if not vals:
        return None
    k = (len(vals) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def describe(values: List[float]) -> Dict:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0, "median": None, "p95": None, "max": None, "mean": None}
    return {"n": len(vals), "median": percentile(vals, 50), "p95": percentile(vals, 95),
            "max": max(vals), "mean": sum(vals) / len(vals)}


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
