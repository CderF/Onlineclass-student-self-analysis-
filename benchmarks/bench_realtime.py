"""
上课时持续发生的开销：每帧处理链路、课程记录（SQLite）、界面线程刷新（benchmarks/proposal.md 第 4 节前三项）。

运行：在仓库根目录执行
    python benchmarks/bench_realtime.py --scope exclude --pace realtime    # 都排除 × 按真实帧率（约 5 分钟 × 5 次）
    python benchmarks/bench_realtime.py --scope exclude --pace max         # 都排除 × 尽可能快
    python benchmarks/bench_realtime.py --scope include --smoke 60         # 都包含：只试跑 60 帧，不保存结果
常用选项：--runs N（默认 5）、--machine 名称（结果子目录，默认按 CPU 型号）、--no-mesh（关闭面部网格）、
--offscreen（不显示窗口）、--video / --cache（换输入，测试用）。

做法（不改 app/ 的任何代码）：
- 运行真实的 MonitorInterface 与 CameraThread：摄像头换成回放 benchmarks/data/session.mp4 的 ReplayCapture，
  信号经过真实的跨线程队列送到界面线程，界面真实刷新。
- 「都排除」：FaceMesh 不构建计算图，face_mesh.process 回放缓存的关键点；表情分类器换成回放缓存的概率，
  不导入 ultralytics / torch。关键点之后的几何计算、画网格、注意力规则、记录、界面都是真实代码。
- 「都包含」：使用真实的 FaceMesh 与 weights/best.pt。
- 给各环节套上计时（墙上时间 + 本线程 CPU 时间）。读帧解码与缓存回放是测量手段本身的开销，单独计时后扣除。
- 每次测量都在新进程中进行（内存从零开始），开头 --warmup 秒（默认 5，含 3 秒校准）不计入。
结果写到 benchmarks/results/<机器>/realtime-<口径>-<节奏>/run<k>.json 与 run<k>_frames.csv.gz、run<k>_resources.csv。
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

from common import (CACHE_DIR, RESULTS_DIR, VIDEO_META, VIDEO_PATH, ResourceSampler, StepTimer,
                    default_machine_tag, describe, machine_info, setup_path, write_json)

MODEL_STEPS = {"exclude": ("replay_mediapipe", "replay_yolo"), "include": ("model_mediapipe", "model_yolo")}
UI_SLOTS = ("set_camera_frame", "update_attention_level", "update_status_text")


class FrameLog:
    """采集线程逐帧的分步计时；read() 返回时开一条记录，下一次 read() 开始时关闭"""

    def __init__(self):
        self.rec = None
        self.frames = []
        self.finished = threading.Event()
        self.t_first = None
        self.cam_cpu = {}  # 采集线程 CPU 时间：first / warm / last

    def begin(self, index, t, decode, dropped):
        self.rec = {"index": index, "begin": t, "decode": decode, "dropped": dropped}
        if self.t_first is None:
            self.t_first = t

    def end(self, t):
        if self.rec is not None:
            self.rec["end"] = t
            self.frames.append(self.rec)
            self.rec = None


class ReplayCache:
    """build_cache.py 生成的缓存。关键点逐帧从文件读取（不整体载入内存，免得计入 RSS）"""

    def __init__(self, cache_dir: Path):
        import numpy as np
        self.meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        self.index = np.load(cache_dir / "index.npy")
        self.probs = np.load(cache_dir / "probs.npy")  # 约 9000 × 7 × 4 字节，可忽略
        self.names = self.meta["class_names"]
        self.file = open(cache_dir / "landmarks.bin", "rb")

    def landmarks(self, i):
        from mediapipe.framework.formats import landmark_pb2
        offset, length = self.index[i]
        if length == 0:
            return None
        self.file.seek(int(offset))
        lm = landmark_pb2.NormalizedLandmarkList()
        lm.ParseFromString(self.file.read(int(length)))
        return [lm]


def run_once(args, out_dir: Path, run_index: int) -> dict:
    if args.offscreen:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    setup_path()
    sampler = ResourceSampler().start()

    import cv2
    video = Path(args.video)
    meta_path = VIDEO_META if video == VIDEO_PATH else None
    probe = cv2.VideoCapture(str(video))
    if not probe.isOpened():
        raise SystemExit(f"❌ 无法打开视频 {video}；请先运行 python benchmarks/record_video.py")
    fps = json.loads(meta_path.read_text(encoding="utf-8"))["fps"] if meta_path and meta_path.exists() \
        else probe.get(cv2.CAP_PROP_FPS)
    n_video = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()

    log = FrameLog()
    timer = StepTimer(lambda: log.rec)
    current = {"index": 0}
    steps = MODEL_STEPS[args.scope]
    cache = None

    # ---- 「都排除」：在导入 app.camera_thread 之前换掉表情分类器模块，避免导入 ultralytics / torch
    if args.scope == "exclude":
        cache = ReplayCache(Path(args.cache))
        if cache.meta["frames"] != n_video:
            raise SystemExit(f"❌ 缓存帧数 {cache.meta['frames']} 与视频帧数 {n_video} 不一致，请重新运行 build_cache.py")

        class ReplayClassifier:
            def __init__(self, weights_path=None):
                self.model = "replay"
                self.classes = dict(enumerate(cache.names))

            def process_face(self, face_img):
                row = cache.probs[current["index"]]
                if row[0] != row[0]:  # NaN：缓存时这一帧没有人脸框
                    return "Unknown", 0.0, None
                k = int(row.argmax())
                return cache.names[k], float(row[k]), row.copy()

        import app
        stub = types.ModuleType("app.yolo_inference")
        stub.ExpressionClassifier = ReplayClassifier
        sys.modules["app.yolo_inference"] = stub
        app.yolo_inference = stub

    import app.camera_thread as ct
    from app import attention_rules, session_recorder
    from app import mediapipe_inference as mi

    class BenchFaceMesh(mi.FaceMeshInference):
        def __init__(self):
            if args.scope == "exclude":
                solution = mi.mp.solutions.face_mesh
                real = solution.FaceMesh
                solution.FaceMesh = lambda **kwargs: None  # 不构建 FaceMesh 计算图（不加载模型）
                try:
                    super().__init__()
                finally:
                    solution.FaceMesh = real
                self.face_mesh = types.SimpleNamespace(process=lambda rgb: types.SimpleNamespace(
                    multi_face_landmarks=cache.landmarks(current["index"])))
            else:
                super().__init__()
            self.face_mesh.process = timer.wrap(self.face_mesh.process, steps[0])
            self.mp_drawing = types.SimpleNamespace(
                draw_landmarks=timer.wrap(self.mp_drawing.draw_landmarks, "draw_mesh"))
            self.estimate_head_pose = timer.wrap(self.estimate_head_pose, "head_pose")
            self._calculate_ear = timer.wrap(self._calculate_ear, "ear_mar")
            self._calculate_mar = timer.wrap(self._calculate_mar, "ear_mar")
            self.process_frame = timer.wrap(self.process_frame, "process_frame")

    base_classifier = ct.ExpressionClassifier

    class BenchClassifier(base_classifier):
        def __init__(self):
            super().__init__()
            self.process_face = timer.wrap(self.process_face, steps[1])

    ct.FaceMeshInference = BenchFaceMesh
    ct.ExpressionClassifier = BenchClassifier
    attention_rules.AttentionAnalyzer.process_frame = timer.wrap(
        attention_rules.AttentionAnalyzer.process_frame, "attention_rules")
    session_recorder.SessionRecorder.add_frame = timer.wrap(
        session_recorder.SessionRecorder.add_frame, "recorder")

    real_capture = cv2.VideoCapture
    warm_at = {"t": None}

    class ReplayCapture:
        """
        代替摄像头。realtime：按视频帧率送帧，处理跟不上时跳到当前时刻对应的帧（与无缓冲的摄像头相同，计为丢帧）；
        max：不等待，连续送帧。解码与跳帧的耗时记为 decode，属于测量手段的开销。
        """

        def __init__(self, index):
            # 单线程解码：FFmpeg 默认在自己的线程里解码，本线程的 thread_time 统计不到，测量开销会扣不干净
            self.cap = real_capture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
            self.i = 0
            self.t0 = None

        def isOpened(self):
            return self.cap.isOpened()

        def release(self):
            self.cap.release()

        def read(self):
            now = time.perf_counter()
            log.end(now)
            if log.finished.is_set():
                time.sleep(0.01)
                return False, None
            if self.t0 is None:
                self.t0 = now
                log.cam_cpu["first"] = time.thread_time()
            if args.pace == "realtime":
                due = self.t0 + self.i / fps
                if now < due:
                    time.sleep(due - now)
            w0, c0 = time.perf_counter(), time.thread_time()
            dropped = 0
            if args.pace == "realtime":
                target = int((time.perf_counter() - self.t0) * fps)
                while self.i < target and self.cap.grab():
                    self.i += 1
                    dropped += 1
            ok, frame = self.cap.read()
            if not ok or (args.smoke and self.i >= args.smoke):
                log.cam_cpu["last"] = time.thread_time()
                log.finished.set()
                return False, None
            current["index"] = self.i
            self.i += 1
            t = time.perf_counter()
            log.begin(current["index"], t, (t - w0, time.thread_time() - c0), dropped)
            # 预热按视频时间（帧号）计：两种送帧节奏都与程序看到的时钟一致
            if warm_at["t"] is None and current["index"] >= args.warmup * fps:
                warm_at["t"] = t
                log.cam_cpu["warm"] = time.thread_time()
            return True, frame

    cv2.VideoCapture = ReplayCapture

    if args.pace == "max":
        # 校准（3 秒）、注意力规则的时间窗口（3 / 8 / 60 秒）与每秒记录都按 time.time() 计时。
        # 连续送帧时把它换成视频时间，窗口里的帧数才与实际使用时相同；否则每帧的规则开销会被放大。
        # time 是全局模块，此替换对整个进程生效（本进程只做这一次测量）。
        base = time.time()
        time.time = lambda: base + current["index"] / fps

    # ---- 界面线程
    from PyQt5.QtCore import QTimer
    from PyQt5.QtWidgets import QApplication
    from app.view.monitor_interface import MonitorInterface

    qapp = QApplication.instance() or QApplication(sys.argv)
    w = MonitorInterface()
    tmp = tempfile.TemporaryDirectory()
    w.db_path = Path(tmp.name) / "bench.db"
    w._ask_course_name = lambda names: None
    if args.no_mesh:
        w.meshSwitch.setChecked(False)

    ui_calls = {name: [] for name in UI_SLOTS}

    def ui_wrap(func, name):
        def slot(*a):
            w0, c0 = time.perf_counter(), time.thread_time()
            func(*a)
            ui_calls[name].append((w0, time.perf_counter() - w0, time.thread_time() - c0))
        return slot

    for name in UI_SLOTS:
        setattr(w, name, ui_wrap(getattr(w, name), name))
    w.resize(1100, 750)
    w.show()

    marks = {}

    def settle_then_start():
        marks["rss_before_start_mb"] = sampler.snapshot()["rss_mb"]
        w._on_start_clicked()

    def poll():
        now = time.perf_counter()
        if warm_at["t"] is not None and "main_warm" not in marks and now >= warm_at["t"]:
            marks["main_warm"] = time.thread_time()
        if log.finished.is_set() and "t_end" not in marks:
            marks["t_end"] = now
            marks["main_end"] = time.thread_time()
            w._on_stop_clicked()
            QTimer.singleShot(500, qapp.quit)

    QTimer.singleShot(1000, settle_then_start)  # 窗口显示稳定后再开始
    poll_timer = QTimer()
    poll_timer.timeout.connect(poll)
    poll_timer.start(50)
    qapp.exec_()
    poll_timer.stop()
    sampler.stop()

    result = analyse(args, log, ui_calls, sampler, marks, warm_at["t"], fps, steps, run_index)
    if out_dir is not None:
        save(out_dir, run_index, result, log, sampler, steps)

    w.close()
    qapp.processEvents()
    from PyQt5 import sip
    sip.delete(w)  # 顶层窗口在退出前销毁，避免退出清理阶段的崩溃（CLAUDE.md #14）
    tmp.cleanup()
    return result


STEP_KEYS = ("decode", "process_frame", "draw_mesh", "head_pose", "ear_mar", "attention_rules", "recorder")


def frame_row(rec, steps):
    """一帧的各项耗时（ms）；派生项见 benchmarks/README.md「指标定义」"""
    def wall(k):
        return rec.get(k, (0.0, 0.0))[0] * 1000

    def cpu(k):
        return rec.get(k, (0.0, 0.0))[1] * 1000

    proc =(rec["end"] - rec["begin"]) * 1000
    mesh, yolo = wall(steps[0]), wall(steps[1])
    row = {"index": rec["index"], "begin": rec["begin"], "dropped": rec["dropped"], "proc_ms": proc,
           "mesh_ms": mesh, "yolo_ms": yolo}
    for k in STEP_KEYS:
        row[f"{k}_ms"] = wall(k)
        row[f"{k}_cpu_ms"] = cpu(k)
    row["mesh_cpu_ms"], row["yolo_cpu_ms"] = cpu(steps[0]), cpu(steps[1])
    # process_frame 内部调用了 face_mesh.process（回放或模型），这里扣掉，只留不含模型的部分
    row["process_frame_ms"] -= mesh
    row["process_frame_cpu_ms"] -= row["mesh_cpu_ms"]
    row["process_frame_other_ms"] = row["process_frame_ms"] - wall("draw_mesh") - wall("head_pose") - wall("ear_mar")
    row["loop_other_ms"] = proc - mesh - row["process_frame_ms"] - yolo - wall("attention_rules") - wall("recorder")
    replay = mesh + yolo if steps[0].startswith("replay") else 0.0
    model = mesh + yolo if steps[0].startswith("model") else 0.0
    row["system_ms"] = proc - replay - model   # 不含模型推理（都排除口径的每帧处理时间）
    row["total_ms"] = proc - replay            # 含模型推理（都包含口径时才有意义）
    return row


def analyse(args, log, ui_calls, sampler, marks, warm_t, fps, steps, run_index):
    t_end = marks.get("t_end")
    rows = [frame_row(r, steps) for r in log.frames]
    measured = [r for r in rows if warm_t is not None and warm_t <= r["begin"] <= t_end]
    intervals = [(b["begin"] - a["begin"]) * 1000 for a, b in zip(measured, measured[1:])]
    window = sampler.window(warm_t, t_end) if warm_t and t_end else {}
    wall_s = (t_end - warm_t) if warm_t and t_end else None

    artifact_cpu = sum(r["decode_cpu_ms"] for r in measured) / 1000
    if steps[0].startswith("replay"):
        artifact_cpu += sum(r["mesh_cpu_ms"] + r["yolo_cpu_ms"] for r in measured) / 1000
    cpu_s = window.get("cpu_s")
    cam_cpu = log.cam_cpu.get("last", 0) - log.cam_cpu.get("warm", log.cam_cpu.get("first", 0))
    main_cpu = marks.get("main_end", 0) - marks.get("main_warm", marks.get("main_end", 0))

    def pct(seconds):
        return 100 * seconds / wall_s if seconds is not None and wall_s else None

    ui = {}
    for name, calls in ui_calls.items():
        inside = [c for c in calls if warm_t and warm_t <= c[0] <= t_end]
        ui[name] = {"calls": len(inside), "wall_ms": describe([c[1] * 1000 for c in inside]),
                    "cpu_ms_total": sum(c[2] for c in inside) * 1000}

    steps_summary = {}
    for k in ("system_ms", "total_ms", "proc_ms", "mesh_ms", "yolo_ms", "process_frame_other_ms", "loop_other_ms") + \
            tuple(f"{s}_ms" for s in STEP_KEYS):
        steps_summary[k] = describe([r[k] for r in measured])
        cpu_key = k.replace("_ms", "_cpu_ms")
        if measured and cpu_key in measured[0]:
            steps_summary[k]["cpu_mean_ms"] = sum(r[cpu_key] for r in measured) / len(measured)

    return {
        "run": run_index,
        "config": {"scope": args.scope, "pace": args.pace, "video": Path(args.video).name, "video_fps": fps,
                   "warmup_s": args.warmup, "draw_mesh": not args.no_mesh, "smoke": args.smoke,
                   "offscreen": args.offscreen, "model_steps": list(steps),
                   "loaded_modules": {m: m in sys.modules for m in ("torch", "ultralytics", "PyQt5.QtWebEngineWidgets")}},
        "machine": machine_info(),
        "measured": {
            "wall_s": wall_s, "frames": len(measured), "frames_total": len(rows),
            "dropped": sum(r["dropped"] for r in measured), "fps_processed": len(measured) / wall_s if wall_s else None,
        },
        "per_frame_ms": steps_summary,
        "frame_interval_ms": describe(intervals),
        "frame_interval_over_limit": sum(1 for x in intervals if x >= 250),
        "cpu": {
            "process_cpu_s": cpu_s, "process_pct": window.get("cpu_pct"),
            "artifact_cpu_s": artifact_cpu,
            "adjusted_pct": pct(cpu_s - artifact_cpu) if cpu_s is not None else None,
            "camera_thread_pct": pct(cam_cpu), "camera_thread_adjusted_pct": pct(cam_cpu - artifact_cpu),
            "main_thread_pct": pct(main_cpu),
            "logical_cores": machine_info()["logical_cores"],
        },
        "memory_mb": {"before_start": marks.get("rss_before_start_mb"), "peak": window.get("rss_peak_mb"),
                      "mean": window.get("rss_mean_mb")},
        "ui_slots": ui,
    }


def save(out_dir: Path, k: int, result, log, sampler, steps):
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"run{k}.json", result)
    rows = [frame_row(r, steps) for r in log.frames]
    if rows:
        t0 = rows[0]["begin"]
        with gzip.open(out_dir / f"run{k}_frames.csv.gz", "wt", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t_s"] + [c for c in rows[0] if c != "begin"])
            writer.writeheader()
            for r in rows:
                writer.writerow({"t_s": round(r["begin"] - t0, 4),
                                 **{c: (round(v, 4) if isinstance(v, float) else v) for c, v in r.items() if c != "begin"}})
    with open(out_dir / f"run{k}_resources.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t_s", "cpu_pct", "rss_mb"])
        s = sampler.samples
        for a, b in zip(s, s[1:]):
            dt = b["t"] - a["t"]
            writer.writerow([round(b["t"] - s[0]["t"], 3), round(100 * (b["cpu_s"] - a["cpu_s"]) / dt, 1) if dt else "",
                             round(b["rss_mb"], 1)])


def brief(r) -> str:
    m, c, f = r["measured"], r["cpu"], r["per_frame_ms"]

    def fmt(x, unit=""):
        return "-" if x is None else f"{x:.1f}{unit}"

    return (f"  帧数 {m['frames']}（丢帧 {m['dropped']}），处理 {fmt(m['fps_processed'])} fps；"
            f"每帧处理（不含模型）中位数 {fmt(f['system_ms']['median'], ' ms')} / P95 {fmt(f['system_ms']['p95'], ' ms')}；"
            f"帧间隔 P95 {fmt(r['frame_interval_ms']['p95'], ' ms')} / 最大 {fmt(r['frame_interval_ms']['max'], ' ms')}；"
            f"CPU（扣除测量开销）{fmt(c['adjusted_pct'], '%')}，界面线程 {fmt(c['main_thread_pct'], '%')}；"
            f"内存峰值 {fmt(r['memory_mb']['peak'], ' MB')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["exclude", "include"], required=True)
    parser.add_argument("--pace", choices=["realtime", "max"], default="realtime")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--no-mesh", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--smoke", type=int, default=0, help="只处理前 N 帧试跑，不保存结果")
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--cache", default=str(CACHE_DIR))
    parser.add_argument("--final-model", action="store_true",
                        help="「都包含」口径的正式实测：确认 weights/best.pt 已是最终的 4 类模型（#26）")
    parser.add_argument("--child", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.smoke:
        args.warmup = 0.0  # 试跑只有几十帧，不设预热
    if args.smoke or args.child:
        out = None if args.smoke else Path(os.environ["BENCH_OUT_DIR"])
        result = run_once(args, out, args.child or 0)
        print(brief(result), flush=True)
        if args.smoke:
            print(json.dumps(result["per_frame_ms"]["system_ms"], ensure_ascii=False))
        return 0

    if args.scope == "include" and not args.final_model:
        print("⚠️ 「都包含」口径的正式实测要等 #26 的 4 类模型就绪（proposal.md 第 1 节）。"
              "现在请用 --smoke 试跑；模型就绪后加 --final-model 正式测量。")
        return 1
    machine = args.machine or default_machine_tag()
    suffix = "-nomesh" if args.no_mesh else ""
    out_dir = RESULTS_DIR / machine / f"realtime-{args.scope}-{args.pace}{suffix}"
    print(f"结果目录：{out_dir.relative_to(RESULTS_DIR.parent.parent)}")
    for k in range(1, args.runs + 1):
        print(f"第 {k}/{args.runs} 次…", flush=True)
        child = [sys.executable, str(Path(__file__).resolve()), *[a for a in sys.argv[1:]], "--child", str(k)]
        code = subprocess.call(child, env={**os.environ, "BENCH_OUT_DIR": str(out_dir)})
        if code != 0:
            print(f"❌ 第 {k} 次测量失败（退出码 {code}）")
            return code
    print("✅ 完成。汇总：python benchmarks/summarize.py" + (f" --machine {args.machine}" if args.machine else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
