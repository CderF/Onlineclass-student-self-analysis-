"""
把一台机器上的测量结果汇总成中文表格：benchmarks/results/<机器>/summary.md

运行：python benchmarks/summarize.py [--machine 名称]（默认：本机的默认名称）
每帧数据把多次测量的帧合并后再算分位数（去掉开头的预热）；整次测量的指标（CPU、内存）取各次的中位数，并给出范围。
"""
import argparse
import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

from common import FRAME_GAP_LIMIT_MS, IDEAL_FRAME_MS, RESULTS_DIR, default_machine_tag, describe

STEP_ROWS = [
    ("system_ms", "**每帧处理合计（不含模型推理）**"),
    ("process_frame_ms", "　FaceMeshInference.process_frame 合计（不含模型）"),
    ("draw_mesh_ms", "　　画面部网格"),
    ("head_pose_ms", "　　头部姿态（solvePnP）"),
    ("ear_mar_ms", "　　EAR / MAR"),
    ("process_frame_other_ms", "　　其余（BGR→RGB、复制画面、人脸框与平滑）"),
    ("attention_rules_ms", "　注意力规则（AttentionAnalyzer）"),
    ("recorder_ms", "　课程记录（SessionRecorder，每秒写入 SQLite）"),
    ("loop_other_ms", "　采集循环其余（裁剪、叠加文字、转 QPixmap、发信号、校准）"),
    ("mesh_ms", "FaceMesh（回放=测量开销 / 推理=模型）"),
    ("yolo_ms", "表情分类（回放=测量开销 / 推理=模型）"),
    ("decode_ms", "读帧解码（测量开销，已扣除）"),
]
UI_LABELS = {"set_camera_frame": "缩放并显示画面", "update_attention_level": "更新分数圆环",
             "update_status_text": "更新状态文字"}
REPORT_ROWS = [
    ("python", "load_session", "读取一节课（读库 + 问题时段 + 文字反馈）"),
    ("python", "detect", "　其中：问题时段判定"),
    ("python", "feedback", "　其中：文字反馈"),
    ("python", "summarize_sessions", "多节课汇总（5 节）"),
    ("python", "export_one", "CSV 导出：一节课"),
    ("python", "export_all", "CSV 导出：全部（5 节）"),
    ("page", "open_page", "首次打开报告页到图表显示（含 WebEngine 启动）"),
    ("page", "open_summary", "切换到「全体の推移」到图表显示"),
]


def fmt(x, digits=1):
    return "–" if x is None else f"{x:.{digits}f}"


def med_range(values, digits=1):
    vals = [v for v in values if v is not None]
    if not vals:
        return "–"
    m = statistics.median(vals)
    return fmt(m, digits) if len(vals) == 1 else f"{fmt(m, digits)}（{fmt(min(vals), digits)}–{fmt(max(vals), digits)}）"


def load_frames(run_dir: Path, run: dict):
    path = run_dir / f"run{run['run']}_frames.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    warm_frames = run["config"]["warmup_s"] * run["config"]["video_fps"]  # 与 bench_realtime.py 相同：按帧号去掉预热
    return [{k: float(v) for k, v in r.items() if v != ""} for r in rows if float(r["index"]) >= warm_frames]


def realtime_section(run_dir: Path) -> list:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("run*.json"))]
    if not runs:
        return []
    cfg = runs[0]["config"]
    frames = [f for r in runs for f in load_frames(run_dir, r)]
    lines = [f"## {run_dir.name}", "",
             f"口径：{'都排除（回放缓存，不运行模型）' if cfg['scope'] == 'exclude' else '都包含（运行模型）'}；"
             f"送帧：{'按视频帧率 ' + fmt(cfg['video_fps']) + ' fps' if cfg['pace'] == 'realtime' else '尽可能快（时钟换成视频时间）'}；"
             f"面部网格：{'显示' if cfg['draw_mesh'] else '不显示'}；{len(runs)} 次测量，合计 {len(frames)} 帧"
             f"（每次去掉开头 {fmt(cfg['warmup_s'], 0)} 秒）。", ""]

    lines += ["### 每帧耗时（ms，各次测量的帧合并统计）", "", "| 环节 | 中位数 | P95 | 最大 | 平均 CPU |", "|---|---|---|---|---|"]
    for key, label in STEP_ROWS:
        vals = [f[key] for f in frames if key in f]
        d = describe(vals)
        cpu_key = key.replace("_ms", "_cpu_ms")
        cpu = statistics.mean(f[cpu_key] for f in frames) if frames and cpu_key in frames[0] else None
        lines.append(f"| {label} | {fmt(d['median'], 2)} | {fmt(d['p95'], 2)} | {fmt(d['max'], 2)} | {fmt(cpu, 2)} |")

    proc = describe([f["system_ms"] for f in frames])
    lines += ["", "### 整次测量（各次的中位数，括号内为范围）", "", "| 指标 | 值 |", "|---|---|",
              f"| 实际处理帧率（fps） | {med_range([r['measured']['fps_processed'] for r in runs])} |",
              f"| 丢帧数 | {med_range([r['measured']['dropped'] for r in runs], 0)} |",
              f"| 进程 CPU（扣除测量开销，单核 = 100%） | {med_range([r['cpu']['adjusted_pct'] for r in runs])} |",
              f"| 　占全部 {runs[0]['cpu']['logical_cores']} 个逻辑核的比例（%） | "
              f"{med_range([r['cpu']['adjusted_pct'] / r['cpu']['logical_cores'] for r in runs if r['cpu']['adjusted_pct'] is not None], 2)} |",
              f"| 　采集线程（扣除测量开销） | {med_range([r['cpu']['camera_thread_adjusted_pct'] for r in runs])} |",
              f"| 　界面线程（信号处理与绘制） | {med_range([r['cpu']['main_thread_pct'] for r in runs])} |",
              f"| 进程 CPU（未扣除，含解码与回放） | {med_range([r['cpu']['process_pct'] for r in runs])} |",
              f"| 内存：开始前（MB） | {med_range([r['memory_mb']['before_start'] for r in runs], 0)} |",
              f"| 内存：测量期间峰值（MB） | {med_range([r['memory_mb']['peak'] for r in runs], 0)} |",
              f"| 内存：测量期间平均（MB） | {med_range([r['memory_mb']['mean'] for r in runs], 0)} |"]
    for name, label in UI_LABELS.items():
        lines.append(f"| 界面槽函数「{label}」每次（ms，中位数） | "
                     f"{med_range([r['ui_slots'][name]['wall_ms']['median'] for r in runs], 2)} |")
    loaded = runs[0]["config"].get("loaded_modules", {})
    if loaded:
        lines.append(f"| 已导入 torch / ultralytics | {'是' if loaded.get('torch') else '否'} / "
                     f"{'是' if loaded.get('ultralytics') else '否'} |")

    lines += ["", "### 判定", ""]
    lines.append(f"- 理想线（每帧处理 ≤ {IDEAL_FRAME_MS:.1f} ms，按 P95 判定）：P95 = {fmt(proc['p95'])} ms → "
                 f"**{'达标' if proc['p95'] is not None and proc['p95'] <= IDEAL_FRAME_MS else '未达标'}**")
    if cfg["pace"] == "realtime":
        gaps = [(b["t_s"] - a["t_s"]) * 1000 for r in runs for a, b in zip(load_frames(run_dir, r), load_frames(run_dir, r)[1:])]
        gd = describe(gaps)
        over = sum(1 for g in gaps if g >= FRAME_GAP_LIMIT_MS)
        lines.append(f"- 底线（帧间隔 < {FRAME_GAP_LIMIT_MS} ms）：P95 = {fmt(gd['p95'])} ms，最大 = {fmt(gd['max'])} ms，"
                     f"超过底线 {over} 次 → **{'达标' if over == 0 else '未达标'}**")
    else:
        lines.append("- 底线（帧间隔）只在按视频帧率送帧的测量中判定。")
    return lines + [""]


def report_section(run_dir: Path) -> list:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("run*.json"))]
    if not runs:
        return []
    lines = ["## 课后报告页（一次性开销）", "",
             f"合成数据：5 节同课程名、各 {runs[0]['config']['minutes']} 分钟的课程记录，测量对象为最后一节；"
             f"{len(runs)} 次测量（每次新进程），各项取中位数，括号内为范围。", "",
             "| 操作 | 耗时（ms） | CPU 时间（ms；报告页两项含 WebEngine 子进程） | 内存峰值（MB，含子进程） |", "|---|---|---|---|"]
    for part, key, label in REPORT_ROWS:
        items = [r[part][key] for r in runs]
        lines.append(f"| {label} | {med_range([i['wall_ms'] for i in items], 0)} | "
                     f"{med_range([i['cpu_ms'] for i in items], 0)} | {med_range([i['rss_peak_mb'] for i in items], 0)} |")
    page = [r["page"] for r in runs]
    lines += ["",
              f"- 打开报告页前的内存：{med_range([p['rss_before_page_mb'] for p in page], 0)} MB；"
              f"关闭前空闲时（含 WebEngine 子进程）：{med_range([p['rss_idle_after_mb'] for p in page], 0)} MB；"
              f"进程数：{page[0]['open_page']['processes']}",
              f"- 全部显示成功：{'是' if all(p['open_page']['ok'] and p['open_summary']['ok'] for p in page) else '否'}", ""]
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", default=None)
    args = parser.parse_args()
    base = RESULTS_DIR / (args.machine or default_machine_tag())
    if not base.exists():
        print(f"❌ 没有找到 {base}")
        return 1

    any_run = next(base.glob("*/run*.json"), None)
    if any_run is None:
        print(f"❌ {base} 下没有测量结果")
        return 1
    m = json.loads(any_run.read_text(encoding="utf-8"))["machine"]
    lines = [f"# 计算量测量结果：{base.name}", "",
             "由 `benchmarks/summarize.py` 生成，请勿手改。指标定义与测量方法见 `benchmarks/README.md`。", "",
             f"- 机器：{m['cpu']}，{m['physical_cores']} 个物理核 / {m['logical_cores']} 个逻辑核，内存 {m['memory_gb']} GB",
             f"- 系统：{m['os']}（{m['machine']}），Python {m['python']}",
             "- 主要依赖：" + "、".join(f"{k} {v}" for k, v in m["packages"].items() if v), ""]
    for d in sorted(base.glob("realtime-*")):
        lines += realtime_section(d)
    if (base / "report").exists():
        lines += report_section(base / "report")
    out = base / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 已生成 {out.relative_to(RESULTS_DIR.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
