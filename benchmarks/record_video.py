"""
录制约 5 分钟的测试视频（benchmarks/data/session.mp4），代替摄像头作为计算量评估的输入。

运行：在仓库根目录、有摄像头权限的终端里执行
    python benchmarks/record_video.py

按屏幕和终端的提示依次做出各种状态，让注意力规则的每个分支都被执行到。
写入文件的是摄像头原始画面（与程序实际处理的一致）；提示文字只画在预览窗口上。
按 q 或 Esc 提前结束（视频作废，需要重录）。
视频含人脸，只保存在本机：benchmarks/data/ 已加入 .gitignore。
"""
import sys
import time

import cv2

from common import DATA_DIR, VIDEO_META, VIDEO_PATH, write_json

# (时长秒, 预览窗口上的英文短标签, 终端里的中文说明)
SCHEDULE = [
    (10, "LOOK AT SCREEN", "正视屏幕，保持不动（对应程序开头 3 秒的校准）"),
    (60, "NORMAL", "正常听课：看屏幕，可以自然地轻微动头、眨眼"),
    (25, "HEAD DOWN", "低头记笔记，保持低头 25 秒（超过 15 秒触发长时低头）"),
    (30, "NORMAL", "抬头，正常听课"),
    (15, "CLOSE EYES 3s x3", "闭眼约 3 秒再睁开，重复 3 次"),
    (30, "YAWN x3", "打哈欠 3 次，每次嘴张大保持 2 秒以上"),
    (20, "NORMAL", "正常听课"),
    (30, "HEAD DOWN -> HIDE FACE", "先低头，再继续往下或侧转，让脸离开画面约 20 秒"),
    (20, "NORMAL", "回到正常听课"),
    (40, "LEAVE SEAT", "离开座位，让脸完全离开画面 40 秒"),
    (20, "BACK / NORMAL", "回到座位，正常听课直到结束"),
]


def measure_fps(cap, seconds: float = 2.0) -> float:
    """摄像头实际帧率：CAP_PROP_FPS 常常不准，按 2 秒内实际读到的帧数计算"""
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if cap.read()[0]:
            n += 1
    return n / (time.perf_counter() - t0)


def main() -> int:
    cap = cv2.VideoCapture(0)  # 与 app/camera_thread.py 相同：默认摄像头、默认分辨率
    if not cap.isOpened():
        print("❌ 无法打开摄像头：请确认终端有摄像头权限，且没有其他程序在使用摄像头")
        return 1
    ok, frame = cap.read()
    if not ok:
        print("❌ 摄像头没有返回画面")
        return 1
    h, w = frame.shape[:2]
    print("正在测量摄像头帧率…")
    fps = measure_fps(cap)
    print(f"分辨率 {w}x{h}，实测帧率 {fps:.1f} fps")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = VIDEO_PATH.with_suffix(".recording.mp4")
    # 文件头的帧率取整：MPEG-4 要求时间基分母 ≤ 65535，任意小数帧率可能导致编码器打不开。
    # 回放计时用 session.json 里记录的实际平均帧率，与文件头无关
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), max(1, round(fps)), (w, h))
    if not writer.isOpened():
        print("❌ 无法创建视频文件")
        return 1

    total = sum(s[0] for s in SCHEDULE)
    print(f"\n共 {len(SCHEDULE)} 个步骤，约 {total // 60} 分 {total % 60} 秒。5 秒后开始，请坐好并看着屏幕。")
    for k in range(5, 0, -1):
        print(f"  {k}…", flush=True)
        time.sleep(1)

    bounds, acc = [], 0
    for dur, label, text in SCHEDULE:
        bounds.append((acc, acc + dur, label, text))
        acc += dur

    frames, current, aborted = 0, -1, False
    t0 = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= total:
            break
        step = next(i for i, b in enumerate(bounds) if b[0] <= elapsed < b[1])
        if step != current:
            current = step
            s, e, label, text = bounds[step]
            print(f"\n[{step + 1}/{len(bounds)}] {s // 60}:{s % 60:02d}–{e // 60}:{e % 60:02d}  {text}", flush=True)
        ok, frame = cap.read()
        if not ok:
            continue
        writer.write(frame)
        frames += 1

        preview = frame.copy()
        s, e, label, _ = bounds[step]
        left = e - elapsed
        cv2.rectangle(preview, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(preview, f"{step + 1}/{len(bounds)}  {label}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 255), 2)
        cv2.putText(preview, f"{left:4.0f}s left   REC {elapsed:5.0f}/{total}s", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow("record_video (q / Esc: abort)", preview)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            aborted = True
            break

    duration = time.perf_counter() - t0
    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    if aborted:
        tmp.unlink(missing_ok=True)
        print("\n已中止，视频没有保存。")
        return 1

    # 写入的帧率按整段实际平均值记录：读帧和预览会拖慢循环，回放时按这个帧率才与录制时的时间一致
    real_fps = frames / duration
    if abs(real_fps - fps) / fps > 0.05:
        print(f"⚠️ 录制期间平均帧率 {real_fps:.1f} fps，与开头实测的 {fps:.1f} fps 相差较大，回放时按平均值计时")
    tmp.replace(VIDEO_PATH)
    write_json(VIDEO_META, {
        "file": VIDEO_PATH.name, "width": w, "height": h, "frames": frames,
        "fps": round(real_fps, 3), "camera_fps_measured": round(fps, 3), "duration_s": round(duration, 2),
        "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "platform": sys.platform,
        "schedule": [{"start_s": b[0], "end_s": b[1], "label": b[2], "instruction": b[3]} for b in bounds],
    })
    print(f"\n✅ 已保存 {VIDEO_PATH.relative_to(VIDEO_PATH.parents[2])}：{frames} 帧，{duration:.0f} 秒，平均 {real_fps:.1f} fps")
    print("下一步：python benchmarks/build_cache.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
