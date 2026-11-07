"""
对测试视频跑一遍模型，缓存每帧的 MediaPipe 关键点与 YOLO 表情概率，供「都排除」口径回放（benchmarks/proposal.md 第 2 节）。

运行：在仓库根目录执行（需要 weights/best.pt）
    python benchmarks/build_cache.py [--video PATH] [--out DIR]

处理顺序与 app/camera_thread.py 相同：整帧送 FaceMeshInference.process_frame（含人脸框的 EMA 平滑），
再按人脸框裁剪送 ExpressionClassifier.process_face。
输出（benchmarks/data/cache/，含人脸信息，不提交）：
  landmarks.bin  每帧 FaceMesh 结果（NormalizedLandmarkList 的序列化字节，按帧首尾相接）
  index.npy      (帧数, 2) int64：每帧在 landmarks.bin 中的偏移与长度；长度 0 表示该帧没有检测到人脸
  probs.npy      (帧数, 类别数) float32：表情概率；没有人脸框或模型未加载的帧为 NaN
  meta.json      帧数、类别名、视频信息
"""
import argparse
import sys
import time

import numpy as np

from common import CACHE_DIR, VIDEO_PATH, setup_path, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(VIDEO_PATH))
    parser.add_argument("--out", default=str(CACHE_DIR))
    args = parser.parse_args()

    setup_path()
    import cv2
    from pathlib import Path
    from app.mediapipe_inference import FaceMeshInference
    from app.yolo_inference import ExpressionClassifier

    video, out = Path(args.video), Path(args.out)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"❌ 无法打开视频 {video}；请先运行 python benchmarks/record_video.py")
        return 1
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mesh = FaceMeshInference()
    classifier = ExpressionClassifier()
    if classifier.model is None:
        print("❌ 表情模型加载失败（weights/best.pt），无法生成概率缓存")
        return 1
    names = [classifier.classes[i] for i in sorted(classifier.classes)]

    # 截获 FaceMesh 的原始结果：process_frame 只返回由关键点算出的特征，不返回关键点本身
    captured = {}
    real_process = mesh.face_mesh.process

    def capture(rgb):
        captured["result"] = result = real_process(rgb)
        return result

    mesh.face_mesh.process = capture

    out.mkdir(parents=True, exist_ok=True)
    index, probs = [], []
    t0 = time.perf_counter()
    with open(out / "landmarks.bin", "wb") as f:
        offset = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, _, _, _, _, bbox, _ = mesh.process_frame(frame, draw_mesh=False)
            faces = captured["result"].multi_face_landmarks
            data = faces[0].SerializeToString() if faces else b""
            f.write(data)
            index.append((offset, len(data)))
            offset += len(data)

            p = None
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                if x_max > x_min and y_max > y_min:
                    _, _, p = classifier.process_face(frame[y_min:y_max, x_min:x_max])
            probs.append(np.full(len(names), np.nan, np.float32) if p is None else np.asarray(p, np.float32))

            if len(index) % 300 == 0:
                print(f"  {len(index)}/{n_total} 帧", flush=True)
    cap.release()

    np.save(out / "index.npy", np.asarray(index, np.int64))
    np.save(out / "probs.npy", np.stack(probs))
    faces = sum(1 for _, n in index if n)
    write_json(out / "meta.json", {
        "video": video.name, "video_bytes": video.stat().st_size, "frames": len(index),
        "frames_with_face": faces, "class_names": names,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - t0, 1),
    })
    print(f"✅ 缓存完成：{len(index)} 帧（有人脸 {faces} 帧），类别 {names}，用时 {time.perf_counter() - t0:.0f} 秒")
    return 0


if __name__ == "__main__":
    sys.exit(main())
