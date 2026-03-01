# Self-Analysis System Demo

本项目为桌面端学生注意力/专注度分析演示（基于 PyQt5 + qfluentwidgets + YOLO/OpenCV）。

## 概览
- 目标：通过摄像头实时检测学生注意力状态（专注/分心/打瞌睡等），并生成统计报告。
- 类型：桌面 GUI 应用（Python + PyQt）。

## 主要特性
- 实时视频流采集与处理（在 `QThread` 中运行推理）。
- **双模态推理**：YOLOv5 用于检测人体/手机/人脸等宏观目标，MediaPipe FaceMesh 提取面部 478 个关键点并计算 EAR/MAR，用于精细的眼睛闭合与张嘴监测。
- 注意力评分引擎封装在 `app/attention_rules.py` 中，可扩展规则。
- UI 支持启动/停止、状态着色、高 DPI 缩放及安全退出（点击窗口关闭时自动释放摄像头）。
- 可视化界面：实时监控页面与占位的统计报告页面（`app/view/`）。
- 兼容 macOS M1/ARM：YOLO 推理类自动选择 `mps` 设备，并修补了 `pathlib.WindowsPath` 的跨平台问题。

## 快速开始

1) 创建虚拟环境（推荐使用 conda）：

```bash
conda create -n sas-demo python=3.10 -y
conda activate sas-demo
# 基本依赖
pip install PyQt5 qfluentwidgets opencv-python numpy
# YOLOv5 依赖
pip install torch torchvision ultralytics
# MediaPipe 用于面部网格
pip install mediapipe
```

说明：
- 如果使用 GPU，请根据系统和 CUDA 版本安装合适的 `torch` 版本。
- macOS M1/M2 用户可利用 `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu` 或直接 `mps` 版本。

2) 准备权重文件

将训练好的权重放在 `weights/` 目录（示例：`weights/best.pt`）。请不要将大型权重推送到远程仓库，建议在 `.gitignore` 中忽略 `weights/*.pt`。

3) 运行应用：

```bash
python main.py
```

运行前请确保系统能访问摄像头；若无摄像头，可在 `app/camera_thread.py` 中修改为读取视频文件。

## 项目结构（简要）

- `main.py`：应用入口，创建主窗口、设置高 DPI 支持并管理导航。
- `app/`：核心代码。
  - `camera_thread.py`：继承自 `QThread`，负责摄像头读入、YOLO/MediaPipe 推理和 UI 信号发送。
  - `yolo_inference.py`：封装YOLOv5推理，自动选择 MPS/CUDA/CPU 并处理跨平台路径问题。
  - `mediapipe_inference.py`：使用 MediaPipe FaceMesh 计算 EAR、MAR，并绘制面部网格。
  - `attention_rules.py`：注意力判定规则与统计逻辑（包含手机检测、PERCLOS疲劳和人脸消失报警）。
  - `view/monitor_interface.py`：监控界面（实时视频、启动/停止按钮、状态标签和异步弹窗）。
  - `view/report_interface.py`：统计报表展示页面，包含统计卡片与图表占位。
- `weights/`：存放模型权重（请勿提交大型权重文件，建议添加 `weights/*.pt` 到 `.gitignore`）。

- `app/attention_rules.py` 以及 `app/mediapipe_inference.py` 为近期新增，用于扩展更精细的生物特征分析。

完整文件说明参见项目中的 `AGENTS.md`。

完整文件说明见项目中的 `AGENTS.md` 文档。

## 开发注意事项

- 所有耗时的 CV/ML 操作必须放入后台线程（`QThread`），避免阻塞主线程。
- 在后台线程中通过信号将处理结果传回 UI，禁止直接修改 UI 控件。
- 图像格式：OpenCV 使用 BGR，显示前需转换为 RGB（`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`）。
- 路径处理请使用 `pathlib` 或 `os.path`，避免硬编码绝对路径。- 应在 `main.py` 中启用 `Qt.AA_EnableHighDpiScaling` 与 `AA_UseHighDpiPixmaps` 以支持高分屏。
- 关闭窗口时会自动安全停止摄像头线程，避免死锁或设备占用。
- 注意更新 `weights` 目录后动态重新加载或重启程序。
- 若添加新依赖（如 MediaPipe），请在本 README 的依赖列表中备注。

## 调试与测试

- 语法检查：

```bash
python -m py_compile main.py
```

- 建议使用 `flake8`/`ruff` 做静态检查，针对更改的文件运行即可。

## 常见问题

- 无法打开摄像头：请检查系统权限（macOS 需在“系统偏好设置 → 隐私与安全性”允许摄像头访问）。
- 推理速度慢：可考虑使用更小的模型、降低输入分辨率或使用 GPU 版本的 PyTorch。

----
