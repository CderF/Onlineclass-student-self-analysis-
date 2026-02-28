import cv2
import torch
import pathlib
import platform

# 跨平台补丁
# 如果当前系统不是 Windows (比如你的 Mac)，则强制将 WindowsPath 重定向为 PosixPath
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
# ===================================================

class YOLOInference:
    """专为 YOLOv5 定制的推理类，已解决跨平台加载问题"""

    def __init__(self, weights_path='weights/best.pt', conf_thres=0.4):
        self.conf_thres = conf_thres
        # 智能选择设备：如果有 Mac M 系列芯片则用 mps 加速
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))

        try:
            print(f"正在通过 torch.hub 加载 YOLOv5 模型 (设备: {self.device})...")
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            self.model.to(self.device)
            self.model.conf = self.conf_thres

            self.classes = self.model.names if hasattr(self.model, 'names') else []
            print("✅ YOLOv5 模型加载成功！类别包含:", self.classes)
        except Exception as e:
            print(f"❌ YOLOv5 模型加载失败: {e}")
            self.model = None

    def process_frame(self, frame):
        """处理单帧图像"""
        if self.model is None:
            return frame, []

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(img)

        detected_classes = []

        df = results.pandas().xyxy[0]
        if not df.empty:
            detected_classes = df['name'].tolist()

        results.render()
        annotated_frame = results.ims[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        return annotated_frame, detected_classes