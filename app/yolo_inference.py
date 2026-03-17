import cv2
import numpy as np
from ultralytics import YOLO
import platform
import pathlib

# 跨平台补丁 (保留，以防万一)
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath
# ===================================================

class ExpressionClassifier:
    """专为最新 YOLO26-cls 定制的人脸表情分类推理类"""

    def __init__(self, weights_path='weights/best.pt'):
        """
        初始化模型。
        :param weights_path: 默认使用官方刚发布的 YOLO26 nano 分类模型进行占位测试。
                             将来训练好 RAF-DB 后，替换为我们自己的 'weights/best_fer.pt'
        """
        try:
            print(f"正在加载分类模型: {weights_path}...")
            # Ultralytics 会自动处理设备分配，并且本地没有 yolo26n-cls.pt 时会自动下载
            self.model = YOLO(weights_path)
            self.classes = self.model.names
            print(f"✅ YOLO26-cls 模型加载成功！共包含 {len(self.classes)} 个类别。")
        except Exception as e:
            print(f"❌ YOLO26-cls 模型加载失败: {e}")
            self.model = None

    def process_face(self, face_img):
        """
        处理传入的人脸切片图像，输出分类结果
        :param face_img: OpenCV 格式的人脸 ROI 图像 (BGR)
        :return: (最高概率类别名称, 置信度, 所有类别的概率数组)
        """
        # 安全防御：防止传入空图导致程序崩溃
        if self.model is None or face_img is None or face_img.size == 0:
            return "Unknown", 0.0, None

        # 推理 (设置 verbose=False 避免每帧都在控制台打印日志)
        results = self.model(face_img, verbose=False)

        # 提取分类结果概率
        probs = results[0].probs

        # 1. 获取最高概率的索引、名称和置信度
        top1_idx = probs.top1
        top1_class = self.classes[top1_idx]
        top1_conf = probs.top1conf.item()

        # 2. 提取所有类别的概率分布 (为1分钟时序平滑矩阵做准备)
        all_probs = probs.data.cpu().numpy()

        return top1_class, top1_conf, all_probs