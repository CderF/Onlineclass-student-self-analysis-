import cv2
import math
import numpy as np
import mediapipe as mp


class FaceMeshInference:
    """使用 MediaPipe 提取面部特征，计算 EAR、MAR，并解算头部 3D 姿态 (Pitch, Yaw, Roll)"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 预定义 MediaPipe 的关键点索引
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.MOUTH_INNER = [78, 308, 13, 14]

        # ====== 新增：预定义的 3D 面部通用模型关键点 (XYZ 坐标) ======
        # 选择 6 个最具代表性的锚点：鼻尖、下巴、左右眼角、左右嘴角
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 1: 鼻尖 (原点)
            (0.0, -330.0, -65.0),  # 152: 下巴
            (-225.0, 170.0, -135.0),  # 33: 左眼左眼角
            (225.0, 170.0, -135.0),  # 263: 右眼右眼角
            (-150.0, -150.0, -125.0),  # 61: 左嘴角
            (150.0, -150.0, -125.0)  # 291: 右嘴角
        ], dtype=np.float64)
        # =========================================================

    def _euclidean_distance(self, p1, p2):
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def _calculate_ear(self, eye_points, landmarks, img_w, img_h):
        pts = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in eye_points]
        v1 = self._euclidean_distance(pts[1], pts[5])
        v2 = self._euclidean_distance(pts[2], pts[4])
        h = self._euclidean_distance(pts[0], pts[3])
        if h == 0: return 0
        return (v1 + v2) / (2.0 * h)

    def _calculate_mar(self, mouth_points, landmarks, img_w, img_h):
        pts = [(landmarks.landmark[i].x * img_w, landmarks.landmark[i].y * img_h) for i in mouth_points]
        v = self._euclidean_distance(pts[2], pts[3])
        h = self._euclidean_distance(pts[0], pts[1])
        if h == 0: return 0
        return v / h

    # ====== PnP 头部姿态解算核心方法 ======
    def estimate_head_pose(self, landmarks, img_w, img_h):
        """利用 OpenCV 的 PnP 算法计算头部的 Pitch (俯仰), Yaw (偏航), Roll (翻滚) 角度"""

        # 1. 提取对应的 2D 像素坐标
        image_points = np.array([
            (landmarks.landmark[1].x * img_w, landmarks.landmark[1].y * img_h),  # 鼻尖
            (landmarks.landmark[152].x * img_w, landmarks.landmark[152].y * img_h),  # 下巴
            (landmarks.landmark[33].x * img_w, landmarks.landmark[33].y * img_h),  # 左眼角
            (landmarks.landmark[263].x * img_w, landmarks.landmark[263].y * img_h),  # 右眼角
            (landmarks.landmark[61].x * img_w, landmarks.landmark[61].y * img_h),  # 左嘴角
            (landmarks.landmark[291].x * img_w, landmarks.landmark[291].y * img_h)  # 右嘴角
        ], dtype=np.float64)

        # 2. 伪造摄像机内参矩阵 (假设焦距等于图像宽度，光心在图像中心)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))  # 假设没有镜头畸变

        # 3. 解算 PnP (Perspective-n-Point)
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0, 0, 0

        # 4. 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # 5. 从旋转矩阵中提取欧拉角 (Euler Angles)
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        # OpenCV 返回的 euler_angles 是一个二维数组
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]

        # ==============================================================
        # 核心防线 1：欧拉角万向节跳变规范化 (Angle Normalization)
        # 强制将异常翻转的角度拉回 -90 到 +90 的人类生理极限范围内
        # ==============================================================
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

        if yaw > 90:
            yaw -= 180
        elif yaw < -90:
            yaw += 180

        if roll > 90:
            roll -= 180
        elif roll < -90:
            roll += 180

        # ==============================================================
        # 统一物理直觉：确保 "低头" 的 Pitch 是负数
        # (因为模型坐标系 Y 轴向下的原因，原版 Pitch 低头往往是正数，这里强行翻转)
        # ==============================================================
        pitch = -pitch

        return pitch, yaw, roll

    # ============================================

    def process_frame(self, frame, draw_mesh=True):
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        ear, mar = None, None
        pitch, yaw, roll = None, None, None
        bbox = None  # <--- 新增：用于存放人脸边界框 (x_min, y_min, x_max, y_max)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            if draw_mesh:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            left_ear = self._calculate_ear(self.LEFT_EYE, landmarks, img_w, img_h)
            right_ear = self._calculate_ear(self.RIGHT_EYE, landmarks, img_w, img_h)
            ear = (left_ear + right_ear) / 2.0
            mar = self._calculate_mar(self.MOUTH_INNER, landmarks, img_w, img_h)

            pitch, yaw, roll = self.estimate_head_pose(landmarks, img_w, img_h)

            # ====== 新增：计算人脸边界框 (Bounding Box) ======
            # 提取所有关键点的 x, y 坐标
            x_coords = [lm.x for lm in landmarks.landmark]
            y_coords = [lm.y for lm in landmarks.landmark]

            # 计算最小外接矩形
            x_min, x_max = int(min(x_coords) * img_w), int(max(x_coords) * img_w)
            y_min, y_max = int(min(y_coords) * img_h), int(max(y_coords) * img_h)

            # 增加 20% 的外扩 Padding，确保包含完整的面部特征
            pad_w = int((x_max - x_min) * 0.2)
            pad_h = int((y_max - y_min) * 0.2)

            x_min = max(0, x_min - pad_w)
            y_min = max(0, y_min - pad_h)
            x_max = min(img_w, x_max + pad_w)
            y_max = min(img_h, y_max + pad_h)

            bbox = (x_min, y_min, x_max, y_max)
            # ============================================

        # <--- 修改这里：多返回一个 bbox --->
        return ear, mar, pitch, yaw, roll, bbox, frame