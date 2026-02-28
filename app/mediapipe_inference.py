import cv2
import math
import mediapipe as mp


class FaceMeshInference:
    """使用 MediaPipe 提取面部 478 个关键点，并计算 EAR 和 MAR，支持绘制面部网格"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # 新增：引入 MediaPipe 的画笔工具和默认样式
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

    def process_frame(self, frame, draw_mesh=True):
        """
        处理帧，返回 (ear, mar, annotated_frame)
        :param draw_mesh: 是否在图像上绘制绿色的面部网格
        """
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        ear, mar = None, None

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # --- 新增：绘制面部网格 ---
            if draw_mesh:
                # 绘制网格底纹
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                # 绘制眼眶、嘴唇等轮廓线
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
            # --------------------------

            left_ear = self._calculate_ear(self.LEFT_EYE, landmarks, img_w, img_h)
            right_ear = self._calculate_ear(self.RIGHT_EYE, landmarks, img_w, img_h)
            ear = (left_ear + right_ear) / 2.0

            mar = self._calculate_mar(self.MOUTH_INNER, landmarks, img_w, img_h)

        return ear, mar, frame