import cv2
from app.attention_rules import AttentionAnalyzer
from app.yolo_inference import YOLOInference
from app.mediapipe_inference import FaceMeshInference
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)
    update_score_signal = pyqtSignal(int)
    update_status_signal = pyqtSignal(str)
    alert_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        # 初始化两大神兵利器
        self.yolo = YOLOInference()
        self.mediapipe = FaceMeshInference()
        self.analyzer = AttentionAnalyzer()

    def run(self):
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                # 1. YOLO 提取宏观行为，会在 frame 上画 YOLO 的框
                annotated_frame, detected_classes = self.yolo.process_frame(frame)

                # 2. MediaPipe 接力处理 annotated_frame，并在上面画网格
                # 注意：这里传入的是 annotated_frame，并且接收返回的带有网格的帧
                ear, mar, final_frame = self.mediapipe.process_frame(annotated_frame, draw_mesh=True)

                # 3. 传入复刻的旧版逻辑引擎
                score, status_text, alert_data = self.analyzer.process_frame(detected_classes, ear, mar)

                # 4. 发射 UI 更新信号
                self.update_score_signal.emit(score)
                self.update_status_signal.emit(status_text)

                if alert_data is not None:
                    _, alert_title, alert_msg = alert_data
                    self.alert_signal.emit(alert_title, alert_msg)

                # 5. 画面转换发送 (使用 final_frame)
                color_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = color_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(color_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.change_pixmap_signal.emit(QPixmap.fromImage(qt_image))

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()