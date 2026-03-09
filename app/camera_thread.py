import cv2
import time
import numpy as np
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
        self.draw_mesh = True

        self.yolo = YOLOInference()
        self.mediapipe = FaceMeshInference()
        self.analyzer = AttentionAnalyzer()

    def set_draw_mesh(self, state: bool):
        self.draw_mesh = state

    def run(self):
        print("后台线程已启动，尝试打开摄像头...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ 错误：无法打开摄像头！")
            self.update_status_signal.emit("Status: Camera Failed")
            self.alert_signal.emit("Camera Error", "无法连接摄像头，请检查设备是否被占用或是否有权限！")
            return

        print("✅ 摄像头已打开，开始读取画面...")

        # ====== 新增：校准缓冲期所需变量 ======
        calibration_duration = 3.0  # 倒计时 3 秒
        calibration_start_time = time.time()
        is_calibrating = True
        pitch_buffer = []
        yaw_buffer = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0
        # =================================

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                try:
                    annotated_frame, detected_classes = self.yolo.process_frame(frame)

                    # 接收新增的姿态返回值
                    ear, mar, pitch, yaw, roll, final_frame = self.mediapipe.process_frame(
                        annotated_frame, draw_mesh=self.draw_mesh
                    )

                    current_time = time.time()

                    # ==========================================
                    # 核心逻辑：系统状态机 (校准态 vs 监控态)
                    # ==========================================
                    if is_calibrating:
                        elapsed = current_time - calibration_start_time
                        remaining = max(0, int(calibration_duration - elapsed) + 1)

                        if elapsed < calibration_duration:
                            # 1. 处于 3 秒缓冲期内：收集角度数据
                            if pitch is not None and yaw is not None:
                                pitch_buffer.append(pitch)
                                yaw_buffer.append(yaw)

                            # 更新 UI 提示语
                            self.update_status_signal.emit(f"CALIBRATING: 请正视屏幕... {remaining}s")
                            score = 100  # 校准期间保持满分
                        else:
                            # 2. 3 秒结束：计算平均值并锁定基准线
                            if len(pitch_buffer) > 0:
                                self.baseline_pitch = np.mean(pitch_buffer)
                                self.baseline_yaw = np.mean(yaw_buffer)
                                print(
                                    f"✅ 校准完成！锁定基准线 -> Pitch: {self.baseline_pitch:.1f}, Yaw: {self.baseline_yaw:.1f}")
                            else:
                                print("⚠️ 校准期间未检测到人脸，使用默认 0 度基准")

                            is_calibrating = False  # 切换状态为监控态
                            self.update_status_signal.emit("Status: Monitoring Active")
                            score = 100

                    else:
                        # 3. 正式监控期：计算当前角度与基准线的差值 (Delta)
                        delta_pitch = pitch - self.baseline_pitch if pitch is not None else None
                        delta_yaw = yaw - self.baseline_yaw if yaw is not None else None

                        # 为了方便你直观感受，我暂时把相对 Delta 角度绘制在左上角
                        if delta_pitch is not None:
                            cv2.putText(final_frame, f"D_Pitch: {delta_pitch:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 255, 0), 2)
                            cv2.putText(final_frame, f"D_Yaw: {delta_yaw:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)

                        # 调用之前的评分系统
                        score, status_text, alert_data = self.analyzer.process_frame(
                            detected_classes, ear, mar, delta_pitch, delta_yaw
                        )
                        self.update_status_signal.emit(status_text)

                        if alert_data is not None:
                            _, alert_title, alert_msg = alert_data
                            self.alert_signal.emit(alert_title, alert_msg)
                    # ==========================================

                    self.update_score_signal.emit(score)

                    color_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = color_frame.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(color_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    self.change_pixmap_signal.emit(QPixmap.fromImage(qt_image))

                except Exception as e:
                    print(f"❌ 后台处理帧时发生异常: {e}")

        cap.release()
        print("后台线程已安全退出。")

    def stop(self):
        self._run_flag = False
        self.wait()