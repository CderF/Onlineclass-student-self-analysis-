import cv2
import time
import numpy as np
from app.attention_rules import AttentionAnalyzer
from app.mediapipe_inference import FaceMeshInference
from app.yolo_inference import ExpressionClassifier
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

        self.classifier = ExpressionClassifier()
        self.mediapipe = FaceMeshInference()
        self.analyzer = AttentionAnalyzer()

    def set_draw_mesh(self, state: bool):
        """控制是否绘制面部网格"""
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

        # 校准缓冲期所需变量
        calibration_duration = 3.0
        calibration_start_time = time.time()
        is_calibrating = True
        pitch_buffer = []
        yaw_buffer = []
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                try:

                    # MediaPipe 必须先跑，全图扫描找脸，并返回 bbox

                    ear, mar, pitch, yaw, roll, bbox, final_frame = self.mediapipe.process_frame(
                        frame, draw_mesh=self.draw_mesh
                    )


                    # 基于人脸框进行裁剪，送入 YOLO26-cls

                    current_expression = "None"
                    all_probs = None

                    if bbox is not None:
                        x_min, y_min, x_max, y_max = bbox
                        # 确保裁剪尺寸有效
                        if x_max > x_min and y_max > y_min:
                            # Numpy 切片裁剪图像 [y:y+h, x:x+w]
                            face_crop = frame[y_min:y_max, x_min:x_max]

                            # 送入分类器进行表情识别！
                            top1_class, top1_conf, all_probs = self.classifier.process_face(face_crop)
                            current_expression = top1_class

                            # 直观展示：将识别到的表情绘制在人脸框上方
                            cv2.rectangle(final_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                            cv2.putText(final_frame, f"Exp: {top1_class} ({top1_conf:.2f})",
                                        (x_min, max(20, y_min - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


                    # 系统状态机 (校准态 vs 监控态)

                    current_time = time.time()

                    if is_calibrating:
                        elapsed = current_time - calibration_start_time
                        remaining = max(0, int(calibration_duration - elapsed) + 1)

                        if elapsed < calibration_duration:
                            # 收集角度数据
                            if pitch is not None and yaw is not None:
                                pitch_buffer.append(pitch)
                                yaw_buffer.append(yaw)

                            self.update_status_signal.emit(f"CALIBRATING: 请正视屏幕... {remaining}s")
                            score = 100
                        else:
                            # 3 秒结束：计算【中位数】并锁定基准线 (彻底免疫极端离群噪点)
                            if len(pitch_buffer) > 0:
                                # --- 核心修改：将 mean 替换为 median ---
                                self.baseline_pitch = np.median(pitch_buffer)
                                self.baseline_yaw = np.median(yaw_buffer)
                                # --------------------------------------
                                print(
                                    f"✅ 校准完成！锁定基准线 -> Pitch: {self.baseline_pitch:.1f}, Yaw: {self.baseline_yaw:.1f}")
                            else:
                                print("⚠️ 校准期间未检测到人脸，使用默认 0 度基准")

                            is_calibrating = False
                            self.update_status_signal.emit("Status: Monitoring Active")
                            score = 100
                    else:
                        # 正式监控期：计算差值 (Delta)
                        delta_pitch = pitch - self.baseline_pitch if pitch is not None else None
                        delta_yaw = yaw - self.baseline_yaw if yaw is not None else None

                        if delta_pitch is not None:
                            cv2.putText(final_frame, f"D_Pitch: {delta_pitch:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8, (0, 255, 0), 2)
                            cv2.putText(final_frame, f"D_Yaw: {delta_yaw:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)

                        # 调用注意力规则引擎
                        score, status_text, alert_data = self.analyzer.process_frame(
                            all_probs, ear, mar, delta_pitch, delta_yaw
                        )
                        self.update_status_signal.emit(status_text)

                        if alert_data is not None:
                            _, alert_title, alert_msg = alert_data
                            self.alert_signal.emit(alert_title, alert_msg)

                    # 发射 UI 更新信号
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