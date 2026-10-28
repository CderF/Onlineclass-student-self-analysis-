import cv2
import time
import numpy as np
from app import session_store
from app.attention_rules import AttentionAnalyzer
from app.mediapipe_inference import FaceMeshInference
from app.session_recorder import SessionRecorder
from app.yolo_inference import ExpressionClassifier
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap


# 空间状态 → 状态文字的颜色语气（界面按语气上色，不再匹配文字关键词）
SPATIAL_STATE_TONES = {
    "distracted": "danger", "hidden_abnormal": "danger", "absent": "danger",
    "taking_notes": "success", "hidden_notes": "success",
}


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)
    update_score_signal = pyqtSignal(int)
    update_status_signal = pyqtSignal(str, str)  # (状态文字, 语气: info / success / warning / danger / muted)
    alert_signal = pyqtSignal(str, str)

    def __init__(self, db_path=None):
        super().__init__()
        self._run_flag = True
        self.draw_mesh = True

        # 课程记录（docs/adr/0002）：摄像头首次成功打开时建立，校准结束后开始写入
        self.db_path = db_path if db_path is not None else session_store.DEFAULT_DB_PATH
        self.session_id = None
        self.recorder = None  # 只在 run() 线程内创建和使用
        self._pausing = False
        self._session_finished = False
        self._record_failed = False

        self.classifier = ExpressionClassifier()
        self.mediapipe = FaceMeshInference()
        self.analyzer = AttentionAnalyzer()

        # 校准基准在整次课程记录内有效，暂停后继续时沿用，不重新校准
        self.is_calibrated = False
        self.baseline_pitch = 0.0
        self.baseline_yaw = 0.0

    def set_draw_mesh(self, state: bool):
        """控制是否绘制面部网格"""
        self.draw_mesh = state

    def run(self):
        print("后台线程已启动，尝试打开摄像头...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ 错误：无法打开摄像头！")
            self.update_status_signal.emit("カメラを開けません", "warning")
            self.alert_signal.emit("カメラエラー", "カメラに接続できません。他のアプリが使用中でないか、アクセス権限があるか確認してください。")
            return

        print("✅ 摄像头已打开，开始读取画面...")
        if self.session_id is None:
            self._guard_record(self._create_session)

        # 校准缓冲期所需变量 校准期为3秒 同时也是Yolo积累原始数据的期间
        """
            关于3秒的校准期具体时长还需探讨
        """
        calibration_duration = 3.0
        calibration_start_time = time.time()
        is_calibrating = not self.is_calibrated
        pitch_buffer = []
        yaw_buffer = []
        if not is_calibrating:
            self.update_status_signal.emit("モニタリング中", "success")
            self._guard_record(self._start_recording, time.time())

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

                            self.update_status_signal.emit(f"キャリブレーション中：画面を正面から見てください… 残り {remaining} 秒", "info")
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
                            self.is_calibrated = True
                            self.update_status_signal.emit("モニタリング中", "success")
                            score = 100
                            self._guard_record(self._set_baseline)
                            self._guard_record(self._start_recording, current_time)
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
                        result = self.analyzer.process_frame(
                            all_probs, ear, mar, delta_pitch, delta_yaw
                        )
                        score = result.score
                        tone = SPATIAL_STATE_TONES.get(result.spatial_state, "warning")
                        self.update_status_signal.emit(result.status_text, tone)

                        if result.alert is not None:
                            _, alert_title, alert_msg = result.alert
                            self.alert_signal.emit(alert_title, alert_msg)

                        if self.recorder is not None:
                            self._guard_record(self.recorder.add_frame, result, current_time)

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
        if self.recorder is not None:
            # 暂停：记下暂停点；结束：只写完剩余数据，「正常结束」由 stop() 在调用方线程标记
            end = self.recorder.pause if self._pausing else self.recorder.close
            self._guard_record(end, time.time())
            self.recorder = None
        print("后台线程已安全退出。")

    def stop(self):
        """结束本次课程记录（运行中或暂停中都可调用，可重复调用）"""
        self._pausing = False
        self._run_flag = False
        self.wait()
        self._finish_session()

    def pause(self):
        """暂停：结束采集循环并释放摄像头"""
        self._pausing = True
        self._run_flag = False
        self.wait()

    def resume(self):
        """继续：重新打开摄像头；滑动窗口与计时器全部清空，校准基准保留"""
        self.analyzer = AttentionAnalyzer()
        self._pausing = False
        self._run_flag = True
        self.start()

    # --- 课程记录 ---

    def _create_session(self):
        conn = session_store.connect(self.db_path)
        try:
            self.session_id = session_store.create_session(conn, time.time())
        finally:
            conn.close()

    def _set_baseline(self):
        if self.session_id is None:
            return
        conn = session_store.connect(self.db_path)
        try:
            session_store.set_baseline(conn, self.session_id, self.baseline_pitch, self.baseline_yaw)
        finally:
            conn.close()

    def _start_recording(self, start_wall: float):
        if self.session_id is not None:
            self.recorder = SessionRecorder(self.session_id, start_wall, self.db_path)

    def _finish_session(self):
        if self.session_id is None or self._session_finished:
            return
        self._session_finished = True
        try:
            conn = session_store.connect(self.db_path)
            try:
                if not session_store.finish_session(conn, self.session_id, time.time()):
                    print("ℹ️ 本次没有记录到任何数据，已删除空的课程记录")
                    self.session_id = None  # 界面据此不再询问课程名
            finally:
                conn.close()
        except Exception as e:
            print(f"❌ 结束课程记录失败: {e}")

    def _guard_record(self, func, *args):
        """记录出错不影响实时监控：停止本次记录并提示一次"""
        if self._record_failed:
            return
        try:
            func(*args)
        except Exception as e:
            print(f"❌ 课程记录写入失败，本次课程将不再记录: {e}")
            self._record_failed = True
            self.recorder = None
            self.alert_signal.emit("記録エラー", "学習データを保存できませんでした。モニタリングは続行しますが、この授業は記録されません。")