import time


class AttentionAnalyzer:
    def __init__(self):
        # 1. 严格复刻旧版的全局参数
        self.EYE_AR_THRESH = 0.15
        self.MAR_THRESH = 0.65
        self.PERCLOS_FATIGUE_THRESH = 0.38

        # ====== 空间姿态判定阈值 ======
        # 注意：你需要根据实际测试确定低头时 D_Pitch 是正数还是负数。
        # 在标准的 OpenCV 坐标系分解中，低头通常会导致 Pitch 变为负数。
        # 如果你测试时发现低头 D_Pitch 变成了 +20，请把这里改成正数。
        self.PITCH_DOWN_THRESH = -20.0
        self.HEAD_DOWN_MAX_FRAMES = 300  # 允许连续低头的最大帧数（按 20fps 算大约 15 秒）
        # ==================================

        # 2. 帧计数器
        self.roll = 0
        self.roll_eye = 0
        self.roll_mouth = 0
        self.perclos_history = []

        self.face_not_detected_count = 0
        self.phone_time = 0
        self.head_down_count = 0  # 新增：低头专用计时器

        # 为了兼容弹窗冷却时间
        self.last_alert_time = 0

    def process_frame(self, detected_classes, ear, mar, delta_pitch, delta_yaw):
        """
        接收 YOLO 标签、MediaPipe 特征以及 3D 相对姿态进行综合判断
        """
        alert_data = None
        status_text = "Status: Normal"
        score = 100
        now = time.time()

        has_phone = 'phone' in detected_classes
        has_face = (ear is not None)

        # --- 1. 手机检测 (不变) ---
        if has_phone:
            self.phone_time += 1
            if self.phone_time > 150:
                if now - self.last_alert_time > 30:
                    alert_data = ('phone', 'Alert', 'Please put down your phone and focus!')
                    self.last_alert_time = now
            status_text = "Distraction: PHONE"
            score = 50
        else:
            self.phone_time = 0

        # --- 2. 疲劳检测 (PERCLOS) ---
        if has_face and ear is not None and mar is not None:
            self.roll += 1

            if ear < self.EYE_AR_THRESH:
                self.roll_eye += 1
            if mar > self.MAR_THRESH:
                self.roll_mouth += 1

            if self.roll >= 400:
                perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                self.perclos_history.append(round(perclos, 3))
                if len(self.perclos_history) > 3:
                    self.perclos_history.pop(0)

                self.roll = 0
                self.roll_eye = 0
                self.roll_mouth = 0

        # --- 3. 基于相对空间角度的“低头/走神”检测 ---
        if has_face and delta_pitch is not None:
            # 判断是否低头 (注意符号！)
            if delta_pitch < self.PITCH_DOWN_THRESH:
                self.head_down_count += 1
                status_text = "Focus: HEAD DOWN"
                score = 60

                # 持续低头超过阈值 (约 15 秒)
                if self.head_down_count > self.HEAD_DOWN_MAX_FRAMES:
                    if now - self.last_alert_time > 30:
                        alert_data = ('head_down', 'Alert', '长时间低头！如果是玩手机请专心，记笔记请注意颈椎休息。')
                        self.last_alert_time = now
            else:
                # 抬头恢复，重置计数器 (容许正常的抬头听课行为)
                self.head_down_count = 0
        else:
            self.head_down_count = 0

        # --- 4. 人脸彻底消失逻辑 (单纯的离座判断) ---
        if not has_face:
            self.face_not_detected_count += 1
            status_text = "Status: ABSENT"
            score = 30

            # 250帧无脸 (约 10-12 秒)
            if self.face_not_detected_count == 250:
                if now - self.last_alert_time > 30:
                    alert_data = ('absence', 'Alert', 'User Absent! Please return to seat.')
                    self.last_alert_time = now
        else:
            self.face_not_detected_count = 0

        # --- 5. 疲劳状态最高优先级覆写 ---
        if len(self.perclos_history) > 0 and self.perclos_history[-1] > self.PERCLOS_FATIGUE_THRESH:
            status_text = "Focus: FATIGUE DETECTED"
            score = 50

        return score, status_text, alert_data