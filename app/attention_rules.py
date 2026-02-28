import time


class AttentionAnalyzer:
    def __init__(self):
        # 1. 严格复刻旧版的全局参数
        self.EYE_AR_THRESH = 0.15
        self.MAR_THRESH = 0.65
        self.PERCLOS_FATIGUE_THRESH = 0.38

        # 2. 帧计数器
        self.roll = 0
        self.roll_eye = 0
        self.roll_mouth = 0
        self.perclos_history = []

        self.face_not_detected_count = 0
        self.phone_time = 0

        # 为了兼容弹窗冷却时间
        self.last_alert_time = 0

    def process_frame(self, detected_classes, ear, mar):
        """
        接收 YOLO 标签和 MediaPipe 的特征值进行综合判断
        """
        alert_data = None
        status_text = "Status: Normal"
        score = 100
        now = time.time()

        has_phone = 'phone' in detected_classes
        has_face = (ear is not None) or ('face' in detected_classes)

        # --- 1. 手机检测 (保持 40s/2000帧 报警逻辑，这里简化为 5秒连续物理时间) ---
        if has_phone:
            self.phone_time += 1
            if self.phone_time > 150:  # 大约 5-10 秒
                if now - self.last_alert_time > 30:
                    alert_data = ('phone', 'Alert', 'Please put down your phone and focus!')
                    self.last_alert_time = now
            status_text = "Distraction: PHONE"
            score = 50
        else:
            self.phone_time = 0

        # --- 2. 疲劳检测 (PERCLOS 逻辑复刻) ---
        if has_face and ear is not None and mar is not None:
            self.roll += 1

            if ear < self.EYE_AR_THRESH:
                self.roll_eye += 1
            if mar > self.MAR_THRESH:
                self.roll_mouth += 1

            # 每 400 帧计算一次 PERCLOS
            if self.roll >= 400:
                perclos = (self.roll_eye / self.roll) + (self.roll_mouth / self.roll) * 0.2
                self.perclos_history.append(round(perclos, 3))
                if len(self.perclos_history) > 3:
                    self.perclos_history.pop(0)

                # 重置计数器
                self.roll = 0
                self.roll_eye = 0
                self.roll_mouth = 0

        # --- 3. 人脸消失逻辑 (复刻：睡眠 vs 走神) ---
        if not has_face:
            self.face_not_detected_count += 1
            status_text = "Status: FACE MISSING"
            score = 30

            # 复刻旧版：250帧无脸
            if self.face_not_detected_count == 250:
                is_deep_sleep = False
                if len(self.perclos_history) == 3 and all(
                        s > self.PERCLOS_FATIGUE_THRESH for s in self.perclos_history):
                    is_deep_sleep = True

                if now - self.last_alert_time > 30:
                    if is_deep_sleep:
                        alert_data = ('deep_sleep', 'CRITICAL', 'High Fatigue & Inactivity Detected! Take a break.')
                    else:
                        alert_data = ('absence', 'Alert', 'User Absent! Please return to seat.')
                    self.last_alert_time = now
        else:
            self.face_not_detected_count = 0

        # 判断最终状态显示
        if len(self.perclos_history) > 0 and self.perclos_history[-1] > self.PERCLOS_FATIGUE_THRESH:
            status_text = "Focus: FATIGUE DETECTED"
            score = 60

        return score, status_text, alert_data