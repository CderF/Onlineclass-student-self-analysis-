import time
import numpy as np
from collections import deque


class AttentionAnalyzer:
    def __init__(self):
        # ==============================================================
        # 1. 空间姿态与物理规则参数
        # ==============================================================
        self.PITCH_DOWN_THRESH = -15.0
        self.HEAD_DOWN_MAX_FRAMES = 300
        self.face_not_detected_count = 0
        self.head_down_count = 0
        self.last_alert_time = 0
        self.eyes_closed_start_time = None  # 用于记录连续闭眼的起始时间

        # ==============================================================
        # 2. 多模态融合时间戳滑动窗口 (全线对齐 8 秒)
        # ==============================================================
        self.micro_buffer = deque()  # 8 秒微观表情概率: (now, all_probs_clean)
        self.perclos_buffer = deque()  # 8 秒疲劳特征缓冲: (now, is_blink, is_yawn)
        self.macro_buffer = deque()  # 60 秒宏观离散标签: (now, state_label)

        # YOLO 类别索引映射
        self.EMOTION_IDX = {
            'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
            'Neutral': 4, 'Sad': 5, 'Surprise': 6
        }

    def process_frame(self, all_probs, ear, mar, delta_pitch, delta_yaw):
        """
        接收 YOLO 分类概率、MediaPipe 特征以及 3D 相对姿态进行综合判断
        """
        alert_data = None
        status_text = "Status: Initializing..."
        score = 100
        now = time.time()
        has_face = (ear is not None and mar is not None)

        current_cognitive_state = "Neutral"  # 默认状态

        # ==============================================================
        # 阶段一 & 中观层：物理防污染拦截 + 微观概率融合 + 疲劳升格
        # ==============================================================
        if has_face and all_probs is not None and len(all_probs) == 7:

            # --- 策略一：前向特征拦截 (物理防污染) ---
            is_yawn = (mar > 0.65)
            is_blink_frame = (ear < 0.15)
            #连续闭眼 2 秒计时器
            if is_blink_frame:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = now  # 刚闭上的瞬间，记录时间
                closed_duration = now - self.eyes_closed_start_time
            else:
                self.eyes_closed_start_time = None  # 一旦睁眼，计时器立刻清零
                closed_duration = 0.0
            #只有闭眼2秒以上才会被认为在闭眼
            is_blink = (closed_duration > 2.0)

            probs_clean = np.copy(all_probs)

            # 如果正在打哈欠或闭眼，强行剥夺 Happy 和 Surprise 的概率，防止 YOLO 误判
            if is_yawn or is_blink:
                probs_clean = np.zeros(7) # 7个概率全变 0
                probs_clean[self.EMOTION_IDX['Neutral']] = 1.0  # 唯独 Neutral 设为 1.0


            # 1. 存入带时间戳的数据包裹
            self.micro_buffer.append((now, probs_clean))
            self.perclos_buffer.append((now, is_blink_frame, is_yawn))

            # 2. 核心：双队列统一剔除超过 8.0 秒的老数据
            while self.micro_buffer and (now - self.micro_buffer[0][0]) > 8.0:
                self.micro_buffer.popleft()
            while self.perclos_buffer and (now - self.perclos_buffer[0][0]) > 8.0:
                self.perclos_buffer.popleft()

            # 3. 提取纯概率数组进行平均计算
            probs_only = [item[1] for item in self.micro_buffer]
            avg_probs = np.mean(probs_only, axis=0)

            p_anger = avg_probs[self.EMOTION_IDX['Anger']]
            p_disgust = avg_probs[self.EMOTION_IDX['Disgust']]
            p_fear = avg_probs[self.EMOTION_IDX['Fear']]
            p_happy = avg_probs[self.EMOTION_IDX['Happy']]
            p_neutral = avg_probs[self.EMOTION_IDX['Neutral']]
            p_sad = avg_probs[self.EMOTION_IDX['Sad']]
            p_surprise = avg_probs[self.EMOTION_IDX['Surprise']]

            # 4. 7 进 4 概率融合公式
            raw_understand = 1.0 * p_happy + 0.3 * p_surprise
            raw_doubt = 1.0 * p_anger + 1.0 * p_sad + 0.7 * p_surprise + 0.1 * p_fear
            raw_disgusted = 1.0 * p_disgust + 0.4 * p_anger + 0.3 * p_sad
            raw_neutral = 1.0 * p_neutral

            # 5. 归一化
            total_prob = raw_understand + raw_doubt + raw_disgusted + raw_neutral
            if total_prob > 0:
                p_u = raw_understand / total_prob
                p_d = raw_doubt / total_prob
                p_dis = raw_disgusted / total_prob
                p_n = raw_neutral / total_prob
            else:
                p_u, p_d, p_dis, p_n = 0, 0, 0, 1.0

            # 6. 取最大值作为当前微观窗口的基础离散状态
            states = ["Understand", "Doubt", "Disgusted", "Neutral"]
            fused_probs = [p_u, p_d, p_dis, p_n]
            max_idx = np.argmax(fused_probs)
            current_cognitive_state = states[max_idx]

            # --- 策略二：计算 8 秒 Perclos 疲劳度，并行使一票否决权 ---
            if len(self.perclos_buffer) > 0 and (now - self.perclos_buffer[0][0]) >= 7.5:   #冷启动保护，先累计满7.5秒再计算
                total_p_frames = len(self.perclos_buffer)
                blink_count = sum(1 for item in self.perclos_buffer if item[1])
                yawn_count = sum(1 for item in self.perclos_buffer if item[2])

                # Perclos 公式: 闭眼率 + 哈欠率 * 0.2
                perclos_val = (blink_count / total_p_frames) + (yawn_count / total_p_frames) * 0.2

                # 如果超过疲劳阈值，直接将第五状态覆写上去！
                if perclos_val > 0.38:
                    current_cognitive_state = "Fatigued"

            # 7. 将这个最终决断的状态存入 1 分钟的宏观窗口
            self.macro_buffer.append((now, current_cognitive_state))

        # ==============================================================
        # 宏观统计层 (物理时间 60 秒滑动窗口计算 Score)
        # ==============================================================
        # 1. 核心：剔除超过 60.0 秒的老数据
        while self.macro_buffer and (now - self.macro_buffer[0][0]) > 60.0:
            self.macro_buffer.popleft()

        if len(self.macro_buffer) > 0:
            # 2. 提取纯标签列表进行统计
            macro_list = [item[1] for item in self.macro_buffer]
            total_frames = len(macro_list)

            u_count = macro_list.count("Understand")    #理解状态
            d_count = macro_list.count("Doubt") #疑惑状态
            dis_count = macro_list.count("Disgusted")   #厌烦状态
            n_count = macro_list.count("Neutral")   #自然听课状态
            f_count = macro_list.count("Fatigued")  # 疲劳状态

            # 3. 动态 Score 公式 (引入 Fatigued 的 -0.5 重磅惩罚)
            raw_score = (1.0 * u_count + 0.9 * n_count + 0.7 * d_count + 0.1 * dis_count - 0.5 * f_count) / total_frames * 100

            # 因为存在负权重，确保分数不会掉到 0 以下，最高不超过 100
            score = max(0, min(100, int(raw_score)))

            status_text = f"Cognitive: {current_cognitive_state.upper()}"

            # 4. 基于认知的弹窗
            if (d_count / total_frames) > 0.4:
                if now - self.last_alert_time > 45:
                    alert_data = ('doubt', 'Learning Alert', '系统检测到您可能遇到知识难点，建议做好标记或暂停回顾。')
                    self.last_alert_time = now
            elif (f_count / total_frames) > 0.3:  # 新增疲劳弹窗
                if now - self.last_alert_time > 60:
                    alert_data = ('fatigue', 'Fatigue Alert', '系统检测到您当前较为疲劳，建议起身活动或喝口水休息一下。')
                    self.last_alert_time = now

        # ==============================================================
        # 阶段三：空间物理规则最高优先级覆写 (一票否决)
        # ==============================================================
        """
        if has_face and delta_pitch is not None:
            if delta_pitch < self.PITCH_DOWN_THRESH:
                self.head_down_count += 1
                status_text = "Focus: HEAD DOWN"
                score = 30

                if self.head_down_count > self.HEAD_DOWN_MAX_FRAMES:
                    if now - self.last_alert_time > 30:
                        alert_data = ('head_down', 'Alert', '长时间低头！如果是玩手机请专心，记笔记请注意颈椎休息。')
                        self.last_alert_time = now
            else:
                self.head_down_count = 0
        else:
            self.head_down_count = 0

        if not has_face:
            self.face_not_detected_count += 1
            status_text = "Status: ABSENT"
            score = 0

            if self.face_not_detected_count == 250:
                if now - self.last_alert_time > 30:
                    alert_data = ('absence', 'Alert', 'User Absent! Please return to seat.')
                    self.last_alert_time = now
        else:
            self.face_not_detected_count = 0
        """
        return score, status_text, alert_data