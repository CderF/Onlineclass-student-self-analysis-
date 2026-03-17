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

        # ==============================================================
        # 2. 彻底基于“物理时间戳”的滑动窗口
        # ==============================================================
        # 不再依赖 FPS，队列里存放的是元组: (timestamp, data)
        self.micro_buffer = deque()  # 存放 3 秒内的微观概率: (now, all_probs)
        self.macro_buffer = deque()  # 存放 60 秒内的宏观标签: (now, state_label)

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
        has_face = (ear is not None)

        current_cognitive_state = "Neutral"  # 默认状态

        # ==============================================================
        # 微观平滑层 (物理时间 9 秒滑动窗口)
        #   计算过去9秒内的根据Yolo输出的 7 个表情标签抽象融合而成的 4 个专注度标签
        #   每秒出一个过去9秒内的平均值
        # ==============================================================
        if has_face and all_probs is not None and len(all_probs) == 7:
            # 1. 存入带时间戳的数据包裹
            self.micro_buffer.append((now, all_probs))

            # 2. 核心：剔除超过 4.0 秒的老数据
            while self.micro_buffer and (now - self.micro_buffer[0][0]) > 4.0:
                self.micro_buffer.popleft()

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

            # 6. 取最大值作为当前微观窗口的唯一离散状态
            states = ["Understand", "Doubt", "Disgusted", "Neutral"]
            fused_probs = [p_u, p_d, p_dis, p_n]
            max_idx = np.argmax(fused_probs)
            current_cognitive_state = states[max_idx]

            # 7. 将这个稳定下来的状态存入 1 分钟的宏观窗口
            self.macro_buffer.append((now, current_cognitive_state))

        # ==============================================================
        # 宏观统计层 (物理时间 60 秒滑动窗口计算 Score)
        #   队列中存放过去60秒内上一阶段输出的标签，每过一帧计算一次Score
        #   每一帧检测出来一个标签
        #   每次计算Score都是将队列内中各个标签套入公式进行计算的
        # ==============================================================
        # 1. 核心：剔除超过 60.0 秒的老数据
        while self.macro_buffer and (now - self.macro_buffer[0][0]) > 60.0:
            self.macro_buffer.popleft()

        if len(self.macro_buffer) > 0:
            # 2. 提取纯标签列表进行统计
            macro_list = [item[1] for item in self.macro_buffer]
            total_frames = len(macro_list)

            u_count = macro_list.count("Understand")
            d_count = macro_list.count("Doubt")
            dis_count = macro_list.count("Disgusted")
            n_count = macro_list.count("Neutral")

            # 3. 动态 Score 公式 (最高不超过 100)
            raw_score = (1.0 * u_count + 0.9 * n_count + 0.7 * d_count + 0.1 * dis_count) / total_frames * 100
            score = min(100, int(raw_score))

            status_text = f"Cognitive: {current_cognitive_state.upper()}"

            # 4. 基于认知的弹窗
            if (d_count / total_frames) > 0.4:
                if now - self.last_alert_time > 45:
                    alert_data = ('doubt', 'Learning Alert', '系统检测到您可能遇到知识难点，建议做好标记或暂停回顾。')
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