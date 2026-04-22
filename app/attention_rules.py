import time
import numpy as np
from collections import deque


class AttentionAnalyzer:
    def __init__(self):
        # 定义变量：空间姿态与物理规则参数
        self.last_known_pitch = 0
        self.PITCH_DOWN_THRESH = -15.0  #低头判定角
        self.HEAD_DOWN_MAX_SECONDS = 15.0  # 超过 15 秒触发弹窗
        self.FACE_NOT_DETECTED_MAX_SECONDS = 120.0 # 超过120秒触发弹窗
        self.head_down_start_time = None  # 用来记录开始低头的那一瞬间的时间戳
        self.score_before_head_down = 100
        self.face_not_detected_time = None #用来记录离开作为的那一瞬间的时间戳
        self.score_before_absent = 100
        self.last_alert_time = 0
        self.eyes_closed_start_time = None  # 用于记录连续闭眼的起始时间

        #阶段一：7进4 表情标签转换为专注度标签 and Perclos升格为第5个专注度标签
        # 多模态融合时间戳滑动窗口（Yolo、Perclos）
        self.micro_buffer = deque()  # 3 秒微观表情概率: (now, all_probs_clean)
        self.perclos_buffer = deque()  # 8 秒疲劳特征缓冲: (now, is_blink, is_yawn)
        self.macro_buffer = deque()  # 60 秒宏观离散标签: (now, state_label)

        # YOLO 类别索引映射（具体标签待确认）
        self.EMOTION_IDX = {
            'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
            'Neutral': 4, 'Sad': 5, 'Surprise': 6
        }

    def process_frame(self, all_probs, ear, mar, delta_pitch, delta_yaw):
        """
        接收 YOLO 分类概率、MediaPipe 特征以及 3D 头部相对姿态进行综合判断
        """
        alert_data = None
        status_text = "Status: Initializing..."
        score = 100
        now = time.time()
        has_face = (ear is not None and mar is not None)

        current_cognitive_state = "Neutral"  # 默认状态

        # 表情标签前向拦截 + 微观概率融合 + 疲劳升格
        if has_face and all_probs is not None and len(all_probs) == 7:

            # 前向特征拦截：防止因为打哈欠或是闭眼了导致yolo误判表情，同时将误判的表情标签概率分配给Neutral(有些过于绝对了)
            is_yawn = (mar > 0.65)
            is_blink_frame = (ear < 0.15)
            # 连续闭眼 2 秒计时器
            if is_blink_frame:
                if self.eyes_closed_start_time is None:
                    self.eyes_closed_start_time = now  # 刚闭上的瞬间，记录时间
                closed_duration = now - self.eyes_closed_start_time
            else:
                self.eyes_closed_start_time = None  # 一旦睁眼，计时器立刻清零
                closed_duration = 0.0
            # 只有闭眼2秒以上才会被认为在闭眼
            is_blink = (closed_duration > 2.0)

            probs_clean = np.copy(all_probs)

            # 如果正在打哈欠或闭眼，强行剥夺 Happy 和 Surprise 的概率，防止 YOLO 误判
            if is_yawn or is_blink:
                probs_clean = np.zeros(7) # 7个概率全变
                probs_clean[self.EMOTION_IDX['Neutral']] = 1.0  # 唯独 Neutral 设为 1.0


            # 存入带时间戳的数据包裹
            self.micro_buffer.append((now, probs_clean))
            self.perclos_buffer.append((now, is_blink_frame, is_yawn))

            # 核心：perclos_buffer 剔除超过8秒的老数据
            while self.micro_buffer and (now - self.micro_buffer[0][0]) > 3.0:
                self.micro_buffer.popleft()
            while self.perclos_buffer and (now - self.perclos_buffer[0][0]) > 8.0:
                self.perclos_buffer.popleft()

            # 提取纯概率数组进行平均计算
            probs_only = [item[1] for item in self.micro_buffer]
            avg_probs = np.mean(probs_only, axis=0)

            p_anger = avg_probs[self.EMOTION_IDX['Anger']]
            p_disgust = avg_probs[self.EMOTION_IDX['Disgust']]
            p_fear = avg_probs[self.EMOTION_IDX['Fear']]
            p_happy = avg_probs[self.EMOTION_IDX['Happy']]
            p_neutral = avg_probs[self.EMOTION_IDX['Neutral']]
            p_sad = avg_probs[self.EMOTION_IDX['Sad']]
            p_surprise = avg_probs[self.EMOTION_IDX['Surprise']]

            # ~~7 进 4 概率融合公式（目前参数定义只有模糊的心理学研究依据，具体落实到数字的计算机领域的研究依据则几乎没有，需要在yolo上做些更改）~~
            # 彻底舍弃7进4概率融合公式，改为使用Transfer Learning迁移学习
            """
            # 激进风格：敏锐地捕捉每一丝异常情绪，但对yolo的检测结果精确度要求高，同时与后续的归一化操作强绑定
            raw_understand = 1.0 * p_happy + 0.3 * p_surprise
            raw_doubt = 1.0 * p_anger + 1.0 * p_sad + 0.7 * p_surprise + 0.1 * p_fear
            raw_disgusted = 1.0 * p_disgust + 0.4 * p_anger + 0.3 * p_sad
            raw_neutral = 1.0 * p_neutral
            """
            # 平衡风格：除了fear外左右表情概率系数均一致
            raw_understand = 1.0 * p_happy + 0.3 * p_surprise
            raw_doubt = 0.8 * p_anger + 0.5 * p_sad + 0.7 * p_surprise + 0.1 * p_fear
            raw_disgusted = 1.0 * p_disgust + 0.2 * p_anger + 0.5 * p_sad + 0.1 * p_fear
            raw_neutral = 1.0 * p_neutral

            # 归一化
            total_prob = raw_understand + raw_doubt + raw_disgusted + raw_neutral
            if total_prob > 0:
                p_u = raw_understand / total_prob
                p_d = raw_doubt / total_prob
                p_dis = raw_disgusted / total_prob
                p_n = raw_neutral / total_prob
            else:
                p_u, p_d, p_dis, p_n = 0, 0, 0, 1.0

            # 取最大值作为当前微观窗口的基础离散状态
            states = ["Understand", "Doubt", "Disgusted", "Neutral"]
            fused_probs = [p_u, p_d, p_dis, p_n]
            max_idx = np.argmax(fused_probs)
            current_cognitive_state = states[max_idx]

            # 计算 8 秒 Perclos 疲劳度，并行使一票否决权
            if len(self.perclos_buffer) > 0 and (now - self.perclos_buffer[0][0]) >= 7.5:   #冷启动保护，先累计满7.5秒再计算
                total_p_frames = len(self.perclos_buffer)
                blink_count = sum(1 for item in self.perclos_buffer if item[1])
                yawn_count = sum(1 for item in self.perclos_buffer if item[2])

                # Perclos 公式: 闭眼率 + 哈欠率 * 0.2
                perclos_val = (blink_count / total_p_frames) + (yawn_count / total_p_frames) * 0.2

                # 如果超过疲劳阈值，直接将第五状态覆写上去！（直接覆写有点草率，后续可以改为Fatigue的概率占比逐渐增加）
                if perclos_val > 0.38:
                    current_cognitive_state = "Fatigued"

            # 将这个最终决断的状态存入 1 分钟的宏观窗口
            self.macro_buffer.append((now, current_cognitive_state))

        # 阶段二：60秒滑块窗口计算Score
        # 核心：剔除超过 60.0 秒的老数据
        while self.macro_buffer and (now - self.macro_buffer[0][0]) > 60.0:
            self.macro_buffer.popleft()

        if len(self.macro_buffer) > 0:
            # 提取纯标签列表进行统计
            macro_list = [item[1] for item in self.macro_buffer]
            total_frames = len(macro_list)

            u_count = macro_list.count("Understand")    #理解状态
            d_count = macro_list.count("Doubt") #疑惑状态
            dis_count = macro_list.count("Disgusted")   #厌烦状态
            n_count = macro_list.count("Neutral")   #自然听课状态
            f_count = macro_list.count("Fatigued")  # 疲劳状态

            # 动态 Score 公式 (引入 Fatigued 的 -0.5 重磅惩罚)
            raw_score = (1.0 * u_count + 0.9 * n_count + 0.7 * d_count + 0.1 * dis_count - 0.5 * f_count) / total_frames * 100

            # 因为存在负权重，确保分数不会掉到 0 以下，最高不超过 100
            score = max(0, min(100, int(raw_score)))

            status_text = f"Cognitive: {current_cognitive_state.upper()}"
            # 目前权重还没有替换成优化后的版本因此无法根据实际情况进行测试以便发现哪些弹窗有助于提高用户体验
            """
            # 基于专注度水平的弹窗
            if (d_count / total_frames) > 0.4 | (dis_count / total_frames) > 0.5:
                if now - self.last_alert_time > 45:
                    alert_data = ('doubt', 'Learning Alert', '系统检测到您可能遇到知识难点，建议做好标记或暂停回顾。')
                    self.last_alert_time = now
            elif (f_count / total_frames) > 0.5:  # 新增疲劳弹窗
                if now - self.last_alert_time > 60:
                    alert_data = ('fatigue', 'Fatigue Alert', '系统检测到您当前较为疲劳，建议起身活动或喝口水休息一下。')
                    self.last_alert_time = now
            """

        # 阶段三：空间物理规则最高优先级覆写 (一票否决)

            # 低头判定逻辑
            if has_face and delta_pitch is not None:
                if delta_pitch < self.PITCH_DOWN_THRESH:
                    self.last_known_pitch = delta_pitch
                    # 刚刚低头的瞬间：按下秒表，并赶紧给当前分数拍个照存档！
                    if self.head_down_start_time is None:
                        self.head_down_start_time = now
                        self.score_before_head_down = score

                    # 算出此刻已经连续低头了多少秒
                    down_duration = now - self.head_down_start_time

                    # 业务逻辑：长时低头前根据之前的专注度分数来判断本次行为的可能性
                    # 补充：目前为简单逻辑的测试阶段，后续可以尝试加入概率计算公式（长期离席判别同理）
                    if down_duration > self.HEAD_DOWN_MAX_SECONDS:
                        # 查阅 15 秒前的历史成绩
                        if self.score_before_head_down >= 60:
                            status_text = "Focus: TAKING NOTES"
                        else:
                            status_text = "Focus: DISTRACTED"
                            # 我们用一个时间差的二次方（或乘以系数）作为惩罚！
                            overtime = down_duration - self.HEAD_DOWN_MAX_SECONDS
                            penalty = int((overtime ** 2) * 0.2)
                            score = max(0, score - penalty)  # 强制覆盖宏观分数，且不低于0

                            # 每 30 秒响一次
                            if now - self.last_alert_time > 30.0:
                                alert_data = ('head_down', 'Learning Alert', '检测到长时间开小差，如需休息请先暂停。')
                                self.last_alert_time = now
                else:
                    # 一旦抬头，秒表清零
                    self.head_down_start_time = None
            else:
                self.head_down_start_time = None

            # 离开座位与极度低头盲区判定逻辑
            if not has_face:

                # 刚消失的瞬间：按下秒表，给当前分数 低头状态拍照
                if self.face_not_detected_time is None:
                    self.face_not_detected_time = now
                    self.score_before_absent = score

                absent_duration = now - self.face_not_detected_time

                #在丢失面部时，前回顾之前有没有处于低头状态，如果有那么给5秒的惯性冗余
                if self.last_known_pitch < self.PITCH_DOWN_THRESH:
                    if absent_duration <= 5.0:
                        status_text = "Focus: TAKING NOTES (Head Down)"
                        score = self.score_before_absent
                    # 数据插补 虽然看不到表情，但强行给宏观大脑塞入 Neutral 标签
                        self.macro_buffer.append((now, "Neutral"))
                    else:
                        # 超过5秒之后 判断之前的分数 执行长时低头的判断逻辑
                        if score >= 60:
                            status_text = "Focus: TAKING NOTES (Head Down)"
                            self.macro_buffer.append((now, "Neutral"))
                        else:
                            status_text = "Status: ABNORMAL (Hidden)"
                            # 我们用一个时间差的二次方（或乘以系数）作为惩罚！
                            overtime = absent_duration - 5.0
                            penalty = int((overtime ** 2) * 0.2)
                            score = max(0, score - penalty)  # 强制覆盖宏观分数，且不低于0

                            # 每 30 秒响一次
                            if now - self.last_alert_time > 30.0:
                                alert_data = ('hidden', 'Alert', '检测到长时间面部丢失，请调整姿态或返回座位！')
                                self.last_alert_time = now

                # 业务逻辑：两分钟内非线性平滑降分
                else:
                    if absent_duration <= self.FACE_NOT_DETECTED_MAX_SECONDS:
                        status_text = "Status: AWAY (Short)"
                        # 1. 计算时间流逝的基础比例 (0.0 到 1.0)
                        base_ratio = absent_duration / self.FACE_NOT_DETECTED_MAX_SECONDS
                        # 2. 套用三次幂曲线，制造“先缓后急”
                        decay_ratio = base_ratio ** 3
                        # 3. 计算并执行惩罚
                        current_penalty = self.score_before_absent * decay_ratio
                        score = max(0, int(self.score_before_absent - current_penalty))
                    else:
                        status_text = "Status: ABSENT"
                        score = 0
                        if now - self.last_alert_time > 30.0:
                            alert_data = ('absence', 'Alert', 'Student Absent! Please return to seat.')
                            self.last_alert_time = now
            else:
                # 脸一回来，秒表清零
                self.face_not_detected_time = None
        return score, status_text, alert_data