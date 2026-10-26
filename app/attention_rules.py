import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AnalysisResult:
    """单帧分析结果。界面与记录模块都从这里取数据，不要解析 status_text。"""
    score: int  # 参与度分数 0-100
    status_text: str  # 仅用于界面显示（日语），颜色由 spatial_state 决定，不要解析文字
    alert: Optional[Tuple[str, str, str]]  # (类型, 标题, 内容)，无警报时为 None
    has_face: bool
    cognitive_state: Optional[str]  # Understand / Doubt / Disgust / Neutral；本帧无表情结果时为 None
    fatigue_index: float  # 疲劳指数 0-1
    spatial_state: str  # normal / head_down / taking_notes / distracted / hidden_notes / hidden_abnormal / away / absent
    delta_pitch: Optional[float]
    delta_yaw: Optional[float]


# 认知状态在界面上的日语显示名（数据中仍保存英文代码）
COGNITIVE_STATE_LABELS_JA = {
    "Understand": "理解", "Doubt": "疑問", "Neutral": "中立", "Disgust": "嫌悪",
}


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
        self.perclos_val = 0.0

        """
            yolo类别索引待更改
        """
        # 多模态融合时间戳滑动窗口（Yolo、MediaPipe疲劳检测）
        self.micro_buffer = deque()  # 3 秒微观表情概率: (now, all_probs_clean) 需更改
        self.perclos_buffer = deque()  # 8 秒疲劳特征缓冲: (now, ear, mar, delta_pitch)
        self.macro_buffer = deque()  # 60 秒宏观离散标签: (now, state_label)

        # YOLO 类别索引映射
        self.EMOTION_IDX = {
            'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
            'Neutral': 4, 'Sad': 5, 'Surprise': 6
        }
    # 虽然利用MediaPipe进行了眨眼 张嘴的检测但是正确率仍不可忽视 需要通过实验来验证
    # 两个不同的人 分别戴眼镜和不戴眼镜 共四种情况 进行眨眼 闭眼 张嘴模拟说话 张嘴模拟打哈欠
    def _calculate_perclos(self) -> float:
        """计算眼睛闭合程度超过80% (EAR < 0.2) 的帧所占的百分比"""
        valid_frames = [item for item in self.perclos_buffer if item[1] is not None]
        if not valid_frames:
            return 0.0
        closed_frames = sum(1 for item in valid_frames if item[1] < 0.2)
        return closed_frames / len(valid_frames)

    def _calculate_blink_freq(self) -> float:
        """计算眨眼频率 (EAR < 0.2 持续 200 毫秒视为一次眨眼，频率为特定动作帧数占所有帧的百分比) （实验判断）"""
        if not self.perclos_buffer:
            return 0.0

        runs = []
        current_run = []
        for item in self.perclos_buffer:
            t, ear, mar, dp = item
            if ear is not None and ear < 0.2:
                if current_run:
                    if t - current_run[-1][0] > 0.25:  # 250ms gap max within a run
                        runs.append(current_run)
                        current_run = []
                current_run.append((t, ear))
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []
        if current_run:
            runs.append(current_run)

        blink_frames_count = 0
        for run in runs:
            duration = run[-1][0] - run[0][0]
            if duration >= 0.2:  # >= 200 ms
                blink_frames_count += len(run)

        return blink_frames_count / len(self.perclos_buffer)

    def _calculate_yawn_freq(self) -> float:
        """计算打哈欠频率 (MAR > 0.65 持续 2 秒视为打了一次哈欠，频率为特定动作帧数占所有帧的百分比)"""
        if not self.perclos_buffer:
            return 0.0

        runs = []
        current_run = []
        for item in self.perclos_buffer:
            t, ear, mar, dp = item
            if mar is not None and mar > 0.65:
                if current_run:
                    if t - current_run[-1][0] > 0.25:  # 250ms gap max within a run
                        runs.append(current_run)
                        current_run = []
                current_run.append((t, mar))
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []
        if current_run:
            runs.append(current_run)

        yawn_frames_count = 0
        for run in runs:
            duration = run[-1][0] - run[0][0]
            if duration >= 2.0:  # >= 2.0 seconds
                yawn_frames_count += len(run)

        return yawn_frames_count / len(self.perclos_buffer)

    def _calculate_nodding_freq(self) -> float:
        """计算困倦性点头频率 (每次低头 [delta_pitch < -15] 的间隔不超过 1.5 秒视为困倦性点头)"""
        if not self.perclos_buffer:
            return 0.0

        runs = []
        current_run = []
        for item in self.perclos_buffer:
            t, ear, mar, dp = item
            if dp is not None and dp < self.PITCH_DOWN_THRESH:
                if current_run:
                    if t - current_run[-1][0] > 0.25:  # 250ms gap max within the same head down
                        runs.append(current_run)
                        current_run = []
                current_run.append((t, dp))
            else:
                if current_run:
                    runs.append(current_run)
                    current_run = []
        if current_run:
            runs.append(current_run)

        num_runs = len(runs)
        if num_runs == 0:
            return 0.0

        is_nodding_run = [False] * num_runs
        for i in range(num_runs):
            if i > 0:
                gap_prev = runs[i][0][0] - runs[i-1][-1][0]
                if gap_prev <= 1.5:
                    is_nodding_run[i] = True
                    is_nodding_run[i-1] = True
            if i < num_runs - 1:
                gap_next = runs[i+1][0][0] - runs[i][-1][0]
                if gap_next <= 1.5:
                    is_nodding_run[i] = True
                    is_nodding_run[i+1] = True

        nodding_frames_count = 0
        for i in range(num_runs):
            if is_nodding_run[i]:
                nodding_frames_count += len(runs[i])

        return nodding_frames_count / len(self.perclos_buffer)

    def process_frame(self, all_probs, ear, mar, delta_pitch, delta_yaw) -> AnalysisResult:
        """
        接收 YOLO 分类概率、MediaPipe 特征以及 3D 头部相对姿态进行综合判断
        """
        alert_data = None
        status_text = "初期化中…"
        score = 100
        now = time.time()
        has_face = (ear is not None and mar is not None)

        current_cognitive_state = None  # 本帧没有表情结果时保持 None，不当作 Neutral 记录
        spatial_state = "normal"

        # 表情标签前向拦截（可删除） + 迁移学习 + 疲劳惩罚
        if has_face and all_probs is not None and len(all_probs) == 7:

            # 前向特征拦截：防止因为打哈欠或是闭眼了导致yolo误判表情，同时将误判的表情标签概率分配给Neutral
            is_yawn = (mar > 0.65)
            is_blink_frame = (ear < 0.2)
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
                probs_clean[self.EMOTION_IDX['Neutral']] = 1.0

            # 存入带时间戳的数据包裹
            self.micro_buffer.append((now, probs_clean))
            self.perclos_buffer.append((now, ear, mar, delta_pitch))

            # 核心：剔除超过指定时间的老数据
            while self.micro_buffer and (now - self.micro_buffer[0][0]) > 3.0:
                self.micro_buffer.popleft()
            while self.perclos_buffer and (now - self.perclos_buffer[0][0]) > 8.0:
                self.perclos_buffer.popleft()

            """
            以下部分待修改，彻底舍弃7进4概率融合公式，改为使用Transfer Learning迁移学习
            """
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
            states = ["Understand", "Doubt", "Disgust", "Neutral"]
            fused_probs = [p_u, p_d, p_dis, p_n]
            max_idx = np.argmax(fused_probs)
            current_cognitive_state = states[max_idx]

            # 计算疲劳度 (Fatigue Index)
            """
                最终实现：疲劳指数=0.35*perclos+0.25*眨眼频率+0.2*打哈欠频率+0.2*困倦性点头频率
            """

            if len(self.perclos_buffer) > 0 and (now - self.perclos_buffer[0][0]) >= 7.5:   #冷启动保护，先累计满7.5秒再计算
                perclos = self._calculate_perclos()
                blink_freq = self._calculate_blink_freq()
                yawn_freq = self._calculate_yawn_freq()
                nodding_freq = self._calculate_nodding_freq()

                # 疲劳指数 = 0.35 * perclos + 0.25 * 眨眼频率 + 0.2 * 打哈欠频率 + 0.2 * 困倦性点头频率
                raw_fatigue = 0.35 * perclos + 0.25 * blink_freq + 0.2 * yawn_freq + 0.2 * nodding_freq
                self.perclos_val = min(1.0, max(0.0, raw_fatigue))

            # 将这个最终决断的状态存入60秒的宏观窗口
            self.macro_buffer.append((now, current_cognitive_state))

        # 阶段二：60秒滑块窗口计算Score
        """
            核心参与度分数计算公式暂定为：
                Score_Engagement = 0.56 * (1 - Score_Fatigue) + 0.44 * Score_emotion
                
                参考文献；Multimodal Detection of Emotional and Cognitive States in E-Learning Through Deep Fusion of Visual and Textual Data with NLP

        """
        # 核心：剔除超过 60.0 秒的老数据 为什么是60秒
        while self.macro_buffer and (now - self.macro_buffer[0][0]) > 60.0:
            self.macro_buffer.popleft()

        if len(self.macro_buffer) > 0:
            # 提取纯标签列表进行统计
            macro_list = [item[1] for item in self.macro_buffer]
            total_frames = len(macro_list)

            u_count = macro_list.count("Understand")    #理解状态
            d_count = macro_list.count("Doubt") #疑惑状态
            dis_count = macro_list.count("Disgust")   #厌烦状态
            n_count = macro_list.count("Neutral")   #自然听课状态

            # 新策略
            perclos_score = (1.0 - self.perclos_val) * 100.0
            emotion_score = (1.0 * u_count + 0.5 * n_count + 0.7 * d_count + 0.1 * dis_count) / total_frames * 100.0
            raw_score = 0.5625 * perclos_score + 0.4375 * emotion_score

            # 保证分数范围在0-100之间
            score = max(0, min(100, int(raw_score)))

            if current_cognitive_state is not None:
                status_text = f"認知状態：{COGNITIVE_STATE_LABELS_JA[current_cognitive_state]}"

            # 警报逻辑：基于 self.perclos_val 直接触发疲劳警报（后续需更改）
            if self.perclos_val > 0.38:
                if now - self.last_alert_time > 60.0:
                    alert_data = ('fatigue', '疲労アラート', '疲れが見られます。少し体を動かすか、水を飲んで休憩しましょう。')
                    self.last_alert_time = now

        # 阶段三：空间物理规则最高优先级覆写　（有必要考虑是否有必要添加此功能）

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
                spatial_state = "head_down"
                if down_duration > self.HEAD_DOWN_MAX_SECONDS:
                    # 查阅 15 秒前的历史成绩
                    if self.score_before_head_down >= 60:
                        status_text = "集中：ノート記録中"
                        spatial_state = "taking_notes"
                    else:
                        status_text = "注意散漫"
                        spatial_state = "distracted"
                        # 我们用一个时间差的二次方（或乘以系数）作为惩罚！
                        overtime = down_duration - self.HEAD_DOWN_MAX_SECONDS
                        penalty = int((overtime ** 2) * 0.2)
                        score = max(0, score - penalty)  # 强制覆盖宏观分数，且不低于0

                        # 每 30 秒响一次
                        if now - self.last_alert_time > 30.0:
                            alert_data = ('head_down', '学習アラート', '長時間集中が途切れています。休憩する場合は先に一時停止してください。')
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
                    status_text = "集中：ノート記録中（下向き）"
                    spatial_state = "hidden_notes"
                    score = self.score_before_absent
                # 数据插补 虽然看不到表情，但强行给宏观大脑塞入 Neutral 标签
                    self.macro_buffer.append((now, "Neutral"))
                else:
                    # 超过5秒之后 判断之前的分数 执行长时低头的判断逻辑
                    # 以丢失瞬间的分数为准：无脸时宏观窗口不再更新，60 秒后清空会让当前分数回落为默认 100
                    if self.score_before_absent >= 60:
                        status_text = "集中：ノート記録中（下向き）"
                        spatial_state = "hidden_notes"
                        self.macro_buffer.append((now, "Neutral"))
                    else:
                        status_text = "異常：顔が長時間検出されません"
                        spatial_state = "hidden_abnormal"
                        # 我们用一个时间差的二次方（或乘以系数）作为惩罚！
                        overtime = absent_duration - 5.0
                        penalty = int((overtime ** 2) * 0.2)
                        score = max(0, self.score_before_absent - penalty)  # 强制覆盖宏观分数，且不低于0

                        # 每 30 秒响一次
                        if now - self.last_alert_time > 30.0:
                            alert_data = ('hidden', 'アラート', '顔が長時間検出されません。姿勢を整えるか、席に戻ってください。')
                            self.last_alert_time = now

            # 业务逻辑：两分钟内非线性平滑降分
            else:
                if absent_duration <= self.FACE_NOT_DETECTED_MAX_SECONDS:
                    status_text = "離席中"
                    spatial_state = "away"
                    # 1. 计算时间流逝的基础比例 (0.0 到 1.0)
                    base_ratio = absent_duration / self.FACE_NOT_DETECTED_MAX_SECONDS
                    # 2. 套用三次幂曲线，制造“先缓后急”
                    decay_ratio = base_ratio ** 3
                    # 3. 计算并执行惩罚
                    current_penalty = self.score_before_absent * decay_ratio
                    score = max(0, int(self.score_before_absent - current_penalty))
                else:
                    status_text = "不在"
                    spatial_state = "absent"
                    score = 0
                    if now - self.last_alert_time > 30.0:
                        alert_data = ('absence', 'アラート', '離席が続いています。席に戻ってください。')
                        self.last_alert_time = now
        else:
            # 脸一回来，秒表清零
            self.face_not_detected_time = None
        return AnalysisResult(
            score=score,
            status_text=status_text,
            alert=alert_data,
            has_face=has_face,
            cognitive_state=current_cognitive_state,
            fatigue_index=self.perclos_val,
            spatial_state=spatial_state,
            delta_pitch=delta_pitch,
            delta_yaw=delta_yaw,
        )