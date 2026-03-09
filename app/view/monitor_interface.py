from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QColor
from qfluentwidgets import (
    CardWidget, PrimaryPushButton, ProgressRing, SwitchButton,
    FluentIcon as FIF, TitleLabel, BodyLabel,
    setTheme, Theme, InfoBar, InfoBarPosition
)

from app.camera_thread import CameraThread


class MonitorInterface(QWidget):
    """
    实时监控界面
    负责展示 YOLO 实时检测画面、注意力分数更新以及异步弹窗警告。
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("monitor-interface")

        # --- Layouts ---
        self.vBoxLayout = QVBoxLayout(self)
        self.headerLayout = QHBoxLayout()
        self.contentLayout = QHBoxLayout()
        self.controlsLayout = QHBoxLayout()

        # --- UI Components ---

        # 1. Header
        self.titleLabel = TitleLabel("Real-time Attention Monitor", self)

        # 2. Main Display (Camera Feed)
        self.videoCard = CardWidget(self)
        self.videoLayout = QVBoxLayout(self.videoCard)
        self.videoLabel = QLabel("Camera Feed Offline", self.videoCard)
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setStyleSheet("background-color: #202020; color: #808080; border-radius: 8px;")
        self.videoLabel.setMinimumSize(640, 480)
        self.videoLayout.addWidget(self.videoLabel)

        # 3. Sidebar / Stats Panel
        self.statsPanel = QWidget(self)
        self.statsLayout = QVBoxLayout(self.statsPanel)

        # Attention Score Indicator
        self.scoreLabel = BodyLabel("Status: Waiting to start...", self.statsPanel)
        self.progressRing = ProgressRing(self.statsPanel)
        self.progressRing.setFixedSize(120, 120)
        self.progressRing.setTextVisible(True)
        self.progressRing.setValue(0)

        # ====== 新增：面部网格控制开关 ======
        self.meshSwitch = SwitchButton(parent=self.statsPanel)
        self.meshSwitch.setText("Draw Face Mesh")
        self.meshSwitch.setChecked(True) # 默认开启
        self.meshSwitch.checkedChanged.connect(self._on_mesh_switch_toggled)
        # ==================================

        # Controls
        self.startBtn = PrimaryPushButton(FIF.PLAY, "Start Monitor", self.statsPanel)
        self.stopBtn = PrimaryPushButton(FIF.PAUSE, "Stop Monitor", self.statsPanel)
        self.stopBtn.setEnabled(False)

        # Connect button signals
        self.startBtn.clicked.connect(self._on_start_clicked)
        self.stopBtn.clicked.connect(self._on_stop_clicked)

        # Add widgets to stats layout
        self.statsLayout.addWidget(self.scoreLabel, 0, Qt.AlignmentFlag.AlignCenter)
        self.statsLayout.addSpacing(10)
        self.statsLayout.addWidget(self.progressRing, 0, Qt.AlignmentFlag.AlignCenter)
        self.statsLayout.addSpacing(20)
        self.statsLayout.addWidget(self.meshSwitch, 0, Qt.AlignmentFlag.AlignCenter) # 将开关加入布局
        self.statsLayout.addStretch(1)
        self.statsLayout.addWidget(self.startBtn)
        self.statsLayout.addWidget(self.stopBtn)

        # --- Assembly ---
        self.headerLayout.addWidget(self.titleLabel)
        self.headerLayout.addStretch(1)

        self.contentLayout.addWidget(self.videoCard, 3)
        self.contentLayout.addWidget(self.statsPanel, 1)

        self.vBoxLayout.addLayout(self.headerLayout)
        self.vBoxLayout.addSpacing(20)
        self.vBoxLayout.addLayout(self.contentLayout)
        self.vBoxLayout.addStretch(1)

        self.vBoxLayout.setContentsMargins(30, 30, 30, 30)

    # --- UI Update Slots ---

    def set_camera_frame(self, pixmap: QPixmap):
        """更新摄像头视频流画面"""
        self.videoLabel.setPixmap(pixmap.scaled(
            self.videoLabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def update_attention_level(self, value: int):
        """仅负责更新圆环进度条的分数"""
        self.progressRing.setValue(value)

    def update_status_text(self, status_text: str):
        """
        接收后台发来的状态文本，并根据关键字动态改变文字颜色。
        """
        self.scoreLabel.setText(status_text)

        # --- 新增：识别校准状态并显示蓝色 ---
        if "CALIBRATING" in status_text:
            self.scoreLabel.setStyleSheet("color: #3B82F6; font-weight: bold;")  # 蓝色
        # ---------------------------------
        elif "Focus" in status_text or "Active" in status_text:
            self.scoreLabel.setStyleSheet("color: #10B981; font-weight: bold;")  # 绿色 (专注)
        elif "PHONE" in status_text or "HEAD DOWN" in status_text or "ABSENT" in status_text:
            self.scoreLabel.setStyleSheet("color: #EF4444; font-weight: bold;")  # 红色 (分心/离线)
        else:
            self.scoreLabel.setStyleSheet("color: #F59E0B; font-weight: bold;")  # 橙色 (监控中/未知)
    def show_async_alert(self, title: str, message: str):
        """
        接收后台的报警信号，弹出 qfluentwidgets 的 InfoBar 轻量级通知。
        不再使用阻塞式的 QMessageBox。
        """
        #触发系统默认提示音
        QApplication.beep()

        # 根据警报级别（标题）动态选择错误样式（红色）或警告样式（黄色）
        if "CRITICAL" in title.upper():
            InfoBar.error(
                title=title,
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000,  # 严重警告，停留 5 秒
                parent=self
            )
        else:
            InfoBar.warning(
                title=title,
                content=message,
                orient=Qt.Horizontal,
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,  # 普通警告，停留 3 秒
                parent=self
            )

# ====== 新增：开关响应事件 ======
    def _on_mesh_switch_toggled(self, is_checked: bool):
        """实时将开关状态传递给正在运行的线程"""
        # 变更为 camera_thread
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.set_draw_mesh(is_checked)
    # ==============================

    def _on_start_clicked(self):
        self.startBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.videoLabel.setText("正在启动摄像头并加载模型...")
        self.scoreLabel.setStyleSheet("color: #808080;")

        # 变更为 camera_thread
        self.camera_thread = CameraThread()

        self.camera_thread.set_draw_mesh(self.meshSwitch.isChecked())

        self.camera_thread.change_pixmap_signal.connect(self.set_camera_frame)
        self.camera_thread.update_score_signal.connect(self.update_attention_level)
        self.camera_thread.update_status_signal.connect(self.update_status_text)
        self.camera_thread.alert_signal.connect(self.show_async_alert)

        self.camera_thread.start()

    def _on_stop_clicked(self):
        self.startBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self.videoLabel.setText("Camera Feed Offline")
        self.update_attention_level(0)
        self.update_status_text("Status: Stopped")
        self.scoreLabel.setStyleSheet("color: #808080;")

        # 变更为 camera_thread
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()