import sys
from PyQt5.QtCore import Qt  # 新增导入 Qt，用于设置高分屏
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from qfluentwidgets import (
    NavigationItemPosition, FluentWindow, SubtitleLabel, setFont,
    Theme, setTheme
)
from qfluentwidgets import FluentIcon as FIF

# Import our custom interfaces
from app.view.monitor_interface import MonitorInterface
from app.view.report_interface import ReportInterface


class MainWindow(FluentWindow):
    """
    Main Application Window containing the navigation sidebar.
    """

    def __init__(self):
        super().__init__()

        # 1. Window Configuration
        self.setWindowTitle("YOLO Attention Monitor")
        self.resize(1100, 750)

        # 2. Create Sub-Interfaces
        self.monitorInterface = MonitorInterface(self)
        self.reportInterface = ReportInterface(self)

        # 3. Initialize Navigation
        self.initNavigation()

        # 4. Optional: Set initial theme
        # setTheme(Theme.DARK)

    def initNavigation(self):
        """
        Setup the sidebar navigation.
        """
        # Add Monitor Page (Home)
        self.addSubInterface(
            self.monitorInterface,
            FIF.CAMERA,
            "Monitor",
            position=NavigationItemPosition.TOP
        )

        # Add Report Page
        self.addSubInterface(
            self.reportInterface,
            FIF.PIE_SINGLE,
            "Analysis",
            position=NavigationItemPosition.TOP
        )

    # ================= 新增：安全退出拦截 =================
    def closeEvent(self, event):
        """
        重写窗口关闭事件。
        确保在用户点击右上角关闭程序时，后台摄像头线程被安全释放。
        """
        # 检查监控界面是否实例化了线程，并且线程还在运行中
        if hasattr(self.monitorInterface, 'thread') and self.monitorInterface.thread.isRunning():
            print("正在安全释放摄像头资源...")
            self.monitorInterface.thread.stop()
            self.monitorInterface.thread.wait()  # 等待线程彻底结束

        # 接受关闭事件，正常退出程序
        event.accept()
    # ======================================================


if __name__ == "__main__":
    # 1. Enable High DPI scaling (强烈建议开启，让 qfluentwidgets 组件更清晰)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # 兼容低版本的 PyQt5

    app = QApplication(sys.argv)

    app.setFont(QFont("PingFang SC", 10))

    # 2. Create and show window
    w = MainWindow()
    w.show()

    sys.exit(app.exec())