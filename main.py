import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from qfluentwidgets import (
    NavigationItemPosition, FluentWindow, SubtitleLabel, setFont,
    Theme, setTheme, setFontFamilies
)
from qfluentwidgets import FluentIcon as FIF
# Import our custom interfaces
from app import session_store
from app.view.monitor_interface import MonitorInterface
from app.view.report_interface import ReportInterface


class MainWindow(FluentWindow):
    """
    Main Application Window containing the navigation sidebar.
    """

    def __init__(self):
        super().__init__()

        # Window Configuration
        self.setWindowTitle("学習エンゲージメント自己分析")
        self.resize(1100, 750)

        # Create Sub-Interfaces
        self.monitorInterface = MonitorInterface(self)
        self.reportInterface = ReportInterface(self)

        # Initialize Navigation
        self.initNavigation()

        # Optional: Set initial theme
        # setTheme(Theme.DARK)

    def initNavigation(self):
        """
        Setup the sidebar navigation.
        """
        # Add Monitor Page (Home)
        self.addSubInterface(
            self.monitorInterface,
            FIF.CAMERA,
            "モニター",
            position=NavigationItemPosition.TOP
        )

        # Add Report Page
        self.addSubInterface(
            self.reportInterface,
            FIF.PIE_SINGLE,
            "分析レポート",
            position=NavigationItemPosition.TOP
        )

    # 安全退出拦截
    def closeEvent(self, event):
        """
        重写窗口关闭事件。
        确保在用户点击右上角关闭程序时，后台摄像头线程被安全释放。
        """
        # 暂停中（线程未运行）关闭窗口也要调用 stop()，课程记录才会被标记为正常结束
        if hasattr(self.monitorInterface, 'camera_thread'):
            print("正在安全释放摄像头资源...")
            self.monitorInterface.camera_thread.stop()  # 内部会等待线程彻底结束

        # 接受关闭事件，正常退出程序
        event.accept()


if __name__ == "__main__":
    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except AttributeError:
        pass  # 兼容低版本的 PyQt5

    app = QApplication(sys.argv)

    # 界面为日语：PingFang SC 会把日语汉字显示成中文字形，改用 Hiragino Sans
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])  # qfluentwidgets 组件不跟随 app 字体，需单独设置

    # 上次运行崩溃遗留的课程记录标为「未正常结束」
    try:
        conn = session_store.connect()
        count = session_store.mark_interrupted_sessions(conn)
        conn.close()
        if count:
            print(f"⚠️ 已将 {count} 条未正常结束的课程记录标记为 interrupted")
    except Exception as e:
        print(f"❌ 打开课程记录数据库失败: {e}")

    # 2. Create and show window
    w = MainWindow()
    w.show()

    sys.exit(app.exec())