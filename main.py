import sys
from typing import Optional

from PyQt5 import sip
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from qfluentwidgets import (
    NavigationItemPosition, FluentWindow, SubtitleLabel, setFont,
    Theme, setTheme, setFontFamilies
)
from qfluentwidgets import FluentIcon as FIF
# Import our custom interfaces
from app import session_store
from app.single_instance import SingleInstance
from app.view.monitor_interface import MonitorInterface
from app.view.report_interface import ReportInterface  # 同时导入 QtWebEngineWidgets，必须早于创建 QApplication


class MainWindow(FluentWindow):
    """
    Main Application Window containing the navigation sidebar.
    """

    def __init__(self):
        super().__init__()

        # Window Configuration
        self.setWindowTitle("学習エンゲージメント自己分析")
        self.resize(1100, 750)
        # 关闭时销毁（#14）：留到解释器退出时，由 PyQt 的清理在遍历包装对象的途中销毁主窗口，
        # 销毁过程中运行的 Python 代码（qfluentwidgets 的事件过滤器、destroyed 槽）会改动正在遍历的表，偶发 SIGSEGV
        self.setAttribute(Qt.WA_DeleteOnClose)

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

    def bring_to_front(self):
        """再次启动程序时调出已开的窗口（#13）：最小化的恢复，并放到最前面"""
        self.setWindowState((self.windowState() & ~Qt.WindowMinimized) | Qt.WindowActive)
        self.show()
        self._activate()
        # macOS：刚从最小化恢复时的激活会被忽略（实测立即与 0 ms 都无效、50 ms 起有效），稍后再激活一次
        QTimer.singleShot(200, self._activate)

    def _activate(self):
        self.raise_()
        self.activateWindow()

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

        # 报告页的后台读取 / 导出（#16）要在窗口销毁前等完：销毁时持有 GIL，后台任务无法继续，会互相等待
        self.reportInterface.shutdown()

        # 接受关闭事件，正常退出程序
        event.accept()


def start_instance() -> Optional[SingleInstance]:
    """
    启动时先拿单实例锁（#13），拿到后再把上次崩溃遗留的课程记录标为「未正常结束」。
    顺序不能反：否则第二个实例会把正在运行的实例正在记录的课程标为 interrupted。
    已有实例在运行时通知它调出窗口，返回 None。需在 QApplication 创建之后调用。
    """
    instance = SingleInstance()
    if not instance.acquire():
        if instance.notify_running():
            print("程序已在运行，已调出已打开的窗口")
        else:
            print("⚠️ 程序已在运行，但无法调出已打开的窗口")
        return None

    try:
        conn = session_store.connect()
        count = session_store.mark_interrupted_sessions(conn)
        conn.close()
        if count:
            print(f"⚠️ 已将 {count} 条未正常结束的课程记录标记为 interrupted")
    except Exception as e:
        print(f"❌ 打开课程记录数据库失败: {e}")
    return instance


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

    instance = start_instance()
    if instance is None:
        sys.exit(0)
    app.aboutToQuit.connect(instance.release)

    # 2. Create and show window
    w = MainWindow()
    instance.activated.connect(w.bring_to_front)
    w.show()

    code = app.exec()
    if not sip.isdeleted(w):   # 未经关闭窗口就退出（如 app.quit()）：同样先保存记录，再在解释器清理前销毁（#14）
        w.close()
        sip.delete(w)
    sys.exit(code)
