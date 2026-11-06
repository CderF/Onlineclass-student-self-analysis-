import sys
from typing import Optional

from PyQt5 import sip
from PyQt5.QtCore import Qt, QTimer, QRect, QSize
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon, QFont
from qfluentwidgets import (
    NavigationItemPosition, FluentWindow, SubtitleLabel, setFont,
    Theme, setTheme, setFontFamilies
)
from qfluentwidgets import FluentIcon as FIF
# Import our custom interfaces
from app import mac_menu, mac_window, session_store
from app.single_instance import SingleInstance
from app.view.monitor_interface import MonitorInterface
from app.view.report_interface import ReportInterface  # 同时导入 QtWebEngineWidgets，必须早于创建 QApplication

IS_MAC = sys.platform == "darwin"


class MainWindow(FluentWindow):
    """
    Main Application Window containing the navigation sidebar.
    """

    # macOS（#20）：红黄绿在左上角，← ☰ 紧接其右、同一行，侧边导航从第二行开始
    MAC_SYSTEM_BUTTONS_X = 8        # 红黄绿区域的左边界
    MAC_SYSTEM_BUTTONS_WIDTH = 75   # 红黄绿区域宽度（与库相同）
    MAC_NAV_TOP = 48                # 侧边导航第一项的起始高度 = 标题栏高度
    MAC_TITLE_ROW_BOTTOM_INSET = 6  # 标题栏一行的内容（← ☰ 标题）上移，与红黄绿垂直居中对齐

    def __init__(self):
        super().__init__()

        # Window Configuration
        self.setWindowTitle("学習エンゲージメント自己分析")
        self.resize(1100, 750)
        # 关闭时销毁（#14）：留到解释器退出时，由 PyQt 的清理在遍历包装对象的途中销毁主窗口，
        # 销毁过程中运行的 Python 代码（qfluentwidgets 的事件过滤器、destroyed 槽）会改动正在遍历的表，偶发 SIGSEGV
        self.setAttribute(Qt.WA_DeleteOnClose)
        # Windows 11：库默认开启 Mica（窗口背景透明，由系统绘制）；窗口中有 QWebEngineView 时系统不再绘制，
        # 透明部分显示为黑色。改用库的不透明浅色背景（只在 Win11 上有作用）
        self.setMicaEffectEnabled(False)

        # Create Sub-Interfaces
        self.monitorInterface = MonitorInterface(self)
        self.reportInterface = ReportInterface(self)

        # Initialize Navigation
        self.initNavigation()
        if IS_MAC:
            self._init_mac_title_row()
            self.mac_menu_bar = mac_menu.build_menu_bar(self, self)

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

    def _init_mac_title_row(self):
        """把导航栏顶部的 ← ☰ 移进标题栏、放在红黄绿右侧；导航项让出标题栏这一行"""
        panel = self.navigationInterface.panel
        title_layout = self.titleBar.hBoxLayout
        title_layout.insertSpacing(0, self.MAC_SYSTEM_BUTTONS_X + self.MAC_SYSTEM_BUTTONS_WIDTH)
        for i, button in enumerate((panel.returnButton, panel.menuButton)):
            panel.topLayout.removeWidget(button)
            title_layout.insertWidget(1 + i, button, 0, Qt.AlignVCenter)
            button.show()  # 换父控件后需重新显示
        title_layout.insertSpacing(3, 8)  # ☰ 与窗口标题之间
        # 红黄绿的中心约在 y=21，且不能再往下移（会被 28px 高的系统标题栏裁掉），这一行改为对齐它
        title_layout.setContentsMargins(0, 0, 0, self.MAC_TITLE_ROW_BOTTOM_INSET)
        # 双击标题栏按系统设置执行（#24）；库固定为最大化 / 还原。双击可用的 ← ☰ 不触发
        self.titleBar.setDoubleClickEnabled(False)
        self.titleBar.installEventFilter(mac_window.TitleBarDoubleClickFilter(
            self.titleBar, self.on_title_double_click, (panel.returnButton, panel.menuButton)))
        margins = panel.vBoxLayout.contentsMargins()
        panel.vBoxLayout.setContentsMargins(margins.left(), self.MAC_NAV_TOP, margins.right(), margins.bottom())

    def systemTitleBarRect(self, size: QSize) -> QRect:
        """红黄绿的摆放区域（只在 macOS 上被调用）。库默认放在窗口右侧"""
        if not IS_MAC:
            return super().systemTitleBarRect(size)
        return QRect(self.MAC_SYSTEM_BUTTONS_X, 0 if self.isFullScreen() else 8,
                     self.MAC_SYSTEM_BUTTONS_WIDTH, size.height())

    def resizeEvent(self, e):
        if not IS_MAC:
            return super().resizeEvent(e)
        # 库把标题栏放在导航栏右侧（x=46）；mac 上标题栏占满整行，← ☰ 在其中
        self.titleBar.move(0, 0)
        self.titleBar.resize(self.width(), self.titleBar.height())

    def show_about(self):
        mac_menu.show_about_panel()

    def quit_app(self):
        """⌘Q：与红色按钮一样走 closeEvent，课程记录照常保存；最后一个窗口关闭后程序退出"""
        self.close()

    # ---- macOS 菜单与标题栏触发的窗口操作（mac_menu.MenuTarget）----

    def edit(self, command: str):
        mac_menu.apply_edit(QApplication.focusWidget(), command)

    def show_monitor(self):
        self.switchTo(self.monitorInterface)   # 与点击侧边导航相同

    def show_report(self):
        self.switchTo(self.reportInterface)

    def toggle_sidebar(self):
        self.navigationInterface.toggle()      # 与点击 ☰ 相同

    def minimize_window(self):
        self.showMinimized()

    def zoom_window(self):
        if not self.isFullScreen():
            mac_window.zoom(self)

    def toggle_full_screen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def close_window(self):
        """⌘W：只有这一个窗口，与红色按钮一样关闭并退出"""
        self.close()

    def on_title_double_click(self):
        if self.isFullScreen():
            return  # 全屏时双击标题栏没有效果（与系统一致）
        action = mac_window.double_click_action(*mac_window.read_double_click_preference())
        if action == mac_window.ZOOM:
            self.zoom_window()
        elif action == mac_window.MINIMIZE:
            self.minimize_window()

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

    mac_menu.set_app_name()  # 菜单栏中的程序名，必须早于创建 QApplication（#21）
    app = QApplication(sys.argv)
    mac_menu.install_translator(app)

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
