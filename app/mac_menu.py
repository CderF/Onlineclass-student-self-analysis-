"""
macOS 菜单栏（#21）。只在 darwin 上生效，其他平台上各函数什么也不做。

- 菜单栏中加粗的程序名：用 `python main.py` 启动时为 python。`set_app_name` 在创建 QApplication 之前
  改写 bundle 信息中的程序名（非官方做法，失败时只打印日志）。
- 应用菜单中的「サービス」「隠す」「ほかを隠す」「すべてを表示」由 Qt 自动生成，文字来自
  翻译上下文 MAC_APPLICATION_MENU；`AppMenuTranslator` 只翻译这一上下文。
- 「について」「終了」是 `build_menu_bar` 创建的带 menu role 的 QAction，Qt 把它们放进应用菜单。
  没有 PreferencesRole / AboutQtRole 的 action，「設定…」「About Qt」就保持隐藏。
  其他菜单项一律设为 NoRole，否则 Qt 会按文字猜测角色，把它们移进应用菜单。
"""
import sys
from typing import Optional, Protocol

from PyQt5.QtCore import QCoreApplication, QTranslator
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction, QMenuBar, QWidget

from app import app_info


class MenuTarget(Protocol):
    """菜单项触发的窗口操作（由 MainWindow 实现，测试时用替身）"""

    def show_about(self) -> None: ...

    def quit_app(self) -> None: ...


def is_mac() -> bool:
    return sys.platform == "darwin"


# Qt 自动生成的应用菜单项的源文字 → macOS 日语版的标准写法（%1 为程序名）
APP_MENU_TEXTS = {
    "Services": "サービス",
    "Hide %1": "%1を隠す",
    "Hide Others": "ほかを隠す",
    "Show All": "すべてを表示",
    "Preferences...": "設定…",
    "Quit %1": "%1を終了",
    "About %1": "%1について",
    "About Qt": "Qtについて",
}


class AppMenuTranslator(QTranslator):
    """只翻译应用菜单（MAC_APPLICATION_MENU）的文字；其他上下文返回 None（不翻译）"""

    CONTEXT = "MAC_APPLICATION_MENU"

    def translate(self, context, source_text, disambiguation=None, n=-1):
        if context == self.CONTEXT:
            return APP_MENU_TEXTS.get(source_text)
        return None


def set_app_name(name: str = None) -> bool:
    """把菜单栏中加粗的程序名改为 name。必须在创建 QApplication 之前调用"""
    if not is_mac():
        return False
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = name or app_info.APP_NAME
        return True
    except Exception as e:
        print(f"⚠️ 无法修改菜单栏中的程序名: {e}")
        return False


def install_translator(app: QCoreApplication) -> Optional[AppMenuTranslator]:
    if not is_mac():
        return None
    translator = AppMenuTranslator(app)
    app.installTranslator(translator)
    return translator


def build_menu_bar(target: MenuTarget, parent: Optional[QWidget] = None) -> Optional[QMenuBar]:
    """构建菜单栏；非 macOS 返回 None。parent 为主窗口，随主窗口一起销毁（#14）"""
    if not is_mac():
        return None
    bar = QMenuBar(parent)
    # 只放带 role 的 action 时，Qt 会把它们移进应用菜单并隐藏这个空菜单
    app_menu = bar.addMenu(app_info.APP_NAME)

    about = QAction(f"{app_info.APP_NAME}について", bar)
    about.setMenuRole(QAction.AboutRole)
    about.triggered.connect(target.show_about)
    app_menu.addAction(about)

    quit_action = QAction(f"{app_info.APP_NAME}を終了", bar)
    quit_action.setMenuRole(QAction.QuitRole)
    quit_action.setShortcut(QKeySequence("Ctrl+Q"))   # macOS 上 Ctrl 即 ⌘
    quit_action.triggered.connect(target.quit_app)
    app_menu.addAction(quit_action)
    return bar


def about_panel_options() -> dict:
    """macOS 标准关于面板的内容。
    面板自带的版本行按程序的本地化显示为英文「Version …」（python 没有日语本地化），
    所以版本号写在说明文字里，自带的版本行留空"""
    return {
        "ApplicationName": app_info.APP_NAME,
        "ApplicationVersion": "",
        "Version": "",
        "Credits": f"バージョン {app_info.APP_VERSION}\n{app_info.ABOUT_TEXT}",
    }


def show_about_panel() -> bool:
    if not is_mac():
        return False
    try:
        import AppKit
        options = dict(about_panel_options())
        paragraph = AppKit.NSMutableParagraphStyle.alloc().init()
        paragraph.setAlignment_(AppKit.NSTextAlignmentCenter)
        attributes = {AppKit.NSParagraphStyleAttributeName: paragraph,
                      AppKit.NSFontAttributeName: AppKit.NSFont.systemFontOfSize_(11),
                      AppKit.NSForegroundColorAttributeName: AppKit.NSColor.labelColor()}
        options["Credits"] = AppKit.NSAttributedString.alloc().initWithString_attributes_(options["Credits"], attributes)
        # 未打包成 .app，系统取到的程序图标是文件夹；改用系统通用的程序图标
        icon = AppKit.NSWorkspace.sharedWorkspace().iconForFileType_("com.apple.application-bundle")
        icon.setSize_((128, 128))
        options["ApplicationIcon"] = icon
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        AppKit.NSApp.orderFrontStandardAboutPanelWithOptions_(options)
        return True
    except Exception as e:
        print(f"⚠️ 无法打开关于面板: {e}")
        return False
