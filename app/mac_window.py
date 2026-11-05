"""
macOS 窗口的原生操作（#22、#24）：缩放，以及双击标题栏按系统设置执行的操作。
"""
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QWidget

ZOOM = "zoom"
MINIMIZE = "minimize"
NOTHING = "none"


def ns_window(widget: QWidget):
    import objc
    return objc.objc_object(c_void_p=int(widget.winId())).window()


def zoom(widget: QWidget) -> None:
    """拡大/縮小：与按住 Option 点绿色按钮相同（NSWindow zoom:）"""
    ns_window(widget).zoom_(None)


def double_click_action(action: Optional[str], legacy_minimize: bool = False) -> str:
    """
    「系统设置 → 桌面与程序坞 → 连按窗口标题栏以」的取值（全局域 AppleActionOnDoubleClick）→ 要执行的操作。
    Maximize（缩放）/ Minimize（最小化）/ None（不执行任何操作）；
    未设置时沿用旧版设置 AppleMiniaturizeOnDoubleClick（为真时最小化），否则按系统默认缩放。
    未知的取值（如以后的系统新增的选项）也按缩放处理。
    """
    if action == "Minimize":
        return MINIMIZE
    if action == "None":
        return NOTHING
    if action is None and legacy_minimize:
        return MINIMIZE
    return ZOOM


def read_double_click_preference() -> tuple:
    """读取当前的系统设置；每次双击时读取，设置改了立即生效"""
    from Foundation import NSUserDefaults
    defaults = NSUserDefaults.standardUserDefaults()
    action = defaults.stringForKey_("AppleActionOnDoubleClick")
    return (str(action) if action is not None else None), bool(defaults.boolForKey_("AppleMiniaturizeOnDoubleClick"))


class TitleBarDoubleClickFilter(QObject):
    """
    接管标题栏的双击（库固定为最大化 / 还原），交给 callback 按系统设置处理。
    标题栏中的按钮（← ☰）沿用 QWidget 默认的按下处理，不接受双击事件，事件会传到标题栏；
    双击落在可用的按钮上时不执行标题栏操作（与 macOS 工具栏按钮一致），按钮不可用时等同于双击标题栏。
    """

    def __init__(self, parent: QObject, callback, buttons=()):
        super().__init__(parent)
        self._callback = callback
        self._buttons = list(buttons)

    def eventFilter(self, obj, e):
        if e.type() == QEvent.MouseButtonDblClick and e.button() == Qt.LeftButton:
            on_button = any(b.isVisible() and b.isEnabled() and b.geometry().contains(e.pos()) for b in self._buttons)
            if not on_button:
                self._callback()
            return True
        return False
