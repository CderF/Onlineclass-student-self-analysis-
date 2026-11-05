"""macOS 窗口适配（#19）中能在 offscreen 下测的部分。真实窗口见 test-scripts/verify_issue2[0-4]_*.py"""
import sys
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QSize

import main
from app import mac_window


def _stub(fullscreen: bool):
    names = ("MAC_SYSTEM_BUTTONS_X", "MAC_SYSTEM_BUTTONS_WIDTH")
    return SimpleNamespace(isFullScreen=lambda: fullscreen, **{n: getattr(main.MainWindow, n) for n in names})


@pytest.mark.parametrize("width", [600, 1100, 2560])
@pytest.mark.parametrize("fullscreen", [False, True])
def test_mac_system_buttons_stay_at_left_whatever_the_width(monkeypatch, width, fullscreen):
    monkeypatch.setattr(main, "IS_MAC", True)
    rect = main.MainWindow.systemTitleBarRect(_stub(fullscreen), QSize(width, 28))
    assert rect.left() >= 0
    assert rect.right() < 100          # 红黄绿在左上角，与窗口宽度无关
    assert rect.y() == (0 if fullscreen else 8)


def test_mac_system_buttons_end_before_title_row_buttons(monkeypatch):
    """红黄绿区域的右边界不超过标题栏中为它留出的宽度，← 才不会与绿色按钮重叠"""
    monkeypatch.setattr(main, "IS_MAC", True)
    rect = main.MainWindow.systemTitleBarRect(_stub(False), QSize(1100, 28))
    reserved = main.MainWindow.MAC_SYSTEM_BUTTONS_X + main.MainWindow.MAC_SYSTEM_BUTTONS_WIDTH
    assert rect.x() + rect.width() <= reserved


# ---- #22 / #23 菜单触发的窗口操作，#24 双击标题栏 ----


class StubWindow(SimpleNamespace):
    """MainWindow 的替身：记录调用的 Qt 窗口操作"""

    def __init__(self, fullscreen=False):
        calls = []
        super().__init__(
            calls=calls, monitorInterface="monitor", reportInterface="report",
            isFullScreen=lambda: fullscreen,
            switchTo=lambda w: calls.append(f"switchTo:{w}"),
            navigationInterface=SimpleNamespace(toggle=lambda: calls.append("nav.toggle")),
            showMinimized=lambda: calls.append("showMinimized"),
            showNormal=lambda: calls.append("showNormal"),
            showFullScreen=lambda: calls.append("showFullScreen"),
            close=lambda: calls.append("close"),
        )
        self.zoom_window = lambda: main.MainWindow.zoom_window(self)
        self.minimize_window = lambda: main.MainWindow.minimize_window(self)


def test_view_menu_operations_match_navigation_clicks():
    w = StubWindow()
    main.MainWindow.show_monitor(w)
    main.MainWindow.show_report(w)
    main.MainWindow.toggle_sidebar(w)
    # 点侧边导航项调用的也是 switchTo，点 ☰ 调用的也是 toggle
    assert w.calls == ["switchTo:monitor", "switchTo:report", "nav.toggle"]


def test_window_menu_operations(monkeypatch):
    zoomed = []
    monkeypatch.setattr(mac_window, "zoom", zoomed.append)
    w = StubWindow()
    main.MainWindow.minimize_window(w)
    main.MainWindow.toggle_full_screen(w)
    main.MainWindow.close_window(w)
    main.MainWindow.zoom_window(w)
    assert w.calls == ["showMinimized", "showFullScreen", "close"] and zoomed == [w]

    full = StubWindow(fullscreen=True)
    main.MainWindow.toggle_full_screen(full)
    main.MainWindow.zoom_window(full)          # 全屏时缩放没有意义
    assert full.calls == ["showNormal"] and zoomed == [w]


@pytest.mark.parametrize("action, legacy, expected", [
    ("Maximize", False, mac_window.ZOOM),
    ("Minimize", False, mac_window.MINIMIZE),
    ("None", False, mac_window.NOTHING),
    (None, False, mac_window.ZOOM),          # 未设置：系统默认为缩放
    (None, True, mac_window.MINIMIZE),       # 未设置但旧版设置为「最小化」
    ("Minimize", True, mac_window.MINIMIZE),
    ("Maximize", True, mac_window.ZOOM),     # 新设置优先
    ("Fill", False, mac_window.ZOOM),        # 未知取值按缩放处理
])
def test_double_click_action_follows_system_setting(action, legacy, expected):
    assert mac_window.double_click_action(action, legacy) == expected


@pytest.mark.parametrize("pref, fullscreen, expected", [
    (("Maximize", False), False, ["zoom"]),
    (("Minimize", False), False, ["showMinimized"]),
    (("None", False), False, []),
    (("Maximize", False), True, []),          # 全屏时双击标题栏没有效果
    (("Minimize", False), True, []),
])
def test_title_double_click_runs_the_configured_action(monkeypatch, pref, fullscreen, expected):
    w = StubWindow(fullscreen)
    monkeypatch.setattr(mac_window, "read_double_click_preference", lambda: pref)
    monkeypatch.setattr(mac_window, "zoom", lambda _: w.calls.append("zoom"))
    main.MainWindow.on_title_double_click(w)
    assert w.calls == expected


@pytest.mark.skipif(sys.platform != "darwin", reason="读取 macOS 系统设置")
def test_read_double_click_preference_returns_setting_and_legacy_flag():
    action, legacy = mac_window.read_double_click_preference()
    assert action is None or isinstance(action, str)
    assert isinstance(legacy, bool)


def test_title_bar_filter_takes_over_left_double_click_only(qapp):
    from PyQt5.QtCore import QEvent, QPoint, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication, QWidget
    bar = QWidget()
    fired = []
    bar.installEventFilter(mac_window.TitleBarDoubleClickFilter(bar, lambda: fired.append(True)))

    def send(kind, button):
        QApplication.sendEvent(bar, QMouseEvent(kind, QPoint(5, 5), button, button, Qt.NoModifier))

    send(QEvent.MouseButtonDblClick, Qt.RightButton)
    send(QEvent.MouseButtonPress, Qt.LeftButton)
    assert fired == []
    send(QEvent.MouseButtonDblClick, Qt.LeftButton)
    assert fired == [True]
    bar.deleteLater()


def test_title_bar_filter_ignores_double_click_on_enabled_buttons(qapp):
    """双击可用的 ← ☰ 不执行标题栏操作；按钮不可用时等同于双击标题栏"""
    from PyQt5.QtCore import QEvent, QPoint, Qt
    from PyQt5.QtGui import QMouseEvent
    from PyQt5.QtWidgets import QApplication, QWidget
    bar = QWidget()
    bar.resize(400, 48)
    button = QWidget(bar)
    button.setGeometry(80, 6, 40, 36)
    bar.show()
    fired = []
    bar.installEventFilter(mac_window.TitleBarDoubleClickFilter(bar, lambda: fired.append(True), [button]))

    def dbl(x):
        QApplication.sendEvent(bar, QMouseEvent(QEvent.MouseButtonDblClick, QPoint(x, 20), Qt.LeftButton,
                                                Qt.LeftButton, Qt.NoModifier))

    dbl(100)
    assert fired == []
    dbl(300)
    assert fired == [True]
    button.setEnabled(False)
    dbl(100)
    assert fired == [True, True]
    bar.deleteLater()
