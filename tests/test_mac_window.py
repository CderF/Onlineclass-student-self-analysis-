"""macOS 窗口适配（#19）中能在 offscreen 下测的部分。真实窗口见 test-scripts/verify_issue20_mac_title_row.py"""
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QSize

import main


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
