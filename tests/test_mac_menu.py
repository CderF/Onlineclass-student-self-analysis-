"""macOS 菜单栏模块（#21）：用替身窗口操作目标测菜单结构与触发。原生菜单见 test-scripts/verify_issue21_mac_menu.py"""
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction

import main
from app import app_info, mac_menu


class StubTarget:
    def __init__(self):
        self.calls = []

    def show_about(self):
        self.calls.append("show_about")

    def quit_app(self):
        self.calls.append("quit_app")


@pytest.fixture
def mac(monkeypatch):
    monkeypatch.setattr(mac_menu.sys, "platform", "darwin")


@pytest.fixture
def menu_bar(qapp, mac):
    target = StubTarget()
    bar = mac_menu.build_menu_bar(target)
    yield bar, target
    bar.deleteLater()


def _actions(bar):
    return [a for menu_action in bar.actions() for a in menu_action.menu().actions() if not a.isSeparator()]


def _by_role(bar, role):
    return [a for a in _actions(bar) if a.menuRole() == role]


def test_about_item_opens_about(menu_bar):
    bar, target = menu_bar
    [about] = _by_role(bar, QAction.AboutRole)
    assert about.text() == "Self-Analysisについて"
    about.trigger()
    assert target.calls == ["show_about"]


def test_quit_item_is_command_q_and_quits_through_target(menu_bar):
    bar, target = menu_bar
    [quit_action] = _by_role(bar, QAction.QuitRole)
    assert quit_action.text() == "Self-Analysisを終了"
    assert quit_action.shortcut() == QKeySequence("Ctrl+Q")   # macOS 上 Ctrl 即 ⌘
    quit_action.trigger()
    assert target.calls == ["quit_app"]


def test_no_settings_or_about_qt_and_no_guessed_roles(menu_bar):
    bar, _ = menu_bar
    roles = [a.menuRole() for a in _actions(bar)]
    assert QAction.PreferencesRole not in roles
    assert QAction.AboutQtRole not in roles
    # 默认的 TextHeuristicRole 会按文字把菜单项移进应用菜单，每一项都必须明确指定角色
    assert QAction.TextHeuristicRole not in roles


def test_nothing_is_created_outside_macos(qapp, monkeypatch):
    monkeypatch.setattr(mac_menu.sys, "platform", "win32")
    assert mac_menu.build_menu_bar(StubTarget()) is None
    assert mac_menu.install_translator(qapp) is None
    assert mac_menu.set_app_name() is False
    assert mac_menu.show_about_panel() is False


def test_app_menu_translator_only_translates_app_menu(qapp, mac):
    translator = mac_menu.install_translator(qapp)
    try:
        tr = QCoreApplication.translate
        expected = {"Services": "サービス", "Hide %1": "%1を隠す", "Hide Others": "ほかを隠す",
                    "Show All": "すべてを表示", "Quit %1": "%1を終了", "About %1": "%1について"}
        for source, text in expected.items():
            assert tr("MAC_APPLICATION_MENU", source) == text
        assert tr("MAC_APPLICATION_MENU", "Hide %1").replace("%1", app_info.APP_NAME) == "Self-Analysisを隠す"
        assert tr("QLineEdit", "&Copy") == "&Copy"          # 其他上下文不受影响
        assert tr("MonitorInterface", "開始") == "開始"
    finally:
        qapp.removeTranslator(translator)


def test_about_panel_shows_name_version_and_note_from_one_place(monkeypatch):
    monkeypatch.setattr(app_info, "APP_VERSION", "9.8.7")
    options = mac_menu.about_panel_options()
    assert options["ApplicationName"] == "Self-Analysis"
    assert "バージョン 9.8.7" in options["Credits"]
    assert app_info.ABOUT_TEXT in options["Credits"]
    assert options["Version"] == "" and options["ApplicationVersion"] == ""   # 不显示英文的 Version 行和 Python 版本


def test_version_is_0_1_0():
    assert app_info.APP_VERSION == "0.1.0"


def test_main_window_quit_goes_through_close(monkeypatch):
    """⌘Q 与红色按钮一样调用 close()，由 closeEvent 保存课程记录"""
    closed = []
    main.MainWindow.quit_app(SimpleNamespace(close=lambda: closed.append(True)))
    assert closed == [True]
    shown = []
    monkeypatch.setattr(mac_menu, "show_about_panel", lambda: shown.append(True))
    main.MainWindow.show_about(SimpleNamespace())
    assert shown == [True]
