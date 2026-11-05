"""macOS 菜单栏模块（#21–#23）：用替身窗口操作目标测菜单结构与触发。原生菜单见 test-scripts/verify_issue21_mac_menu.py"""
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAction

import main
from app import app_info, mac_menu


class StubTarget:
    """替身窗口操作目标：记录被调用的操作名（edit 记为 edit:<命令>）"""

    OPERATIONS = ("show_about", "quit_app", "show_monitor", "show_report", "toggle_sidebar",
                  "minimize_window", "zoom_window", "toggle_full_screen", "close_window")

    def __init__(self):
        self.calls = []
        for name in self.OPERATIONS:   # 与 MainWindow 的方法一样不带参数
            setattr(self, name, self._recorder(name))

    def _recorder(self, name):
        return lambda: self.calls.append(name)

    def edit(self, command):
        self.calls.append(f"edit:{command}")


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


# ---- #22 ウインドウ / #23 表示・編集 ----

def _menu(bar, title):
    [menu] = [a.menu() for a in bar.actions() if a.text() == title]
    return [a for a in menu.actions() if not a.isSeparator()]


def test_menu_order_follows_macos_convention(menu_bar):
    bar, _ = menu_bar
    assert [a.text() for a in bar.actions()] == ["Self-Analysis", "編集", "表示", "ウインドウ"]


@pytest.mark.parametrize("title, expected", [
    ("編集", [("取り消す", "Ctrl+Z", "edit:undo"), ("カット", "Ctrl+X", "edit:cut"), ("コピー", "Ctrl+C", "edit:copy"),
             ("ペースト", "Ctrl+V", "edit:paste"), ("すべてを選択", "Ctrl+A", "edit:selectAll")]),
    ("表示", [("モニター", "Ctrl+1", "show_monitor"), ("分析レポート", "Ctrl+2", "show_report"),
             ("サイドバーを表示/非表示", "", "toggle_sidebar")]),
    ("ウインドウ", [("しまう", "Ctrl+M", "minimize_window"), ("拡大/縮小", "", "zoom_window"),
                 ("フルスクリーンにする", "Meta+Ctrl+F", "toggle_full_screen"), ("閉じる", "Ctrl+W", "close_window")]),
])
def test_menu_items_names_shortcuts_and_targets(menu_bar, title, expected):
    """macOS 上 Qt 的 Ctrl 即 ⌘、Meta 即 ⌃：⌘Z…⌘A、⌘1 / ⌘2、⌘M、⌃⌘F、⌘W"""
    bar, target = menu_bar
    actions = _menu(bar, title)
    assert [(a.text(), a.shortcut().toString()) for a in actions] == [(t, k) for t, k, _ in expected]
    for action in actions:
        assert action.menuRole() == QAction.NoRole
        action.trigger()
    assert target.calls == [c for _, _, c in expected]


def test_apply_edit_on_line_edit(qapp):
    from PyQt5.QtWidgets import QLineEdit
    clipboard = qapp.clipboard()
    edit = QLineEdit()
    edit.insert("abc")
    assert mac_menu.apply_edit(edit, "selectAll") and edit.selectedText() == "abc"
    assert mac_menu.apply_edit(edit, "copy") and clipboard.text() == "abc"
    assert mac_menu.apply_edit(edit, "cut") and edit.text() == ""
    assert mac_menu.apply_edit(edit, "paste") and edit.text() == "abc"
    assert mac_menu.apply_edit(edit, "undo") and edit.text() == ""
    edit.deleteLater()


def test_apply_edit_on_report_page_goes_to_web_page(qapp):
    """报告页获得焦点的是 QWebEngineView 内部的子控件：沿父控件找到网页视图，交给网页执行"""
    from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineView
    from PyQt5.QtWidgets import QWidget
    view = QWebEngineView()
    inner = QWidget(view)
    triggered = []
    view.page = lambda: SimpleNamespace(triggerAction=triggered.append)
    for command in ("undo", "cut", "copy", "paste", "selectAll"):
        assert mac_menu.apply_edit(inner, command)
    assert triggered == [QWebEnginePage.Undo, QWebEnginePage.Cut, QWebEnginePage.Copy, QWebEnginePage.Paste,
                         QWebEnginePage.SelectAll]
    view.deleteLater()


def test_apply_edit_without_editable_focus_does_nothing(qapp):
    from PyQt5.QtWidgets import QPushButton
    button = QPushButton()
    assert mac_menu.apply_edit(None, "copy") is False
    assert mac_menu.apply_edit(button, "selectAll") is False
    button.deleteLater()
