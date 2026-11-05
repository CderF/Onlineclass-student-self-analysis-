"""
#23 macOS「表示」「編集」菜单的专项验证（补充 tests/test_mac_menu.py、tests/test_mac_window.py）

运行：在仓库根目录执行（会弹出真实窗口，约 40 秒；显示器需保持常亮，可用 caffeinate -d）
    python test-scripts/verify_issue23_mac_view_edit_menu.py

按 `python main.py` 的真实启动流程运行子进程（框架见 verify_issue21_mac_menu.py），按键均为发给 NSApp 的真实按键事件。检查项：
  1. 原生「編集」「表示」菜单：项目名称、顺序、快捷键，都可以点击
  2. ⌘2 / ⌘1 与菜单项切换页面：当前页面与侧边导航的选中项一致；←（返回）随之可用
  3. 「サイドバーを表示/非表示」与点 ☰ 效果相同（展开 / 收起）
  4. 课程名输入框（真实的 CourseNameDialog）：⌘A / ⌘C / ⌘X / ⌘V / ⌘Z 与对应菜单项；粘贴日语
  5. 报告页：⌘A + ⌘C、「すべてを選択」+「コピー」都能复制页面文字
非 macOS 平台：跳过全部检查（退出码 0）。数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_issue21_mac_menu as harness  # noqa: E402

failures = []

EXPECTED_EDIT = [("取り消す", "z", "⌘"), ("カット", "x", "⌘"), ("コピー", "c", "⌘"), ("ペースト", "v", "⌘"),
                 ("すべてを選択", "a", "⌘")]
EXPECTED_VIEW = [("モニター", "1", "⌘"), ("分析レポート", "2", "⌘"), ("サイドバーを表示/非表示", "", "⌘")]

ACTIVATE = "AppKit.NSApp.activateIgnoringOtherApps_(True); w.raise_(); w.activateWindow(); QTest.qWait(1000)\n"

PAGE_STATE = """
nav = w.navigationInterface
def page_state():
    current = w.stackedWidget.currentWidget()
    selected = [name for name in ('monitor-interface', 'report-interface') if nav.widget(name).isSelected]
    return [current.objectName(), selected, w.navigationInterface.panel.returnButton.isEnabled()]
"""


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def main(db: Path):
    print("\n[1] 「編集」「表示」菜单")
    a = harness.App(db, "23A")
    check("程序正常启动", bool((a.state() or {}).get("ready")))
    a.run(ACTIVATE)
    for title, expected in (("編集", EXPECTED_EDIT), ("表示", EXPECTED_VIEW)):
        items = a.run(f"result = native_menu_items({title!r})")
        visible = [tuple(i[:3]) for i in items if not i[3]]
        check(f"「{title}」项目名称、顺序、快捷键", visible == expected, f"实际 {visible}")
        check(f"「{title}」所有项目都可以点击", all(i[4] for i in items if not i[3]), str(items))

    print("\n[2] 切换页面")
    r = a.run(PAGE_STATE + "result = [page_state()]\n"
              "send_key('2', 'cmd'); QTest.qWait(800); result.append(page_state())\n"
              "send_key('1', 'cmd'); QTest.qWait(800); result.append(page_state())\n"
              "click_menu_item('表示', '分析レポート'); QTest.qWait(800); result.append(page_state())\n"
              "click_menu_item('表示', 'モニター'); QTest.qWait(800); result.append(page_state())")
    start, by_key2, by_key1, by_menu2, by_menu1 = r
    check("初始为监控页", start[:2] == ["monitor-interface", ["monitor-interface"]], str(start))
    check("⌘2：报告页，侧边导航选中「分析レポート」，← 可用", by_key2 == ["report-interface", ["report-interface"], True],
          str(by_key2))
    check("⌘1：监控页，侧边导航选中「モニター」", by_key1[:2] == ["monitor-interface", ["monitor-interface"]], str(by_key1))
    check("菜单项「分析レポート」「モニター」同样切换，选中项一致",
          by_menu2[:2] == ["report-interface", ["report-interface"]]
          and by_menu1[:2] == ["monitor-interface", ["monitor-interface"]], f"{by_menu2} / {by_menu1}")
    r = a.run("w.navigationInterface.panel.returnButton.click(); QTest.qWait(600);"
              "result = w.stackedWidget.currentWidget().objectName()")
    check("用菜单切换后，← 回到上一页（与点击侧边导航的历史记录一致）", r == "report-interface", str(r))
    a.run("send_key('1', 'cmd'); QTest.qWait(500)")

    print("\n[3] 侧边栏")
    r = a.run("p = w.navigationInterface.panel; result = [p.width()]\n"
              "click_menu_item('表示', 'サイドバーを表示/非表示'); QTest.qWait(700); result.append(p.width())\n"
              "click_menu_item('表示', 'サイドバーを表示/非表示'); QTest.qWait(700); result.append(p.width())\n"
              "p.menuButton.click(); QTest.qWait(700); result.append(p.width())\n"
              "p.menuButton.click(); QTest.qWait(700); result.append(p.width())")
    check("菜单项展开 / 收起侧边栏，与点 ☰ 的结果相同", r[0] == 48 and r[1] > 100 and r[2] == 48 and r[3] == r[1] and r[4] == 48,
          f"宽度 {r}")

    print("\n[4] 课程名输入框")
    r = a.run(ACTIVATE + """
from app.view.course_name_dialog import CourseNameDialog
clip = app.clipboard()
d = CourseNameDialog(['線形代数 第3回'], w); d.show(); QTest.qWait(500)
box = d.nameBox; box.setFocus(); QTest.qWait(300)
focus_ok = app.focusWidget() is box
steps = []
def step(name):
    QTest.qWait(300); steps.append([name, box.text(), box.selectedText(), clip.text()])
box.insert('abc'); step('insert')
send_key('a', 'cmd'); step('⌘A')
send_key('c', 'cmd'); step('⌘C')
send_key('x', 'cmd'); step('⌘X')
send_key('v', 'cmd'); step('⌘V')
send_key('z', 'cmd'); step('⌘Z')
box.insert('xyz'); step('insert')
for item in ('すべてを選択', 'コピー', 'カット', 'ペースト', '取り消す'):
    click_menu_item('編集', item); step(item)
clip.setText('線形代数 第3回'); box.clear(); step('clear')
send_key('v', 'cmd'); step('⌘V 日语')
d.close(); d.deleteLater(); QTest.qWait(300)
result = [focus_ok, steps]
""")
    focus_ok, steps = r
    by = {s[0]: s[1:] for s in steps}
    check("输入框获得焦点", focus_ok)
    check("⌘A 全选", by["⌘A"][1] == "abc", str(by["⌘A"]))
    check("⌘C 复制", by["⌘C"][2] == "abc" and by["⌘C"][0] == "abc", str(by["⌘C"]))
    check("⌘X 剪切", by["⌘X"][0] == "" and by["⌘X"][2] == "abc", str(by["⌘X"]))
    check("⌘V 粘贴", by["⌘V"][0] == "abc", str(by["⌘V"]))
    check("⌘Z 取消上一步（粘贴）", by["⌘Z"][0] == "", str(by["⌘Z"]))
    check("菜单项：すべてを選択 / コピー / カット / ペースト / 取り消す",
          by["すべてを選択"][1] == "xyz" and by["コピー"][2] == "xyz" and by["カット"][0] == ""
          and by["ペースト"][0] == "xyz" and by["取り消す"][0] == "", str(steps[-7:-2]))
    check("粘贴日语课程名", by["⌘V 日语"][0] == "線形代数 第3回", str(by["⌘V 日语"]))

    print("\n[5] 报告页")
    r = a.run(ACTIVATE + """
from PyQt5.QtWebEngineWidgets import QWebEngineView
clip = app.clipboard()
send_key('2', 'cmd'); QTest.qWait(4000)
view = w.reportInterface.findChild(QWebEngineView)
view.setFocus(); QTest.qWait(500)
focus = type(app.focusWidget()).__name__
clip.setText(''); send_key('a', 'cmd'); QTest.qWait(500); send_key('c', 'cmd'); QTest.qWait(800)
by_key = clip.text()
clip.setText(''); view.setFocus(); QTest.qWait(300)
click_menu_item('編集', 'すべてを選択'); QTest.qWait(500); click_menu_item('編集', 'コピー'); QTest.qWait(800)
by_menu = clip.text()
result = [focus, by_key[:200], by_menu[:200]]
""")
    focus, by_key, by_menu = r
    check("⌘A + ⌘C 复制报告页文字", "授業ごと" in by_key and "記録の管理" in by_key, f"焦点 {focus}，剪贴板 {by_key[:60]!r}")
    check("「すべてを選択」+「コピー」复制报告页文字", "授業ごと" in by_menu and "記録の管理" in by_menu, repr(by_menu[:60]))

    a.run("w.close()")
    code = a.wait_exit()
    check("关闭后退出码 0", code == 0, str(code))


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("SKIP  只在 macOS 上运行（Windows 上没有菜单栏，由作者手动回归）")
        sys.exit(0)
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "data" / "self_analysis.db")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
