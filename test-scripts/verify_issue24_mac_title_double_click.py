"""
#24 macOS 双击标题栏按系统设置执行的专项验证（补充 tests/test_mac_window.py）

运行：在仓库根目录执行（会弹出真实窗口、进入一次全屏，约 40 秒；显示器需保持常亮，可用 caffeinate -d）
    python test-scripts/verify_issue24_mac_title_double_click.py

按 `python main.py` 的真实启动流程运行子进程（框架见 verify_issue21_mac_menu.py），
向窗口投递真实的原生双击事件（NSEvent，clickCount = 2），并替换读取系统设置的函数来注入不同的取值。检查项：
  1. 不注入：按本机当前的系统设置执行
  2. 「拡大/縮小」（Maximize）：双击标题栏空白处、窗口标题文字处都会缩放，再双击还原
  3. 「しまう」（Minimize）：最小化；「何もしない」（None）：窗口不变
  4. 双击 ← / ☰ 不触发标题栏操作；单击 ← / ☰ 仍然执行原来的操作
  5. 全屏时双击标题栏没有效果
非 macOS 平台：跳过全部检查（退出码 0）。数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_issue21_mac_menu as harness  # noqa: E402

failures = []

SETUP = """
from app import mac_window
AppKit.NSApp.activateIgnoringOtherApps_(True); w.raise_(); w.activateWindow(); QTest.qWait(1000)
ns = mac_window.ns_window(w)
real_read = mac_window.read_double_click_preference
injected = {}
mac_window.read_double_click_preference = lambda: injected['pref'] if 'pref' in injected else real_read()
def state():
    return {'zoomed': bool(ns.isZoomed()), 'width': ns.frame().size.width, 'minimized': w.isMinimized(),
            'fullscreen': w.isFullScreen()}
def dbl(x, y, wait=1200):
    double_click(w, x, y); QTest.qWait(wait); return state()
def restore():
    w.showNormal(); AppKit.NSApp.activateIgnoringOtherApps_(True); w.raise_(); w.activateWindow(); QTest.qWait(1200)
    if ns.isZoomed():
        ns.zoom_(None); QTest.qWait(1000)
    return state()
tb = w.titleBar
empty = (tb.width() - 60, 20)                                   # 标题栏右侧的空白处
label = tb.titleLabel.mapTo(w, tb.titleLabel.rect().center())  # 窗口标题文字
label = (label.x(), label.y())
p = w.navigationInterface.panel
back = p.returnButton.mapTo(w, p.returnButton.rect().center()); back = (back.x(), back.y())
menu = p.menuButton.mapTo(w, p.menuButton.rect().center()); menu = (menu.x(), menu.y())
result = [real_read(), state(), empty, label]
"""


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def main(db: Path):
    sys.path.insert(0, str(harness.ROOT))
    from app import mac_window

    a = harness.App(db, "24A")
    check("程序正常启动", bool((a.state() or {}).get("ready")))
    real_pref, start, empty, label = a.run(SETUP)
    print(f"      本机系统设置：AppleActionOnDoubleClick={real_pref[0]!r}，AppleMiniaturizeOnDoubleClick={real_pref[1]}")

    print("\n[1] 按本机当前的系统设置")
    expected = mac_window.double_click_action(*real_pref)
    r = a.run("result = [dbl(*empty), restore()]")
    after, restored = r
    ok = {mac_window.ZOOM: after["zoomed"] and after["width"] > start["width"],
          mac_window.MINIMIZE: after["minimized"],
          mac_window.NOTHING: after == start}[expected]
    check(f"双击标题栏执行「{expected}」", ok, f"{start} → {after}")

    print("\n[2] 拡大/縮小")
    r = a.run("injected['pref'] = ('Maximize', False)\n"
              "result = [dbl(*empty), dbl(*empty), dbl(*label), dbl(*label)]")
    check("双击空白处缩放", r[0]["zoomed"] and r[0]["width"] > start["width"] and not r[0]["fullscreen"], str(r[0]))
    check("再双击还原", not r[1]["zoomed"] and r[1]["width"] == start["width"], str(r[1]))
    check("双击窗口标题文字处同样缩放 / 还原", r[2]["zoomed"] and not r[3]["zoomed"], f"{r[2]} / {r[3]}")

    print("\n[3] しまう / 何もしない")
    r = a.run("injected['pref'] = ('Minimize', False); m = dbl(*empty); result = [m, restore()]")
    check("「しまう」：最小化", r[0]["minimized"] and not r[0]["zoomed"], str(r[0]))
    r = a.run("injected['pref'] = ('None', False); result = [dbl(*empty), dbl(*label)]")
    check("「何もしない」：窗口不变", r[0] == r[1] == {**start, "minimized": False}, str(r))
    r = a.run("injected['pref'] = (None, True); m = dbl(*empty); result = [m, restore()]")
    check("未设置新设置、旧版设置为最小化：最小化", r[0]["minimized"], str(r[0]))

    print("\n[4] ← ☰")
    r = a.run("""
injected['pref'] = ('Maximize', False)
def go(page):
    w.switchTo(page); QTest.qWait(400)
# 历史中有两页，← 点一次后仍可用：双击 ← 由按钮自己处理（返回两次），不交给标题栏
go(w.reportInterface); go(w.monitorInterface); go(w.reportInterface)
on_dbl_back = dbl(*back)
page_after_dbl_back = w.stackedWidget.currentWidget().objectName()
# ← 不可用（没有可返回的页面）时，它不处理点击，双击等同于双击标题栏（与 macOS 工具栏中不可用的按钮一致）
while p.returnButton.isEnabled():   # 页面历史是全局的：返回到底，← 才不可用
    p.returnButton.click(); QTest.qWait(300)
disabled = not p.returnButton.isEnabled()
on_dbl_disabled_back = dbl(*back)
restore()
# 单击 ← 仍然返回上一页
go(w.reportInterface)
p.returnButton.click(); QTest.qWait(500)
page_after_click_back = w.stackedWidget.currentWidget().objectName()
on_dbl_menu = dbl(*menu)
if p.width() > 48:   # 双击 ☰ 是两次单击，库在展开动画中忽略收起，可能停在展开状态
    p.menuButton.click(); QTest.qWait(700)
p.menuButton.click(); QTest.qWait(700); expanded = p.width()
p.menuButton.click(); QTest.qWait(700)
result = [on_dbl_back, page_after_dbl_back, disabled, on_dbl_disabled_back, page_after_click_back, on_dbl_menu,
          expanded, p.width()]
""")
    on_dbl_back, page_dbl, disabled, on_dbl_disabled, page_click, on_dbl_menu, expanded, collapsed = r
    check("双击 ←（可用）不缩放窗口，由 ← 自己处理", not on_dbl_back["zoomed"] and not on_dbl_back["minimized"]
          and page_dbl != "report-interface", f"{on_dbl_back}，页面 {page_dbl}")
    check("← 不可用时双击，等同于双击标题栏", disabled and on_dbl_disabled["zoomed"], f"不可用={disabled}，{on_dbl_disabled}")
    check("双击 ☰ 不缩放窗口", not on_dbl_menu["zoomed"] and not on_dbl_menu["minimized"], str(on_dbl_menu))
    check("单击 ← 仍然返回上一页", page_click == "monitor-interface", page_click)
    check("单击 ☰ 仍然展开 / 收起侧边栏", expanded > 100 and collapsed == 48, f"{expanded} / {collapsed}")

    print("\n[5] 全屏")
    r = a.run("injected['pref'] = ('Maximize', False); w.showFullScreen(); QTest.qWait(2500)\n"
              "f = dbl(*empty); injected['pref'] = ('Minimize', False); g = dbl(*empty)\n"
              "w.showNormal(); QTest.qWait(2500); result = [f, g, state()]")
    check("全屏时双击标题栏没有效果", r[0]["fullscreen"] and r[1]["fullscreen"] and not r[1]["minimized"], str(r[:2]))
    check("退出全屏", not r[2]["fullscreen"], str(r[2]))

    a.run("w.close()")
    code = a.wait_exit()
    check("关闭后退出码 0", code == 0, str(code))


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("SKIP  只在 macOS 上运行（Windows 上双击标题栏的行为不变，由作者手动回归）")
        sys.exit(0)
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "data" / "self_analysis.db")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
