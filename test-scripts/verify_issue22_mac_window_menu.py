"""
#22 macOS「ウインドウ」菜单的专项验证（补充 tests/test_mac_menu.py、tests/test_mac_window.py）

运行：在仓库根目录执行（会弹出真实窗口、进入一次全屏，约 1 分钟；显示器需保持常亮，可用 caffeinate -d）
    python test-scripts/verify_issue22_mac_window_menu.py

按 `python main.py` 的真实启动流程运行子进程（框架见 verify_issue21_mac_menu.py），检查项：
  1. 原生「ウインドウ」菜单：项目名称、顺序、快捷键，都可以点击；位于「表示」之后（macOS 惯例）
  2. ⌘M（真实按键事件）把窗口收进程序坞；「しまう」菜单项同样
  3. 「拡大/縮小」缩放窗口，再点一次还原
  4. ⌃⌘F 进入全屏、再按一次退出；「フルスクリーンにする」菜单项同样
  5. 记录中按 ⌘W：程序退出（退出码 0），记录为 completed；暂停中点「閉じる」、记录中点红色按钮：同样
非 macOS 平台：跳过全部检查（退出码 0）。数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_issue21_mac_menu as harness  # noqa: E402

failures = []

EXPECTED = [
    ("しまう", "m", "⌘"),
    ("拡大/縮小", "", "⌘"),          # 没有快捷键的项目，修饰键为 NSMenuItem 的默认值
    ("フルスクリーンにする", "f", "⌃⌘"),
    ("閉じる", "w", "⌘"),
]

RESTORE = """
w.showNormal(); AppKit.NSApp.activateIgnoringOtherApps_(True); w.raise_(); w.activateWindow(); QTest.qWait(1200)
"""


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def status(db, sid):
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone()
    conn.close()
    return row[0] if row else None


def main(db: Path):
    print("\n[1] 「ウインドウ」菜单")
    a = harness.App(db, "22A")
    check("程序正常启动", bool((a.state() or {}).get("ready")))
    a.run(RESTORE)
    items = a.run("result = native_menu_items('ウインドウ')")
    visible = [tuple(i[:3]) for i in items if not i[3]]
    check("项目名称、顺序、快捷键", visible == EXPECTED, f"实际 {visible}")
    check("所有项目都可以点击", all(i[4] for i in items if not i[3]), str(items))
    titles = a.run("m = AppKit.NSApp.mainMenu(); result = [str(m.itemAtIndex_(i).title()) for i in range(m.numberOfItems())"
                   " if not m.itemAtIndex_(i).isHidden()]")
    check("菜单栏顺序：应用菜单、編集、表示、ウインドウ", titles == ["Apple", "編集", "表示", "ウインドウ"], str(titles))

    print("\n[2] しまう")
    r = a.run("send_key('m', 'cmd'); QTest.qWait(1500); result = w.isMinimized()")
    check("⌘M 把窗口收进程序坞", r is True)
    a.run(RESTORE)
    r = a.run("click_menu_item('ウインドウ', 'しまう'); QTest.qWait(1500); result = w.isMinimized()")
    check("「しまう」菜单项把窗口收进程序坞", r is True)
    a.run(RESTORE)

    print("\n[3] 拡大/縮小")
    r = a.run("""
from app import mac_window
ns = mac_window.ns_window(w)
before = (ns.isZoomed(), ns.frame().size.width)
click_menu_item('ウインドウ', '拡大/縮小'); QTest.qWait(1200)
zoomed = (ns.isZoomed(), ns.frame().size.width)
click_menu_item('ウインドウ', '拡大/縮小'); QTest.qWait(1200)
result = [before, zoomed, (ns.isZoomed(), ns.frame().size.width), w.isFullScreen()]
""")
    before, zoomed, after, fullscreen = r
    check("点「拡大/縮小」后窗口缩放（变大，不进入全屏）", not before[0] and zoomed[0] and zoomed[1] > before[1] and not fullscreen,
          f"{before} → {zoomed}")
    check("再点一次还原", not after[0] and after[1] == before[1], f"{zoomed} → {after}")

    print("\n[4] フルスクリーン")
    r = a.run("send_key('f', 'ctrl', 'cmd'); QTest.qWait(2500); a = w.isFullScreen();"
              "send_key('f', 'ctrl', 'cmd'); QTest.qWait(2500); result = [a, w.isFullScreen()]")
    check("⌃⌘F 进入全屏，再按一次退出", r == [True, False], str(r))
    r = a.run("click_menu_item('ウインドウ', 'フルスクリーンにする'); QTest.qWait(2500); a = w.isFullScreen();"
              "click_menu_item('ウインドウ', 'フルスクリーンにする'); QTest.qWait(2500); result = [a, w.isFullScreen()]")
    check("「フルスクリーンにする」菜单项进入全屏，再点一次退出", r == [True, False], str(r))

    print("\n[5] 閉じる / 红色按钮")
    a.send("click start")
    time.sleep(6)
    s = a.send("state")
    sid = s["session"]
    check("按 ⌘W 前正在记录", s["status"] == "recording" and s["samples"] > 0, str(s))
    a.run(RESTORE + "send_key('w', 'cmd')")
    code = a.wait_exit()
    check("记录中 ⌘W：程序退出，退出码 0", code == 0, f"退出码 {code}")
    check("记录标为 completed", status(db, sid) == "completed", str(status(db, sid)))

    b = harness.App(db, "22B")
    b.state()
    b.send("click start")
    time.sleep(6)
    b.send("click pause")
    time.sleep(1)
    sid = b.send("state")["session"]
    b.run(RESTORE + "click_menu_item('ウインドウ', '閉じる')")
    code = b.wait_exit()
    check("暂停中点「閉じる」：退出码 0，记录为 completed", code == 0 and status(db, sid) == "completed",
          f"退出码 {code}，{status(db, sid)}")

    c = harness.App(db, "22C")
    c.state()
    c.send("click start")
    time.sleep(6)
    sid = c.send("state")["session"]
    c.run(RESTORE + "from app import mac_window\n"
          "mac_window.ns_window(w).standardWindowButton_(AppKit.NSWindowCloseButton).performClick_(None)")
    code = c.wait_exit()
    check("记录中点红色按钮：退出码 0，记录为 completed", code == 0 and status(db, sid) == "completed",
          f"退出码 {code}，{status(db, sid)}")


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("SKIP  只在 macOS 上运行（Windows 上没有菜单栏，由作者手动回归）")
        sys.exit(0)
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "data" / "self_analysis.db")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    sys.exit(1 if failures else 0)
