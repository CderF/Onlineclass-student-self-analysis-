"""
#20 macOS 左上角布局的专项验证（补充 tests/test_mac_window.py）

运行：在仓库根目录执行（会弹出真实窗口并进入一次全屏，约 30 秒）
    python test-scripts/verify_issue20_mac_title_row.py

MainWindow 在 offscreen 下会 segfault，所以用原生平台打开真实窗口。检查项：
  1. 红黄绿（NSWindow 原生按钮）在窗口左侧；屏幕截图上红色按钮确实可见（没有被系统标题栏裁掉）
  2. ← ☰ 在红黄绿右侧、与其同一行（垂直中心相差不超过 2px）、互不重叠
  3. 侧边导航第一项从第二行（标题栏下方）开始
  4. 改变窗口尺寸、点 ☰ 展开侧边栏后，以上关系不变；展开时标题栏一行仍在侧边栏上层
  5. 点绿色按钮进入全屏：← ☰ 可见、位置不变、可以点击；再点一次退出全屏后布局恢复
  6. 点 ← 回到上一页
「按住 Option 点绿色按钮为缩放」无法用程序模拟修饰键，列在 test-scripts/README.md 的 mac 手动检查清单中。
非 macOS 平台：跳过全部检查（退出码 0），Windows 由作者手动回归。数据库写在临时目录，不碰 data/。
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
failures = []

TOLERANCE = 2


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def main() -> int:
    if sys.platform != "darwin":
        print("SKIP  只在 macOS 上运行（Windows 由作者手动回归，见 test-scripts/README.md）")
        return 0

    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    import Cocoa
    import objc
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication

    import main as main_module
    from app import session_store

    session_store.DEFAULT_DB_PATH = Path(tempfile.mkdtemp()) / "verify20.db"

    app = QApplication(sys.argv)
    w = main_module.MainWindow()
    w.resize(1000, 700)
    w.move(80, 80)
    w.show()
    QTest.qWait(800)
    panel = w.navigationInterface.panel

    def ns_window():
        return objc.objc_object(c_void_p=int(w.winId())).window()

    def traffic_lights():
        """红黄绿在窗口坐标中的矩形 (x, top, w, h)；NSView 原点在左下角，换算成左上角"""
        result = {}
        for name, kind in (("close", Cocoa.NSWindowCloseButton), ("min", Cocoa.NSWindowMiniaturizeButton),
                           ("zoom", Cocoa.NSWindowZoomButton)):
            b = ns_window().standardWindowButton_(kind)
            f, sv = b.frame(), b.superview().frame()
            result[name] = (f.origin.x, sv.size.height - f.origin.y - f.size.height, f.size.width, f.size.height,
                            not b.isHidden())
        return result

    def rect_of(widget):
        p = widget.mapTo(w, widget.rect().topLeft())
        return p.x(), p.y(), widget.width(), widget.height()

    def first_item_top():
        item = list(panel.items.values())[0].widget
        return item.mapTo(w, item.rect().topLeft()).y()

    def check_layout(tag, expect_lights=True):
        lights = traffic_lights()
        cx, ctop, cw, ch, cvis = lights["close"]
        zx = lights["zoom"][0] + lights["zoom"][2]
        light_center = ctop + ch / 2
        rx, ry, rw, rh = rect_of(panel.returnButton)
        mx, my, mw, mh = rect_of(panel.menuButton)
        if expect_lights:
            check(f"[{tag}] 红黄绿在窗口左侧", cx < 40 and zx < 100 and cvis, f"close x={cx}, zoom 右边={zx}, 窗口宽={w.width()}")
            check(f"[{tag}] ← 在红黄绿右侧、不重叠", rx >= zx, f"← x={rx}, zoom 右边={zx}")
            check(f"[{tag}] ← ☰ 与红黄绿同一行", abs(ry + rh / 2 - light_center) <= TOLERANCE
                  and abs(my + mh / 2 - light_center) <= TOLERANCE,
                  f"红黄绿中心 y={light_center}, ← 中心={ry + rh / 2}, ☰ 中心={my + mh / 2}")
        check(f"[{tag}] ☰ 在 ← 右侧、不重叠、同一高度", mx >= rx + rw and my == ry, f"← {rx},{ry} ☰ {mx},{my}")
        check(f"[{tag}] ← ☰ 可见", panel.returnButton.isVisible() and panel.menuButton.isVisible())
        top = first_item_top()
        check(f"[{tag}] 侧边导航从第二行开始", top >= w.titleBar.height(), f"第一项 y={top}, 标题栏高={w.titleBar.height()}")
        return lights

    # 1–3 初始状态
    lights = check_layout("初始")
    cx, ctop, cw, ch, _ = lights["close"]
    g = w.geometry()
    img = app.primaryScreen().grabWindow(0, g.x(), g.y(), 120, 48).toImage()
    ratio = img.width() / 120
    color = img.pixelColor(int((cx + cw / 2) * ratio), int((ctop + ch / 2) * ratio))
    check("[初始] 截图中红色按钮可见（未被裁掉）", color.red() > 200 and color.green() < 140 and color.blue() < 140,
          f"按钮中心像素 rgb=({color.red()},{color.green()},{color.blue()})")

    # 4 改变尺寸、展开侧边栏
    for size in ((1300, 850), (800, 560)):
        w.resize(*size)
        QTest.qWait(400)
        check_layout(f"尺寸 {w.width()}x{w.height()}")
    panel.menuButton.click()
    QTest.qWait(600)
    check("[☰ 展开] 侧边栏已展开", panel.width() > 100, f"宽={panel.width()}, 模式={panel.displayMode.name}")
    siblings = w.children()
    check("[☰ 展开] 标题栏一行在侧边栏上层", panel.parent() is not w
          or siblings.index(w.titleBar) > siblings.index(panel))
    check_layout("☰ 展开")
    panel.menuButton.click()
    QTest.qWait(600)
    check("[☰ 收起] 侧边栏已收起", panel.width() <= 48, f"宽={panel.width()}")

    # 5 全屏
    before = (rect_of(panel.returnButton), rect_of(panel.menuButton))
    ns_window().standardWindowButton_(Cocoa.NSWindowZoomButton).performClick_(None)
    QTest.qWait(2500)
    check("[全屏] 点绿色按钮进入全屏", w.isFullScreen())
    check_layout("全屏", expect_lights=False)
    check("[全屏] ← ☰ 位置不变", (rect_of(panel.returnButton), rect_of(panel.menuButton)) == before)
    panel.menuButton.click()
    QTest.qWait(600)
    check("[全屏] ☰ 可以点击（侧边栏展开）", panel.width() > 100, f"宽={panel.width()}")
    panel.menuButton.click()
    QTest.qWait(600)
    ns_window().standardWindowButton_(Cocoa.NSWindowZoomButton).performClick_(None)
    QTest.qWait(2500)
    check("[退出全屏] 点绿色按钮退出全屏", not w.isFullScreen())
    check_layout("退出全屏")

    # 6 ← 返回上一页
    list(panel.items.values())[1].widget.click()
    QTest.qWait(500)
    on_report = w.stackedWidget.currentWidget() is w.reportInterface
    panel.returnButton.click()
    QTest.qWait(500)
    check("[←] 切到报告页后点 ← 回到监控页", on_report and w.stackedWidget.currentWidget() is w.monitorInterface)

    w.close()
    QTest.qWait(300)
    print(f"\n{'全部通过' if not failures else f'{len(failures)} 项失败'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
