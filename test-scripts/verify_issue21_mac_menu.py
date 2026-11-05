"""
#21 macOS 菜单栏骨架的专项验证（补充 tests/test_mac_menu.py）

运行：在仓库根目录执行（会弹出真实窗口，约 1 分钟）
    python test-scripts/verify_issue21_mac_menu.py

每个「程序」都是一个独立进程，按 `python main.py` 的真实启动流程运行（runpy 执行 main.py 的 __main__），
只是数据库换成临时文件、摄像头换成假摄像头；本脚本通过标准输入向它发命令，读回它的状态。
检查项：
  1. 原生应用菜单：可见项目的日语名称、顺序与快捷键；「設定…」「About Qt」不可见；菜单栏控件不出现在窗口内
  2. 菜单栏中加粗的程序名为 Self-Analysis（屏幕截图 + macOS Vision 文字识别）
  3. 点「Self-Analysisについて」打开标准关于面板，显示程序名、版本号 0.1.0 与日语说明
  4. 记录中按 ⌘Q（向 NSApp 发送真实的按键事件）：经过 closeEvent，记录标为 completed，程序退出码 0
  5. 暂停中按 ⌘Q：同样标为 completed；未开始记录时按 ⌘Q：正常退出、不留空记录
非 macOS 平台：跳过全部检查（退出码 0）。数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import os
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
failures = []

EXPECTED_APP_MENU = [
    ("Self-Analysisについて", "", "⌘"),
    ("サービス", "", "⌘"),
    ("Self-Analysisを隠す", "h", "⌘"),
    ("ほかを隠す", "h", "⌥⌘"),
    ("すべてを表示", "", "⌘"),
    ("Self-Analysisを終了", "q", "⌘"),
]


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


# ---------- 子进程：真实的 main.py ----------

def child(db: str) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SELF.parent))
    os.chdir(ROOT)
    import runpy

    from PyQt5.QtWidgets import QApplication

    from app import session_store
    from app.view.monitor_interface import MonitorInterface
    import verify_issue4_recording as v4

    session_store.DEFAULT_DB_PATH = Path(db)
    v4.install_fake_camera()
    MonitorInterface._ask_course_name = lambda self, names: None
    real_exec = QApplication.exec
    keep = []

    def exec_(app):
        g = sys._getframe(1).f_globals          # main.py 的 __main__ 模块里的 w
        keep.append(Controller(app, g["w"], db))
        return real_exec()

    QApplication.exec = exec_
    sys.argv = [str(ROOT / "main.py")]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def modifiers_text(mask) -> str:
    import AppKit
    text = ""
    for flag, sym in ((AppKit.NSEventModifierFlagControl, "⌃"), (AppKit.NSEventModifierFlagOption, "⌥"),
                      (AppKit.NSEventModifierFlagShift, "⇧"), (AppKit.NSEventModifierFlagCommand, "⌘")):
        if mask & flag:
            text += sym
    return text


def view_texts(view) -> list:
    """收集一个 NSView 树中所有文字（关于面板）"""
    out = []
    if hasattr(view, "stringValue") and view.respondsToSelector_("stringValue"):
        out.append(str(view.stringValue()))
    if view.respondsToSelector_("string") and view.respondsToSelector_("textStorage"):
        out.append(str(view.string()))
    for sub in view.subviews():
        out.extend(view_texts(sub))
    return [t for t in out if t]


def ocr(path: str) -> list:
    import Vision
    from Foundation import NSURL
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(path), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLanguages_(["en-US"])
    handler.performRequests_error_([request], None)
    return [str(r.topCandidates_(1)[0].string()) for r in (request.results() or [])]


class Controller:
    def __init__(self, app, w, db):
        from PyQt5.QtCore import QSocketNotifier, QTimer
        self.app, self.w, self.db = app, w, db
        self.notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Read)
        self.notifier.activated.connect(self._on_command)
        QTimer.singleShot(1500, lambda: self._reply({"ready": True}))

    def _reply(self, extra=None):
        m = self.w.monitorInterface
        sid = getattr(getattr(m, "camera_thread", None), "session_id", None)
        conn = sqlite3.connect(self.db)
        row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone() if sid else None
        samples = conn.execute("SELECT COUNT(*) FROM samples WHERE session_id = ?", (sid,)).fetchone()[0] if sid else 0
        conn.close()
        state = {"session": sid, "status": row[0] if row else None, "samples": samples}
        state.update(extra or {})
        print("STATE " + json.dumps(state, ensure_ascii=False), flush=True)

    def _app_menu(self):
        """应用菜单；先按打开菜单时的方式重新校验各项是否可用（NSMenu 的 enabled 只在校验时更新）"""
        import AppKit
        menu = AppKit.NSApp.mainMenu().itemAtIndex_(0).submenu()
        menu.update()
        return menu

    def _on_command(self):
        import AppKit
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        from PyQt5.QtWidgets import QMenuBar
        line = sys.stdin.readline()
        if not line:
            self.notifier.setEnabled(False)
            return
        cmd = line.split()
        m = self.w.monitorInterface
        extra = {}
        if cmd[0] == "click":
            QTest.mouseClick({"start": m.startBtn, "pause": m.pauseBtn}[cmd[1]], Qt.LeftButton)
        elif cmd[0] == "menu":
            menu = self._app_menu()
            items = []
            for i in range(menu.numberOfItems()):
                it = menu.itemAtIndex_(i)
                if not it.isSeparatorItem():
                    items.append([str(it.title()), str(it.keyEquivalent()),
                                  modifiers_text(it.keyEquivalentModifierMask()), bool(it.isHidden()),
                                  bool(it.isEnabled())])
            bars = self.w.findChildren(QMenuBar)
            extra = {"items": items, "menubar_widgets": len(bars), "menubar_visible": any(b.isVisible() for b in bars)}
        elif cmd[0] == "menubar":
            AppKit.NSApp.activateIgnoringOtherApps_(True)
            QTest.qWait(1000)
            shot = os.path.join(tempfile.gettempdir(), "verify_issue21_menubar.png")
            # 截取左上角一小块：截图过宽时 Vision 识别不出文字
            self.app.primaryScreen().grabWindow(0, 0, 0, 240, 36).save(shot)
            extra = {"active": int(self.app.applicationState()) == 4, "texts": ocr(shot), "shot": shot}
        elif cmd[0] == "about":
            menu = self._app_menu()
            for i in range(menu.numberOfItems()):
                if str(menu.itemAtIndex_(i).title()).endswith("について") and not menu.itemAtIndex_(i).isHidden():
                    menu.performActionForItemAtIndex_(i)
                    break
            QTest.qWait(1500)
            panels = [win for win in AppKit.NSApp.windows() if win.isVisible() and win.isKindOfClass_(AppKit.NSPanel)]
            texts = [t for p in panels for t in view_texts(p.contentView())]
            shot = os.path.join(tempfile.gettempdir(), "verify_issue21_about.png")
            if panels:
                f = panels[0].frame()
                screen_h = AppKit.NSScreen.mainScreen().frame().size.height
                self.app.primaryScreen().grabWindow(0, int(f.origin.x), int(screen_h - f.origin.y - f.size.height),
                                                    int(f.size.width), int(f.size.height)).save(shot)
            for p in panels:
                p.close()
            extra = {"panels": len(panels), "texts": texts, "shot": shot}
        elif cmd[0] == "cmdq":
            self._reply({"quitting": True})
            event = AppKit.NSEvent.keyEventWithType_location_modifierFlags_timestamp_windowNumber_context_characters_charactersIgnoringModifiers_isARepeat_keyCode_(
                AppKit.NSEventTypeKeyDown, (0, 0), AppKit.NSEventModifierFlagCommand, 0,
                AppKit.NSApp.keyWindow().windowNumber() if AppKit.NSApp.keyWindow() else 0,
                None, "q", "q", False, 12)
            AppKit.NSApp.sendEvent_(event)
            return
        self._reply(extra)


# ---------- 父进程 ----------

class App:
    def __init__(self, db: Path, name: str):
        self.err = open(Path(tempfile.gettempdir()) / f"verify_issue21_{name}.err", "w")
        self.proc = subprocess.Popen([sys.executable, str(SELF), "--child", str(db)], cwd=ROOT, text=True,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err)
        self.lines = queue.Queue()
        self.out = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.out.append(line)
            if line.startswith("STATE "):
                self.lines.put(json.loads(line[6:]))
        self.lines.put(None)

    def state(self, timeout=30):
        try:
            return self.lines.get(timeout=timeout)
        except queue.Empty:
            return None

    def send(self, cmd, timeout=30):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        return self.state(timeout)

    def wait_exit(self, timeout=30):
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return "timeout"


def status(db, sid):
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone()
    conn.close()
    return row[0] if row else None


def count_sessions(db):
    conn = sqlite3.connect(str(db))
    n = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    conn.close()
    return n


def main(db: Path):
    sys.path.insert(0, str(ROOT))
    from app.app_info import ABOUT_TEXT, APP_NAME, APP_VERSION

    print("\n[1–3] 应用菜单、菜单栏中的程序名、关于面板")
    a = App(db, "A")
    s = a.state()
    check("程序正常启动", bool(s and s.get("ready")), str(s))
    s = a.send("menu")
    visible = [tuple(i[:3]) for i in s["items"] if not i[3]]
    hidden = [i[0] for i in s["items"] if i[3]]
    check("应用菜单可见项目的名称、顺序、快捷键", visible == EXPECTED_APP_MENU, f"实际 {visible}")
    enabled = {i[0]: i[4] for i in s["items"] if not i[3]}
    # 「すべてを表示」在没有被隐藏的程序时由系统置灰，属正常
    check("「について」「隠す」「ほかを隠す」「終了」可以点击",
          all(enabled.get(t) for t in ("Self-Analysisについて", "Self-Analysisを隠す", "ほかを隠す", "Self-Analysisを終了")),
          str(enabled))
    check("「設定…」「About Qt」不可见", all(not i[0].startswith(("設定", "About Qt", "Qtについて")) for i in s["items"] if not i[3]),
          f"隐藏项 {hidden}")
    check("菜单栏控件不出现在窗口内（使用原生菜单栏）", s["menubar_widgets"] == 1 and not s["menubar_visible"], str(s))
    s = a.send("menubar")
    joined = " ".join(s["texts"])
    check("菜单栏中加粗的程序名为 Self-Analysis", s["active"] and APP_NAME in joined and "python" not in joined.lower(),
          f"识别到 {s['texts']}，截图 {s['shot']}")
    s = a.send("about")
    texts = s["texts"]
    check("「について」打开关于面板", s["panels"] == 1, f"面板数 {s['panels']}")
    texts = [t for t in texts if not t.startswith("<NSImage")]
    check("关于面板显示程序名、版本号、日语说明（无英文的 Version 行、无 Python 版本）",
          APP_NAME in texts and any(f"バージョン {APP_VERSION}" in t for t in texts) and any(ABOUT_TEXT in t for t in texts)
          and not any("Version" in t or "3.10" in t for t in texts), f"{texts}，截图 {s['shot']}")

    print("\n[4] 记录中按 ⌘Q")
    a.send("click start")
    time.sleep(6)
    s = a.send("state")
    sid = s["session"]
    check("按 ⌘Q 前正在记录", s["status"] == "recording" and s["samples"] > 0, str(s))
    a.send("cmdq")
    code = a.wait_exit()
    check("⌘Q 后程序退出，退出码 0", code == 0, f"退出码 {code}")
    check("记录标为 completed（经过 closeEvent → stop()）", status(db, sid) == "completed", str(status(db, sid)))
    check("closeEvent 释放了摄像头", any("正在安全释放摄像头资源" in l for l in a.out))

    print("\n[5] 暂停中按 ⌘Q；未开始记录时按 ⌘Q")
    b = App(db, "B")
    b.state()
    b.send("click start")
    time.sleep(6)
    b.send("click pause")
    time.sleep(1)
    s = b.send("state")
    sid = s["session"]
    b.send("cmdq")
    code = b.wait_exit()
    check("暂停中 ⌘Q：退出码 0，记录标为 completed", code == 0 and status(db, sid) == "completed",
          f"退出码 {code}，{status(db, sid)}")
    before = count_sessions(db)
    c = App(db, "C")
    c.state()
    c.send("cmdq")
    code = c.wait_exit()
    check("未开始记录时 ⌘Q：退出码 0，不留空记录", code == 0 and count_sessions(db) == before,
          f"退出码 {code}，记录数 {before} → {count_sessions(db)}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    elif sys.platform != "darwin":
        print("SKIP  只在 macOS 上运行（Windows 上不创建菜单栏，由作者手动回归）")
        sys.exit(0)
    else:
        with tempfile.TemporaryDirectory() as d:
            main(Path(d) / "data" / "self_analysis.db")
        print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
        sys.exit(1 if failures else 0)
