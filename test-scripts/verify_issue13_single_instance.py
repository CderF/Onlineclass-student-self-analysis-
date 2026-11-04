"""
#13 同时打开两个程序的专项验证（补充 tests/test_single_instance.py）

运行：在仓库根目录执行（会弹出真实窗口，约 1 分钟）
    python test-scripts/verify_issue13_single_instance.py

每个「程序」都是一个独立进程，按 `python main.py` 的真实启动流程运行（runpy 执行 main.py 的 __main__），
只是数据库换成临时文件、摄像头换成假摄像头；本脚本通过标准输入向它发命令（点「開始」等），读回它的状态。
检查项：
  1. A 正在记录（窗口最小化）时启动 B：B 立刻退出、不出窗口；A 的窗口恢复并调到最前面；A 的记录仍为 recording 且继续增加
  2. A 暂停中启动 B：同样不影响；继续后结束为 completed，时间轴连续；A 正常关闭后锁文件被删除
  3. A 被 SIGKILL（崩溃）后启动 C：C 接管失效的锁正常启动，A 留下的记录标为 interrupted（原有行为保持）
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import json
import os
import queue
import signal
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
        g = sys._getframe(1).f_globals          # main.py 的 __main__ 模块里的 w、instance
        keep.append(Controller(app, g["w"], g["instance"], db))
        return real_exec()

    QApplication.exec = exec_
    sys.argv = [str(ROOT / "main.py")]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


class Controller:
    def __init__(self, app, w, instance, db):
        from PyQt5.QtCore import QSocketNotifier
        self.app, self.w, self.db = app, w, db
        self.activated = 0
        instance.activated.connect(self._on_activated)
        self.notifier = QSocketNotifier(sys.stdin.fileno(), QSocketNotifier.Read)
        self.notifier.activated.connect(self._on_command)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(1500, lambda: self._reply({"ready": True}))

    def _on_activated(self):
        self.activated += 1

    def _reply(self, extra=None):
        m = self.w.monitorInterface
        sid = getattr(getattr(m, "camera_thread", None), "session_id", None)
        conn = sqlite3.connect(self.db)
        row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone() if sid else None
        samples = conn.execute("SELECT COUNT(*) FROM samples WHERE session_id = ?", (sid,)).fetchone()[0] if sid else 0
        conn.close()
        try:
            from AppKit import NSWorkspace
            front = NSWorkspace.sharedWorkspace().frontmostApplication().processIdentifier()
        except Exception:
            front = None
        state = {"pid": os.getpid(), "activated": self.activated, "visible": self.w.isVisible(),
                 "minimized": self.w.isMinimized(), "active_window": self.w.isActiveWindow(),
                 "app_active": int(self.app.applicationState()) == 4, "front_pid": front,
                 "session": sid, "status": row[0] if row else None, "samples": samples}
        state.update(extra or {})
        print("STATE " + json.dumps(state), flush=True)

    def _on_command(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest
        line = sys.stdin.readline()
        if not line:
            self.notifier.setEnabled(False)
            return
        cmd = line.split()
        m = self.w.monitorInterface
        extra = {}
        if cmd[0] == "click":
            QTest.mouseClick({"start": m.startBtn, "pause": m.pauseBtn, "stop": m.stopBtn}[cmd[1]], Qt.LeftButton)
        elif cmd[0] == "minimize":
            self.w.showMinimized()
        elif cmd[0] == "shot":
            g = self.w.frameGeometry()
            img = self.app.primaryScreen().grabWindow(0, g.x(), g.y(), g.width(), g.height())
            img.save(cmd[1])
            extra = {"shot_size": [img.width(), img.height()]}
        elif cmd[0] == "close":
            self._reply({"closing": True})
            self.w.close()
            return
        self._reply(extra)


# ---------- 父进程 ----------

class App:
    def __init__(self, db: Path, name: str):
        self.name = name
        self.err = open(Path(tempfile.gettempdir()) / f"verify_issue13_{name}.err", "w")
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


def second_instance(db: Path):
    """按真实流程再启动一次程序；预期立刻退出。返回 (退出码, 耗时秒, 输出)"""
    start = time.monotonic()
    try:
        p = subprocess.run([sys.executable, str(SELF), "--child", str(db)], cwd=ROOT, text=True,
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=30)
        return p.returncode, time.monotonic() - start, p.stdout + p.stderr
    except subprocess.TimeoutExpired as e:
        return "timeout", time.monotonic() - start, str(e.stdout)


def timeline(db, sid):
    conn = sqlite3.connect(str(db))
    ts = [r[0] for r in conn.execute("SELECT t FROM samples WHERE session_id = ? ORDER BY t", (sid,))]
    conn.close()
    return ts


def status(db, sid):
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone()
    conn.close()
    return row[0] if row else None


def main(db: Path):
    lock = db.with_name(db.name + ".lock")
    print("\n[1] A 记录中（窗口最小化）时启动 B")
    a = App(db, "A")
    s = a.state()
    check("A 正常启动", bool(s and s.get("ready")), str(s))
    a.send("click start")
    time.sleep(6)
    s = a.send("minimize")
    subprocess.run(["open", "-a", "Finder"])         # 前台换成别的程序，A 不再是前台程序
    time.sleep(1.5)
    s = a.send("state")
    sid = s["session"]
    check("A 在记录、窗口已最小化、A 不在前台", s["status"] == "recording" and s["samples"] > 0 and s["minimized"]
          and not s["app_active"] and s["front_pid"] != s["pid"], str(s))
    code, secs, out = second_instance(db)
    check("B 立刻退出（退出码 0、不出窗口）", code == 0 and secs < 10 and "程序已在运行，已调出已打开的窗口" in out,
          f"退出码 {code}，{secs:.1f} 秒，输出 {out.strip()[-80:]!r}")
    time.sleep(1.5)
    s = a.send("state")
    shot = Path(tempfile.gettempdir()) / "verify_issue13_activated.png"
    a.send(f"shot {shot}")
    check("A 收到通知一次，窗口恢复且可见", s["activated"] == 1 and s["visible"] and not s["minimized"], str(s))
    check("A 的窗口调到最前面（成为活动窗口，A 是前台程序）",
          s["active_window"] and s["app_active"] and s["front_pid"] == s["pid"],
          f"active_window={s['active_window']} app_active={s['app_active']} front_pid={s['front_pid']} pid={s['pid']}，截图 {shot}")
    before = s["samples"]
    time.sleep(2)
    s = a.send("state")
    check("A 的记录仍为 recording 且继续增加", s["status"] == "recording" and s["samples"] > before,
          f"{before} → {s['samples']} 秒")

    print("\n[2] A 暂停中启动 B，之后继续、结束、关闭")
    a.send("click pause")
    time.sleep(1)
    code, secs, out = second_instance(db)
    check("B 立刻退出", code == 0 and secs < 10, f"退出码 {code}，{secs:.1f} 秒")
    s = a.send("state")
    check("暂停中的记录仍为 recording", s["status"] == "recording", str(s))
    a.send("click pause")
    time.sleep(2.5)
    a.send("click stop")
    time.sleep(1.5)
    ts = timeline(db, sid)
    check("继续后正常结束为 completed，时间轴连续", status(db, sid) == "completed" and ts == list(range(len(ts))),
          f"{status(db, sid)}，{len(ts)} 秒")
    a.send("close")
    try:
        code = a.proc.wait(20)
    except subprocess.TimeoutExpired:
        a.proc.kill()
        code = "timeout"
    check("A 关闭后锁文件被删除", not lock.exists(), f"A 退出码 {code}")

    print("\n[3] A 被 SIGKILL 后启动 C")
    a2 = App(db, "A2")
    a2.state()
    a2.send("click start")
    time.sleep(6)
    s = a2.send("state")
    crashed = s["session"]
    a2.proc.send_signal(signal.SIGKILL)
    a2.proc.wait(10)
    check("A 被强制结束时正在记录，锁文件留在原处", s["samples"] > 0 and status(db, crashed) == "recording" and lock.exists(),
          f"{s['samples']} 秒")
    c = App(db, "C")
    s = c.state()
    check("C 接管失效的锁正常启动", bool(s and s.get("ready")), str(s))
    check("A 留下的记录标为 interrupted", status(db, crashed) == "interrupted", str(status(db, crashed)))
    c.send("close")
    try:
        code = c.proc.wait(20)
    except subprocess.TimeoutExpired:
        c.proc.kill()
        code = "timeout"
    check("C 关闭后锁文件被删除", not lock.exists(), f"C 退出码 {code}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        try:
            import Quartz
            if Quartz.CGSessionCopyCurrentDictionary().get("CGSSessionScreenIsLocked"):
                print("屏幕已锁定或休眠：前台程序是 loginwindow，无法检查窗口是否调到最前面。解锁后重跑。")
                sys.exit(2)
        except ImportError:
            pass
        with tempfile.TemporaryDirectory() as d:
            main(Path(d) / "data" / "self_analysis.db")
        print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
        sys.exit(1 if failures else 0)
