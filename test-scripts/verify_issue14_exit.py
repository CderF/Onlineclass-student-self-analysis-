"""
#14 程序退出时偶发 SIGSEGV 的专项验证

运行：在仓库根目录执行（反复弹出真实窗口，默认 3 种退出方式 × 20 次，约 8 分钟）
    python test-scripts/verify_issue14_exit.py              # 当前代码
    python test-scripts/verify_issue14_exit.py --baseline   # 对照：关掉修复（不在关闭时销毁主窗口），预期出现退出码 139
    python test-scripts/verify_issue14_exit.py --runs 40

每次都是独立进程，按 `python main.py` 的真实启动流程运行（runpy 执行 main.py 的 __main__，原生平台），
数据库换成临时文件，退出方式：
  close   3 秒后关闭窗口
  quit    3 秒后 app.quit()（不经关闭窗口）
  record  用假摄像头记录约 5 秒后关闭窗口，检查记录已保存为 completed
检查项：每种方式全部退出码为 0、没有 Fatal Python error；record 的记录全部为 completed。
崩溃时用 -X faulthandler 把 Python 栈写到输出。任一检查失败时退出码为 1。
"""
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()
failures = []
MODES = ("close", "quit", "record")


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


def child(db: str, mode: str, baseline: bool) -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(SELF.parent))
    os.chdir(ROOT)
    import runpy

    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import QApplication

    from app import session_store
    from app.view.monitor_interface import MonitorInterface

    session_store.DEFAULT_DB_PATH = Path(db)
    if mode == "record":
        import verify_issue4_recording as v4
        v4.install_fake_camera()
        MonitorInterface._ask_course_name = lambda self, names: None
    real_exec = QApplication.exec

    def exec_(app):
        g = sys._getframe(1).f_globals              # main.py 的 __main__ 模块
        w = g["w"]
        if baseline:                                 # 修复前：关闭时不销毁、exec() 之后也不销毁
            w.setAttribute(Qt.WA_DeleteOnClose, False)

            class NoDelete:
                @staticmethod
                def isdeleted(obj):
                    return True
            g["sip"] = NoDelete
        if mode == "record":
            QTimer.singleShot(1500, w.monitorInterface.startBtn.click)
            QTimer.singleShot(6500, w.close)
        elif mode == "quit":
            QTimer.singleShot(3000, app.quit)
        else:
            QTimer.singleShot(3000, w.close)
        return real_exec()

    QApplication.exec = exec_
    sys.argv = [str(ROOT / "main.py")]
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")


def run_once(db: Path, mode: str, baseline: bool):
    cmd = [sys.executable, "-X", "faulthandler", str(SELF), "--child", str(db), mode] + (["--baseline"] if baseline else [])
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=60)
        return p.returncode, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", ""


def main(runs: int, baseline: bool):
    print(f"{'对照（关掉修复）' if baseline else '当前代码'}：每种退出方式 {runs} 次", flush=True)
    for mode in MODES:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "data" / "self_analysis.db"
            codes, fatal = Counter(), 0
            start = time.monotonic()
            for i in range(runs):
                code, out = run_once(db, mode, baseline)
                codes[code] += 1
                if "Fatal Python error" in out:
                    fatal += 1
                if code != 0:
                    print(f"      第 {i + 1} 次退出码 {code}", flush=True)
            name = f"{mode}：{runs} 次全部退出码 0、无 Fatal Python error"
            check(name, codes == Counter({0: runs}) and fatal == 0,
                  f"退出码 {dict(codes)}，Fatal {fatal} 次，{time.monotonic() - start:.0f} 秒")
            if mode == "record":
                conn = sqlite3.connect(str(db))
                status = Counter(r[0] for r in conn.execute("SELECT status FROM sessions"))
                conn.close()
                check(f"record：{runs} 条记录全部保存为 completed", status == Counter({"completed": runs}), str(dict(status)))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3], "--baseline" in sys.argv)
    else:
        runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 20
        main(runs, "--baseline" in sys.argv)
        print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
        sys.exit(1 if failures else 0)
