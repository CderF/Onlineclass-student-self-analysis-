"""
#12 点击「開始」后界面保持响应的专项验证（补充 tests/test_model_loading.py）

运行：在仓库根目录执行（会短暂弹出真实窗口，约 40 秒）
    python test-scripts/verify_issue12_start_responsiveness.py               # 假摄像头
    python test-scripts/verify_issue12_start_responsiveness.py --real-camera  # 真实摄像头（需在有摄像头权限的终端里运行）

检查项（原生窗口、真实 Qt 鼠标点击「開始」）：
  1. 点击处理（_on_start_clicked）的耗时，以及点击后 6 秒内界面线程的单次最大阻塞（1 ms 定时器测量）< 100 ms；
     同时记录后台各阶段：模型加载、打开摄像头、首帧、FaceMesh / 表情模型首次推理
  2. 模型在后台线程构造；暂停后继续不重复加载
  3. 表情模型加载失败：窗口里出现「モデルエラー」提示，不自动消失、点关闭按钮才消失，监控照常进行
修改前（模型在界面线程构造）的实测：真实摄像头下点击处理约 40 ms、界面最大阻塞 58–67 ms；
慢的是后台打开摄像头（首次约 1.1 秒），不阻塞界面线程。
数据库写在临时目录，不碰 data/。任一检查失败时退出码为 1。
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(ROOT)  # ExpressionClassifier 以相对路径加载 weights/best.pt

failures = []
# QApplication 保持到进程退出：随函数返回释放时，PyQt 清理偶发 SIGSEGV（#14）
KEEP_ALIVE = []
REAL = "--real-camera" in sys.argv
LIMIT_MS = 100


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        failures.append(name)


class Timeline:
    """后台各阶段的时间点（相对点击时刻，毫秒）"""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.events = []

    def reset(self):
        self.t0 = time.perf_counter()
        self.events = []

    def mark(self, name):
        self.events.append((round((time.perf_counter() - self.t0) * 1000), name))


def instrument(timeline: Timeline, built: dict):
    """包装模型类与摄像头，记录构造线程与各阶段时间"""
    import app.camera_thread as ct
    for name in ("ExpressionClassifier", "FaceMeshInference"):
        real = getattr(ct, name)

        def make(*args, real=real, name=name, **kwargs):
            built[name].append(threading.current_thread() is threading.main_thread())
            timeline.mark(f"{name}() 开始")
            obj = real(*args, **kwargs)
            timeline.mark(f"{name}() 返回")
            return obj

        setattr(ct, name, make)
    real_cap = ct.cv2.VideoCapture

    class Cap:
        def __init__(self, index):
            timeline.mark("打开摄像头 开始")
            self.c = real_cap(index)
            timeline.mark("打开摄像头 返回")
            self.first = True

        def isOpened(self):
            return self.c.isOpened()

        def read(self):
            r = self.c.read()
            if self.first:
                self.first = False
                timeline.mark("首帧")
            return r

        def release(self):
            self.c.release()

    ct.cv2.VideoCapture = Cap


def main(db: Path):
    import PyQt5.QtWebEngineWidgets  # noqa: F401
    from PyQt5.QtGui import QFont
    from PyQt5.QtTest import QTest
    from PyQt5.QtWidgets import QApplication
    from qfluentwidgets import setFontFamilies

    from app import session_store
    from verify_issue16_ui_responsiveness import StallProbe

    session_store.DEFAULT_DB_PATH = db
    import main as app_main
    if not REAL:
        import verify_issue4_recording as v4
        v4.install_fake_camera()
    timeline = Timeline()
    built = {"ExpressionClassifier": [], "FaceMeshInference": []}
    instrument(timeline, built)

    app = QApplication.instance() or QApplication(sys.argv)
    KEEP_ALIVE.append(app)
    app.setFont(QFont("Hiragino Sans", 10))
    setFontFamilies(["Hiragino Sans", "Segoe UI"])
    w = app_main.MainWindow()
    w.resize(1100, 800)
    w.show()
    w.raise_()
    m = w.monitorInterface
    m._ask_course_name = lambda names: None
    QTest.qWait(2000)

    real_click = m._on_start_clicked
    handler = {}

    def timed_click():
        t = time.perf_counter()
        real_click()
        handler["ms"] = (time.perf_counter() - t) * 1000
    m.startBtn.clicked.disconnect()
    m.startBtn.clicked.connect(timed_click)

    frames = []
    probe = StallProbe()
    print(f"\n[1] 点击「開始」后的界面响应（{'真实摄像头' if REAL else '假摄像头'}）")
    for round_ in (1, 2):
        timeline.reset()
        frames.clear()
        probe.start()
        QTest.mouseClick(m.startBtn, 1)            # 真实 Qt 鼠标点击
        timeline.mark("点击处理返回")
        m.camera_thread.change_pixmap_signal.connect(lambda _: frames.append(time.perf_counter()))
        QTest.qWait(6000)
        stall = probe.stop()
        first_frame = round((frames[0] - timeline.t0) * 1000) if frames else None
        check(f"第 {round_} 次开始：点击处理 < 30 ms、6 秒内界面线程单次最大阻塞 < {LIMIT_MS} ms",
              handler.get("ms", 999) < 30 and stall < LIMIT_MS,
              f"点击处理 {handler.get('ms', -1):.1f} ms，最大阻塞 {stall:.0f} ms，首个画面 {first_frame} ms")
        print(f"      后台阶段（ms）：{timeline.events}", flush=True)
        if round_ == 1:
            print("\n[2] 模型在后台线程构造、暂停后继续不重复加载")
            check("模型在后台线程构造", built == {"ExpressionClassifier": [False], "FaceMeshInference": [False]}, str(built))
            for _ in range(2):
                QTest.mouseClick(m.pauseBtn, 1)
                QTest.qWait(500)
                QTest.mouseClick(m.pauseBtn, 1)
                QTest.qWait(1500)
            check("暂停 / 继续两次后仍只构造过一次", built == {"ExpressionClassifier": [False], "FaceMeshInference": [False]},
                  str(built))
        QTest.mouseClick(m.stopBtn, 1)
        QTest.qWait(1500)

    if not REAL:
        print("\n[3] 表情模型加载失败时的提示")
        import app.camera_thread as ct

        class Broken:
            model = None

            def process_face(self, face_img):
                return "Unknown", 0.0, None

        ct.ExpressionClassifier = Broken
        from qfluentwidgets import InfoBar
        before = set(id(b) for b in w.findChildren(InfoBar))
        scores = []
        QTest.mouseClick(m.startBtn, 1)
        m.camera_thread.update_score_signal.connect(scores.append)
        QTest.qWait(1500)
        bars = [b for b in w.findChildren(InfoBar) if id(b) not in before and b.isVisible()]
        texts = [(b.title, b.content) for b in bars]
        w.grab().save(str(Path(tempfile.gettempdir()) / "verify_issue12_model_error.png"))
        check("窗口里出现「モデルエラー」提示", any(t == "モデルエラー" and "weights/best.pt" in c for t, c in texts), str(texts))
        QTest.qWait(3500)
        error_bars = [b for b in bars if b.title == "モデルエラー"]
        check("4 秒后提示仍在（不自动消失）", len(error_bars) == 1 and error_bars[0].isVisible())
        check("提示期间监控照常进行（画面与分数持续更新）", len(scores) > 50, f"{len(scores)} 次分数更新")
        if error_bars:
            QTest.mouseClick(error_bars[0].closeButton, 1)   # 真实点击关闭按钮
            QTest.qWait(800)
            left = [b for b in w.findChildren(InfoBar) if b.title == "モデルエラー" and b.isVisible()]
            check("点击关闭按钮后提示消失", left == [])
        QTest.mouseClick(m.stopBtn, 1)
        QTest.qWait(1500)
    w.close()
    QTest.qWait(500)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        main(Path(d) / "start.db")
    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILED: {failures}", flush=True)
    os._exit(1 if failures else 0)  # 跳过解释器清理：PyQt 退出清理偶发 SIGSEGV（#14）
