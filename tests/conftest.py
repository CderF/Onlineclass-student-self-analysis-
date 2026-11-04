import os
import sys
from pathlib import Path

import pytest

# 无界面运行 Qt；必须在导入 PyQt5 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# ExpressionClassifier 以相对路径加载 weights/best.pt，测试统一在仓库根目录运行
os.chdir(ROOT)


@pytest.fixture(scope="session")
def qapp():
    import PyQt5.QtWebEngineWidgets  # noqa: F401  必须在创建 QApplication 之前导入
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """所有测试使用临时数据库，绝不写入仓库的 data/"""
    from app import session_store
    path = tmp_path / "data" / "self_analysis.db"
    monkeypatch.setattr(session_store, "DEFAULT_DB_PATH", path)
    return path


HAS_WEIGHTS = (ROOT / "weights" / "best.pt").exists()
needs_weights = pytest.mark.skipif(not HAS_WEIGHTS, reason="需要本地权重 weights/best.pt")


class FakeCapture:
    """
    替代 cv2.VideoCapture：约 30 fps 返回人脸近景；show_face=False 时返回无人脸的黑画面。
    reopen_delay：第二次及以后打开（暂停后继续）时，首帧前等待的秒数，模拟摄像头预热
    """
    instances = []
    show_face = True
    can_open = True
    reopen_delay = 0.0
    _face = None

    def __init__(self, index):
        self.open = FakeCapture.can_open
        self.first_delay = FakeCapture.reopen_delay if FakeCapture.instances else 0.0
        FakeCapture.instances.append(self)

    @classmethod
    def face(cls):
        if cls._face is None:
            import cv2
            import ultralytics
            img = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "zidane.jpg"))
            cls._face = img[0:400, 800:1200]
        return cls._face

    def isOpened(self):
        return self.open

    def read(self):
        import time
        import numpy as np
        time.sleep(0.03 + self.first_delay)
        self.first_delay = 0.0
        img = self.face()
        return True, (img.copy() if FakeCapture.show_face else np.zeros_like(img))

    def release(self):
        self.open = False


@pytest.fixture
def fake_camera(monkeypatch):
    import app.camera_thread as ct
    FakeCapture.instances = []
    FakeCapture.show_face = True
    FakeCapture.can_open = True
    FakeCapture.reopen_delay = 0.0
    monkeypatch.setattr(ct.cv2, "VideoCapture", FakeCapture)
    return FakeCapture
