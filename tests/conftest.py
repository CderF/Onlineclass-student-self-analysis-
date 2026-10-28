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
    """替代 cv2.VideoCapture：约 30 fps 返回人脸近景；show_face=False 时返回无人脸的黑画面"""
    instances = []
    show_face = True
    can_open = True
    _face = None

    def __init__(self, index):
        self.open = FakeCapture.can_open
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
        time.sleep(0.03)
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
    monkeypatch.setattr(ct.cv2, "VideoCapture", FakeCapture)
    return FakeCapture
