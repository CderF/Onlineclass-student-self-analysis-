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
