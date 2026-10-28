import json
from pathlib import Path
from typing import Any, Callable, List, Optional

from PyQt5.QtCore import QObject, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInfo, QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from app import report_data, session_store
from app.session_store import PathLike

WEB_DIR = Path(__file__).resolve().parent / "web"
LOCAL_SCHEMES = {"file", "qrc", "data"}


class ReportBridge(QObject):
    """
    通过 QWebChannel 暴露给页面的 Python 对象（页面里为 bridge）。
    槽在界面线程执行，每次调用用独立连接读库；返回 JSON 字符串 {"data": ..., "error": ...}。
    """
    data_changed_signal = pyqtSignal()  # 课程记录可能有变化，页面应重新读取

    def __init__(self, db_path: Optional[PathLike] = None, parent=None):
        super().__init__(parent)
        self.db_path = db_path  # None 表示使用 session_store.DEFAULT_DB_PATH

    def _respond(self, func: Callable, *args) -> str:
        try:
            conn = session_store.connect(self.db_path)
            try:
                data, error = func(conn, *args), None
            finally:
                conn.close()
        except Exception as e:
            print(f"❌ 报告页读取课程记录失败: {e}")
            data, error = None, str(e)
        return json.dumps({"data": data, "error": error}, ensure_ascii=False)

    @pyqtSlot(result=str)
    def list_sessions(self) -> str:
        return self._respond(report_data.list_sessions)

    @pyqtSlot(int, result=str)
    def load_session(self, session_id: int) -> str:
        return self._respond(report_data.load_session, session_id)


class LocalOnlyInterceptor(QWebEngineUrlRequestInterceptor):
    """报告页只允许加载本地资源，拦截一切网络请求（ADR 0001：离线可用、数据不外发）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.blocked: List[str] = []

    def interceptRequest(self, info: QWebEngineUrlRequestInfo) -> None:
        url = info.requestUrl()
        if url.scheme() not in LOCAL_SCHEMES:
            self.blocked.append(url.toString())
            info.block(True)


class ReportPage(QWebEnginePage):
    """把页面里的 console 输出转到终端，并记下 JS 错误（测试用）"""

    def __init__(self, profile: QWebEngineProfile, parent=None):
        super().__init__(profile, parent)
        self.js_errors: List[str] = []

    def javaScriptConsoleMessage(self, level, message, line, source):
        print(f"[report.js] {message} ({source}:{line})")
        if level == QWebEnginePage.ErrorMessageLevel:
            self.js_errors.append(message)


class ReportInterface(QWidget):
    """
    分析报告页：整页是本地 HTML（app/view/web/report.html），图表由 ECharts 绘制（ADR 0001）。
    每次切换到本页时通知页面重新读取课程记录。
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("report-interface")

        self.webView = QWebEngineView(self)
        self.profile = QWebEngineProfile()  # 无名称 = 不写磁盘缓存 / Cookie
        self.interceptor = LocalOnlyInterceptor(self.profile)
        self.profile.setUrlRequestInterceptor(self.interceptor)
        self.page = ReportPage(self.profile, self.webView)
        # profile 必须在页面之后释放（否则退出时崩溃）：作为页面的子对象，在页面析构完成后才被删除
        self.profile.setParent(self.page)
        self.page.setBackgroundColor(Qt.transparent)
        self.webView.setPage(self.page)
        self.webView.setContextMenuPolicy(Qt.NoContextMenu)

        self.bridge = ReportBridge(parent=self)
        self.channel = QWebChannel(self)
        self.channel.registerObject("bridge", self.bridge)
        self.page.setWebChannel(self.channel)

        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.webView)

        self.webView.load(QUrl.fromLocalFile(str(WEB_DIR / "report.html")))

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.bridge.data_changed_signal.emit()
