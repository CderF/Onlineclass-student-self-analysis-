import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInfo, QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView
from PyQt5.QtWidgets import QVBoxLayout, QWidget
from qfluentwidgets import MessageBox

from app import report_data, session_store
from app.session_store import PathLike
from app.view.course_name_dialog import RenameCourseDialog

WEB_DIR = Path(__file__).resolve().parent / "web"
LOCAL_SCHEMES = {"file", "qrc", "data"}


class ReportBridge(QObject):
    """
    通过 QWebChannel 暴露给页面的 Python 对象（页面里为 bridge）。
    槽在界面线程执行，每次调用用独立连接读写库；返回 JSON 字符串 {"data": ..., "error": ...}。
    改名 / 删除（#10）在槽内弹出 Qt 对话框（模态），data 为 {"done": 是否执行了}；记录中的课拒绝操作。
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
            print(f"❌ 报告页访问课程记录失败: {e}")
            data, error = None, str(e)
        return json.dumps({"data": data, "error": error}, ensure_ascii=False)

    @pyqtSlot(result=str)
    def list_sessions(self) -> str:
        return self._respond(report_data.list_sessions)

    @pyqtSlot(int, result=str)
    def load_session(self, session_id: int) -> str:
        return self._respond(report_data.load_session, session_id)

    @pyqtSlot(result=str)
    def summarize_sessions(self) -> str:
        return self._respond(report_data.summarize_sessions)

    @pyqtSlot(int, result=str)
    def rename_session(self, session_id: int) -> str:
        return self._respond(self._rename, session_id)

    @pyqtSlot(int, result=str)
    def delete_session(self, session_id: int) -> str:
        return self._respond(self._delete, session_id)

    def _rename(self, conn: sqlite3.Connection, session_id: int) -> Dict[str, bool]:
        session = self._editable(conn, session_id)
        accepted, name = self.ask_course_name(session_store.list_course_names(conn), session["course_name"])
        return {"done": accepted and session_store.rename_session(conn, session_id, name)}

    def _delete(self, conn: sqlite3.Connection, session_id: int) -> Dict[str, bool]:
        session = self._editable(conn, session_id)
        return {"done": self.confirm_delete(session) and session_store.delete_session(conn, session_id)}

    @staticmethod
    def _editable(conn: sqlite3.Connection, session_id: int) -> Dict[str, Any]:
        row = conn.execute(
            "SELECT id, course_name, started_at, status FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise LookupError(f"课程记录 {session_id} 不存在")
        if row["status"] == "recording":
            raise PermissionError(f"课程记录 {session_id} 正在记录中，不能修改")
        return dict(row)

    def _dialog_parent(self) -> Optional[QWidget]:
        parent = self.parent()
        return parent.window() if isinstance(parent, QWidget) else None

    def ask_course_name(self, course_names: List[str], current: Optional[str]) -> Tuple[bool, Optional[str]]:
        """改课程名对话框；返回 (是否保存, 新课程名)。测试时替换"""
        dialog = RenameCourseDialog(course_names, current, self._dialog_parent())
        return (True, dialog.course_name()) if dialog.exec() else (False, None)

    def confirm_delete(self, session: Dict[str, Any]) -> bool:
        """删除确认对话框。测试时替换"""
        started = time.strftime("%Y/%m/%d %H:%M", time.localtime(session["started_at"]))
        box = MessageBox(
            "この記録を削除しますか？",
            f"{started}　{session['course_name'] or '授業名なし'}\n"
            "この授業の記録をすべて削除します。\n元に戻すことはできません。",
            self._dialog_parent(),
        )
        box.yesButton.setText("削除")
        box.cancelButton.setText("キャンセル")
        return bool(box.exec())


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
