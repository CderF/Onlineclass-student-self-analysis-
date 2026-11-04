import json
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Optional, Tuple

from PyQt5.QtCore import QObject, QPoint, QRect, QRunnable, QStandardPaths, QThreadPool, QUrl, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInfo, QWebEngineUrlRequestInterceptor
from PyQt5.QtWebEngineWidgets import QWebEnginePage, QWebEngineProfile, QWebEngineView
from PyQt5.QtWidgets import QFileDialog, QLabel, QSizePolicy, QVBoxLayout, QWidget
from qfluentwidgets import MessageBox

from app import export_csv, report_data, session_store
from app.session_store import PathLike
from app.view.course_name_dialog import RenameCourseDialog

WEB_DIR = Path(__file__).resolve().parent / "web"
LOCAL_SCHEMES = {"file", "qrc", "data"}


class _Relay(QObject):
    """把后台线程的结果送回界面线程（排队连接）"""
    finished = pyqtSignal(int, str)


class _Task(QRunnable):
    def __init__(self, work: Callable[[], str], deliver: Callable[[str], None]):
        super().__init__()
        self.work = work
        self.deliver = deliver

    def run(self) -> None:
        result = self.work()
        try:
            self.deliver(result)
        except RuntimeError:
            pass  # 报告页已销毁


class ReportBridge(QObject):
    """
    通过 QWebChannel 暴露给页面的 Python 对象（页面里为 bridge）。
    页面只调用 request(请求编号, 方法名, JSON 参数)，结果经 response_signal(请求编号, JSON) 送回（#16）。
    每个方法的结果为 JSON 字符串 {"data": ..., "error": ...}，每次调用用独立连接读写库：
      - 读取（list_sessions / load_session / summarize_sessions）在后台线程执行，不阻塞界面线程
      - 改名 / 删除（#10）在界面线程执行：在其中弹出 Qt 对话框（模态），data 为 {"done": 是否执行了}；记录中的课拒绝操作
      - CSV 导出（#11）在界面线程选择文件夹，写文件在后台线程，开始写时发出 export_started_signal；
        data 为 {"done", "dir", "sessions"}；同一时间只进行一个导出
    各方法也可以在 Python 里直接同步调用（测试用）。
    """
    data_changed_signal = pyqtSignal()  # 课程记录可能有变化，页面应重新读取
    response_signal = pyqtSignal(int, str)  # (请求编号, JSON)
    export_started_signal = pyqtSignal(int)  # 已选好文件夹、开始在后台写文件

    BACKGROUND = ("list_sessions", "load_session", "summarize_sessions")
    FOREGROUND = ("rename_session", "delete_session")
    EXPORTS = ("export_session", "export_all")

    def __init__(self, db_path: Optional[PathLike] = None, parent=None):
        super().__init__(parent)
        self.db_path = db_path  # None 表示使用 session_store.DEFAULT_DB_PATH
        self.export_dir: Optional[str] = None  # 上次导出选择的文件夹（本次运行内记住）
        self.export_request: Optional[int] = None  # 进行中的导出的请求编号
        self.closed = False  # shutdown 之后不再接受请求
        # 线程池先于 relay 创建：销毁时线程池先析构（等后台任务结束），relay 仍在
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self._relay = _Relay(self)
        self._relay.finished.connect(self._finish, Qt.QueuedConnection)

    @pyqtSlot(int, str, str)
    def request(self, request_id: int, method: str, args_json: str) -> None:
        args = json.loads(args_json or "[]")
        if self.closed:
            return
        if method in self.BACKGROUND:
            func = getattr(self, method)
            self.pool.start(_Task(lambda: func(*args), lambda result: self._relay.finished.emit(request_id, result)))
        elif method in self.FOREGROUND:
            self._relay.finished.emit(request_id, getattr(self, method)(*args))
        elif method in self.EXPORTS:
            self._start_export(request_id, [int(args[0])] if method == "export_session" else None)
        else:
            self._relay.finished.emit(request_id, self._error(f"未知的方法：{method}"))

    def _finish(self, request_id: int, result: str) -> None:
        if request_id == self.export_request:
            self.export_request = None
        self.response_signal.emit(request_id, result)

    def wait_for_background(self, msecs: int = -1) -> bool:
        """等待后台任务全部结束（waitForDone 期间释放 GIL，后台任务可以继续）"""
        return self.pool.waitForDone(msecs)

    @staticmethod
    def _error(message: str) -> str:
        print(f"❌ 报告页访问课程记录失败: {message}")
        return json.dumps({"data": None, "error": message}, ensure_ascii=False)

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

    def list_sessions(self) -> str:
        return self._respond(report_data.list_sessions)

    def load_session(self, session_id: int) -> str:
        return self._respond(report_data.load_session, int(session_id))

    def summarize_sessions(self) -> str:
        return self._respond(report_data.summarize_sessions)

    def rename_session(self, session_id: int) -> str:
        return self._respond(self._rename, int(session_id))

    def delete_session(self, session_id: int) -> str:
        return self._respond(self._delete, int(session_id))

    def export_session(self, session_id: int) -> str:
        """同步导出（测试用；页面经 request 走后台）"""
        return self._respond(self._export, [int(session_id)])

    def export_all(self) -> str:
        return self._respond(self._export, None)

    def _export(self, conn: sqlite3.Connection, session_ids: Optional[List[int]]) -> Dict[str, Any]:
        self._check_exportable(conn, session_ids)
        folder = self._choose_folder()
        if not folder:
            return {"done": False, "dir": None, "sessions": 0}
        return self._export_to(conn, folder, session_ids)

    def _start_export(self, request_id: int, session_ids: Optional[List[int]]) -> None:
        """界面线程：检查并选择文件夹；写文件交给后台线程"""
        reply = self._relay.finished.emit
        if self.export_request is not None:
            reply(request_id, self._error("已有导出正在进行"))
            return
        checked = json.loads(self._respond(self._check_exportable, session_ids))
        if checked["error"]:
            reply(request_id, json.dumps(checked, ensure_ascii=False))
            return
        folder = self._choose_folder()
        if not folder:
            reply(request_id, json.dumps({"data": {"done": False, "dir": None, "sessions": 0}, "error": None}))
            return
        self.export_request = request_id
        self.export_started_signal.emit(request_id)
        self.pool.start(_Task(lambda: self._respond(self._export_to, folder, session_ids),
                              lambda result: reply(request_id, result)))

    def _check_exportable(self, conn: sqlite3.Connection, session_ids: Optional[List[int]]) -> None:
        if session_ids is not None:
            self._editable(conn, session_ids[0])
        elif not export_csv.exportable_ids(conn):
            raise export_csv.ExportError("没有可导出的课程记录")

    def _choose_folder(self) -> Optional[str]:
        folder = self.choose_export_dir(self.export_dir or QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation))
        if folder:
            self.export_dir = folder
        return folder

    @staticmethod
    def _export_to(conn: sqlite3.Connection, folder: str, session_ids: Optional[List[int]]) -> Dict[str, Any]:
        result = export_csv.export(conn, Path(folder), session_ids)
        print(f"✅ 已导出 {result['sessions']} 节课的 CSV: {result['dir']}")
        return {"done": True, "dir": result["dir"], "sessions": result["sessions"]}

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

    def _page_frozen(self) -> ContextManager[None]:
        """遮罩对话框（qfluentwidgets，窗口内的子控件）显示期间，用截图替换网页（#18）"""
        freeze = getattr(self.parent(), "page_frozen", None)
        return freeze() if freeze else nullcontext()

    def ask_course_name(self, course_names: List[str], current: Optional[str]) -> Tuple[bool, Optional[str]]:
        """改课程名对话框；返回 (是否保存, 新课程名)。测试时替换"""
        dialog = RenameCourseDialog(course_names, current, self._dialog_parent())
        with self._page_frozen():
            accepted = dialog.exec()
        return (True, dialog.course_name()) if accepted else (False, None)

    def choose_export_dir(self, start: str) -> Optional[str]:
        """选择导出位置（在其中新建导出文件夹）；取消时返回 None。测试时替换"""
        return QFileDialog.getExistingDirectory(self._dialog_parent(), "CSV の書き出し先を選択", start) or None

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
        with self._page_frozen():
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

        # 弹出遮罩对话框期间代替网页显示的截图（见 page_frozen）
        self.snapshot = QLabel(self)
        self.snapshot.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.snapshot.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.snapshot.hide()

        self.vBoxLayout = QVBoxLayout(self)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.webView)
        self.vBoxLayout.addWidget(self.snapshot)

        self.webView.load(QUrl.fromLocalFile(str(WEB_DIR / "report.html")))

    def shutdown(self) -> None:
        """
        销毁前调用（MainWindow.closeEvent、测试）：不再接受请求，等后台任务结束（#16）。
        不能留到析构时再等：C++ 析构在持有 GIL 的情况下等线程池，后台任务拿不到 GIL，两者互相等待。
        """
        self.bridge.closed = True
        self.bridge.wait_for_background()

    @contextmanager
    def page_frozen(self) -> Iterator[None]:
        """
        期间用网页的截图代替网页（#18）。网页内容由 GPU 合成，会盖在同一窗口内的普通子控件之上，
        qfluentwidgets 的遮罩对话框（MessageBoxBase / MessageBox）因此被网页挡住；换成截图后遮罩与对话框正常显示在最上层。
        """
        # 从窗口截取网页所在区域：网页背景透明，单独截网页视图时透明处会变成黑色
        window = self.webView.window()
        self.snapshot.setPixmap(window.grab(QRect(self.webView.mapTo(window, QPoint(0, 0)), self.webView.size())))
        # 先隐藏再显示（恢复时相反）：两者不会同时可见，布局不会把高度分给两者
        self.webView.hide()
        self.snapshot.show()
        try:
            yield
        finally:
            self.snapshot.hide()
            self.snapshot.clear()
            self.webView.show()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.bridge.data_changed_signal.emit()
