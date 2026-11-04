"""同一数据库只允许一个程序实例（#13）：后启动的实例把已开的窗口调到前面后退出"""
import hashlib
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QLockFile, QObject, pyqtSignal
from PyQt5.QtNetwork import QLocalServer, QLocalSocket

from app import session_store


class SingleInstance(QObject):
    """
    锁文件放在数据库旁（<db>.lock），每个数据库各自一把锁。持锁进程崩溃或被强制结束后，
    QLockFile 按 PID（及 flock）判定锁已失效，下一个实例可以直接接管。
    持锁的实例开一个本地套接字，后启动的实例连上它表示「请把窗口调到前面」。
    """
    activated = pyqtSignal()

    def __init__(self, db_path: Optional[session_store.PathLike] = None):
        super().__init__()
        db_path = Path(db_path if db_path is not None else session_store.DEFAULT_DB_PATH).resolve()
        self.lock_path = db_path.with_name(db_path.name + ".lock")
        self.server_name = "self-analysis-" + hashlib.sha1(str(db_path).encode()).hexdigest()[:16]
        # 保持默认的 staleLockTime（30 秒）：运行中实例的锁超过 30 秒也会被判为「疑似失效」，但 Unix 上删除前
        # 还会确认没有进程对它持有 flock，持锁进程还在就删不掉。设成 0 反而会让主机名变化（换网络）后崩溃留下的锁永远无法接管
        self._lock = QLockFile(str(self.lock_path))
        self._server: Optional[QLocalServer] = None

    def acquire(self) -> bool:
        """
        返回 False 表示另一个实例正在运行。锁文件建不了（如目录无写权限）时照常启动，与加锁前的行为一致。
        需在 QApplication 创建之后调用（锁文件里记录程序路径，用于识别 PID 被复用）。
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._lock.tryLock(0):
            if self._lock.error() == QLockFile.LockFailedError:
                return False
            print(f"⚠️ 无法创建锁文件 {self.lock_path}（错误 {self._lock.error()}），不检查是否已有实例在运行")
            return True
        QLocalServer.removeServer(self.server_name)   # 上一个实例崩溃时留下的套接字文件
        self._server = QLocalServer(self)
        if self._server.listen(self.server_name):
            self._server.newConnection.connect(self._on_connection)
        else:
            print(f"⚠️ 无法监听本地套接字 {self.server_name}：{self._server.errorString()}（再次启动时无法调出窗口）")
        return True

    def _on_connection(self) -> None:
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            socket.disconnected.connect(socket.deleteLater)
            socket.disconnectFromServer()
        self.activated.emit()

    def notify_running(self, timeout_s: float = 3.0) -> bool:
        """通知持锁的实例调出窗口。对方刚拿到锁、还没开始监听时会稍等重试。"""
        deadline = time.monotonic() + timeout_s
        while True:
            socket = QLocalSocket()
            socket.connectToServer(self.server_name)
            if socket.waitForConnected(500):
                socket.disconnectFromServer()
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def release(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._lock.unlock()
