"""删除记录后数据库文件中不留残留（#17）：secure_delete 与删除后清空 WAL"""
import sqlite3
import struct
from pathlib import Path

import pytest

from app import session_store
from app.session_recorder import SessionRecorder
from tests.test_session_store import T0, frame

NAME = "ZZ削除される授業ZZ"
FATIGUE = 0.123456789012345      # SQLite 以 8 字节大端 IEEE 754 存 REAL
MARKS = (NAME.encode(), struct.pack(">d", FATIGUE))


@pytest.fixture
def db(tmp_path):
    return tmp_path / "data" / "test.db"


def residue(db):
    """数据库文件与 WAL 中标记字节出现的次数"""
    out = {}
    for suffix in ("", "-wal"):
        path = Path(str(db) + suffix)
        data = path.read_bytes() if path.exists() else b""
        out[suffix or "db"] = sum(data.count(m) for m in MARKS)
    return out


def secret_lesson(db, conn, start=T0, seconds=300):
    sid = session_store.create_session(conn, start)
    rec = SessionRecorder(sid, start, db)
    for i in range(seconds):
        rec.add_frame(frame(fatigue=FATIGUE), start + i)
    rec.close(start + seconds)
    session_store.finish_session(conn, sid, start + seconds)
    session_store.set_course_name(conn, sid, NAME)
    return sid


def keep_lesson(db, conn):
    sid = session_store.create_session(conn, T0 - 1000)
    rec = SessionRecorder(sid, T0 - 1000, db)
    for i in range(60):
        rec.add_frame(frame(), T0 - 1000 + i)
    rec.close(T0 - 940)
    session_store.finish_session(conn, sid, T0 - 940)
    session_store.set_course_name(conn, sid, "英語")
    return sid


def test_connect_enables_secure_delete(db):
    conn = session_store.connect(db)
    assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    conn.close()


def test_secure_delete_on_even_if_build_default_is_off(db, monkeypatch):
    """模拟编译时未开启 SECURE_DELETE 的环境：connect() 也会显式打开"""
    real = sqlite3.connect

    def off_by_default(*args, **kwargs):
        c = real(*args, **kwargs)
        c.execute("PRAGMA secure_delete = OFF")
        return c

    monkeypatch.setattr(session_store.sqlite3, "connect", off_by_default)
    conn = session_store.connect(db)
    assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    conn.close()


def test_delete_leaves_no_residue_while_connection_open(db):
    conn = session_store.connect(db)
    keep = keep_lesson(db, conn)
    sid = secret_lesson(db, conn)
    assert sum(residue(db).values()) > 0                      # 删除前确实写进了文件
    assert session_store.delete_session(conn, sid) is True
    assert residue(db) == {"db": 0, "-wal": 0}
    assert Path(str(db) + "-wal").stat().st_size == 0         # WAL 已截断
    assert [r[0] for r in conn.execute("SELECT course_name FROM sessions")] == ["英語"]
    assert conn.execute("SELECT COUNT(*) FROM samples WHERE session_id = ?", (keep,)).fetchone()[0] == 60
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()


@pytest.mark.parametrize("secure", ["OFF", "ON"])
def test_control_without_fix_leaves_residue(db, secure):
    """对照：不做 #17 的处理（关掉 secure_delete、删除后不清空 WAL）时能找到残留，确认本测试的检查方法有效"""
    conn = session_store.connect(db)
    conn.execute(f"PRAGMA secure_delete = {secure}")
    sid = secret_lesson(db, conn)
    with conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
    assert residue(db)["-wal"] > 0         # 未清空 WAL：写入时的旧页面仍在
    conn.close()                          # 最后一个连接关闭时写回主库并删除 WAL
    if secure == "OFF":
        assert residue(db)["db"] > 0       # 未开 secure_delete：主库的空闲空间里留有数据
    else:
        assert residue(db)["db"] == 0


def test_reader_holding_snapshot_delays_wal_cleanup(db, capsys):
    """另一个连接仍在读取旧快照时 WAL 清不掉；该连接结束、最后一个连接关闭后不再有残留"""
    conn = session_store.connect(db)
    sid = secret_lesson(db, conn)
    reader = session_store.connect(db)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM samples").fetchone()
    assert session_store.delete_session(conn, sid) is True
    assert "未能清空 WAL" in capsys.readouterr().out
    assert residue(db)["db"] == 0 and residue(db)["-wal"] > 0
    reader.rollback()
    assert session_store.truncate_wal(conn) is True
    assert residue(db) == {"db": 0, "-wal": 0}
    reader.close()
    conn.close()
    assert residue(db) == {"db": 0, "-wal": 0} and not Path(str(db) + "-wal").exists()


def test_concurrent_recorder_does_not_block_cleanup(db):
    """录制中（另一连接每秒写入、但不长期持有读快照）删除别的课：WAL 能清空，录制继续写入"""
    conn = session_store.connect(db)
    sid = secret_lesson(db, conn)
    live = session_store.create_session(conn, T0 + 5000)
    rec = SessionRecorder(live, T0 + 5000, db)
    for i in range(30):
        rec.add_frame(frame(), T0 + 5000 + i)
    assert session_store.delete_session(conn, sid) is True
    assert residue(db) == {"db": 0, "-wal": 0}
    for i in range(30, 60):
        rec.add_frame(frame(), T0 + 5000 + i)
    rec.close(T0 + 5060)
    assert conn.execute("SELECT COUNT(*) FROM samples WHERE session_id = ?", (live,)).fetchone()[0] == 60
    conn.close()


def test_nothing_deleted_skips_checkpoint(db, monkeypatch):
    conn = session_store.connect(db)
    calls = []
    monkeypatch.setattr(session_store, "truncate_wal", lambda c: calls.append(c) or True)
    assert session_store.delete_session(conn, 999) is False
    assert calls == []
    conn.close()
