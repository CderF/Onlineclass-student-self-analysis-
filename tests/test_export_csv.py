"""CSV 导出（#11）：app/export_csv.py、ReportBridge 的导出槽与「記録の管理」页面"""
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pytest

from app import export_csv, report_data, session_store
from app.export_csv import ExportError, export, iso, wall_at
from tests.test_record_management import click_action, open_manage, table, with_pause
from tests.test_report_interface import conn, db, open_report, run_js, wait_for  # noqa: F401
from tests.test_report_summary import lesson
from tests.test_session_store import T0

FILES = ["events.csv", "pauses.csv", "problems.csv", "samples.csv", "sessions.csv"]


def read(path):
    raw = Path(path).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")      # UTF-8 BOM
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def num(v):
    return None if v == "" else float(v)


@pytest.fixture
def out(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


# ---------- 时间格式 ----------

def test_iso_has_local_offset():
    text = iso(T0 + 0.25)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.250[+-]\d\d:\d\d", text)
    assert datetime.fromisoformat(text).timestamp() == pytest.approx(T0 + 0.25)
    assert iso(None) is None


@pytest.mark.parametrize("t, wall", [
    (0, 100.0), (2.5, 102.5), (3.0, 103.0), (3.9, 103.9),   # 暂停前最后一秒内
    (4.0, 500.0), (5.25, 501.25), (6.0, 502.0),              # 继续后（t=4 起墙上时间跳到 500）
])
def test_wall_at(t, wall):
    ts, walls = [0, 1, 2, 3, 4, 5], [100.0, 101.0, 102.0, 103.0, 500.0, 501.0]
    assert wall_at(ts, walls, t) == pytest.approx(wall)


def test_wall_at_without_samples():
    assert wall_at([], [], 3.0) is None


# ---------- 导出内容 ----------

def test_export_all_tables_match_db(db, conn, out):
    a = with_pause(db, conn, T0, name="線形代数")
    b = lesson(db, conn, T0 + 1000, 120, name=None, away=[(10, 40)], doubt=[(50, 110)])
    res = export(conn, out, now=T0)
    folder = Path(res["dir"])
    assert folder.parent == out and folder.name == f"self_analysis_export_{datetime.fromtimestamp(T0):%Y%m%d_%H%M%S}"
    assert sorted(p.name for p in folder.iterdir()) == FILES
    assert [p.name for p in out.iterdir()] == [folder.name]   # 没有留下临时文件夹
    assert res["sessions"] == 2

    cols, sessions = read(folder / "sessions.csv")
    assert cols == export_csv.SESSION_COLUMNS
    assert [int(r["session_id"]) for r in sessions] == [a, b]      # 按开始时间升序
    by_id = {r["session_id"]: r for r in sessions}
    db_s = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM sessions")}
    for sid in (a, b):
        r, d = by_id[str(sid)], db_s[sid]
        assert r["course_name"] == (d["course_name"] or "") and r["status"] == d["status"]
        assert float(r["started_at"]) == d["started_at"] and float(r["ended_at"]) == d["ended_at"]
        assert datetime.fromisoformat(r["started_at_iso"]).timestamp() == pytest.approx(d["started_at"], abs=1e-3)
        summary = report_data.load_session(conn, sid)["session"]
        assert int(r["listening_seconds"]) == summary["seconds"] and float(r["avg_score"]) == summary["avg_score"]

    cols, samples = read(folder / "samples.csv")
    assert cols == export_csv.SAMPLE_COLUMNS
    db_rows = conn.execute("SELECT * FROM samples ORDER BY session_id, t").fetchall()
    assert len(samples) == len(db_rows) == res["rows"]["samples"]
    for r, d in zip(samples, db_rows):
        assert int(r["session_id"]) == d["session_id"] and int(r["t"]) == d["t"]
        for c in ("wall_time", "score", "fatigue_index", "face_ratio", "delta_pitch", "delta_yaw"):
            assert num(r[c]) == d[c]                  # 浮点数原样往返
        assert (r["cognitive_state"] or None) == d["cognitive_state"] and r["spatial_state"] == d["spatial_state"]
        assert int(r["frame_count"]) == d["frame_count"]
    assert {r["course_name"] for r in samples if r["session_id"] == str(a)} == {"線形代数"}

    cols, events = read(folder / "events.csv")
    assert cols == export_csv.EVENT_COLUMNS
    db_ev = conn.execute("SELECT * FROM events ORDER BY session_id, t_start, id").fetchall()
    assert [(int(r["session_id"]), r["kind"], r["type"], float(r["t_start"]), num(r["t_end"]), float(r["wall_start"]))
            for r in events] == [(d["session_id"], d["kind"], d["type"], d["t_start"], d["t_end"], d["wall_start"]) for d in db_ev]
    assert {r["kind"] for r in events} == {"spatial", "alert"}

    cols, pauses = read(folder / "pauses.csv")
    assert cols == export_csv.PAUSE_COLUMNS
    (p,) = pauses
    d = conn.execute("SELECT * FROM pauses").fetchone()
    assert (int(p["session_id"]), float(p["t"]), float(p["paused_at"]), float(p["resumed_at"])) == (a, d["t"], d["paused_at"], d["resumed_at"])
    assert p["resumed_at_iso"] and p["paused_at_iso"]


def test_problems_match_report_and_map_to_wall_time(db, conn, out):
    sid = lesson(db, conn, T0, 300, name="英語", away=[(100, 160)], doubt=[(0, 90)])
    folder = Path(export(conn, out)["dir"])
    cols, rows = read(folder / "problems.csv")
    assert cols == export_csv.PROBLEM_COLUMNS
    detected = report_data.load_session(conn, sid)["problems"]
    assert len(detected) >= 2
    assert [(r["category"], float(r["start"]), float(r["end"])) for r in rows] == \
           [(p["category"], p["start"], p["end"]) for p in detected]
    for r in rows:
        # lesson() 每秒一帧、t 秒的帧在墙上时间 T0 + t
        assert float(r["wall_start"]) == pytest.approx(T0 + float(r["start"]))
        assert float(r["wall_end"]) == pytest.approx(T0 + float(r["end"]))
        assert float(r["duration"]) == pytest.approx(float(r["end"]) - float(r["start"]))


def test_problem_after_pause_uses_resumed_wall_time(db, conn, out):
    """暂停 10 分钟后的离席时段：墙上时间要跳过暂停"""
    from app.session_recorder import SessionRecorder
    from tests.test_session_store import frame
    sid = session_store.create_session(conn, T0)
    rec = SessionRecorder(sid, T0, db)
    for i in range(30):
        rec.add_frame(frame(), T0 + i)
    rec.pause(T0 + 30)
    rec = SessionRecorder(sid, T0 + 630, db)      # 继续后 t 从 30 开始
    for i in range(60):
        away = 20 <= i < 50
        rec.add_frame(frame(spatial="away" if away else "normal", has_face=not away), T0 + 630 + i)
    rec.close(T0 + 690)
    session_store.finish_session(conn, sid, T0 + 690)
    _, rows = read(Path(export(conn, out)["dir"]) / "problems.csv")
    (p,) = [r for r in rows if r["category"] == "distraction"]
    assert (float(p["start"]), float(p["end"])) == (50.0, 80.0)
    assert float(p["wall_start"]) == pytest.approx(T0 + 650) and float(p["wall_end"]) == pytest.approx(T0 + 680)


def test_recording_sessions_skipped_in_all(db, conn, out):
    done = lesson(db, conn, T0, 30, name="英語")
    lesson(db, conn, T0 + 100, 30, name="統計学", finish=False)
    res = export(conn, out)
    _, sessions = read(Path(res["dir"]) / "sessions.csv")
    assert [int(r["session_id"]) for r in sessions] == [done] and res["sessions"] == 1
    _, samples = read(Path(res["dir"]) / "samples.csv")
    assert {r["session_id"] for r in samples} == {str(done)}


def test_single_session_and_errors(db, conn, out):
    a = lesson(db, conn, T0, 30, name="英語")
    lesson(db, conn, T0 + 100, 30)
    rec = lesson(db, conn, T0 + 200, 30, finish=False)
    res = export(conn, out, [a], now=T0)
    assert Path(res["dir"]).name.endswith(f"_session{a}") and res["sessions"] == 1
    _, sessions = read(Path(res["dir"]) / "sessions.csv")
    assert [int(r["session_id"]) for r in sessions] == [a]
    with pytest.raises(ExportError, match="记录中"):
        export(conn, out, [rec])
    with pytest.raises(ExportError, match="不存在"):
        export(conn, out, [a, 999])
    assert len(list(out.iterdir())) == 1   # 失败时没有新建文件夹


def test_nothing_to_export(db, conn, out):
    with pytest.raises(ExportError):
        export(conn, out)
    lesson(db, conn, T0, 30, finish=False)
    with pytest.raises(ExportError):
        export(conn, out)
    assert list(out.iterdir()) == []


def test_same_second_exports_get_unique_folders(db, conn, out):
    lesson(db, conn, T0, 10)
    first = export(conn, out, now=T0)["dir"]
    second = export(conn, out, now=T0)["dir"]
    assert second == first + "_2" and Path(second).is_dir()


@pytest.mark.parametrize("name", ['線形代数, 第3回', 'He said "hi"', "改行\nあり", "=1+1"])
def test_course_name_round_trips(db, conn, out, name):
    lesson(db, conn, T0, 5, name=name)
    _, rows = read(Path(export(conn, out)["dir"]) / "sessions.csv")
    assert rows[0]["course_name"] == name


def test_failure_leaves_nothing(db, conn, out, monkeypatch):
    lesson(db, conn, T0, 10)
    calls = []
    real = export_csv._write

    def flaky(path, columns, rows):
        calls.append(path.name)
        if len(calls) == 3:
            raise OSError("disk full")
        real(path, columns, rows)

    monkeypatch.setattr(export_csv, "_write", flaky)
    with pytest.raises(OSError):
        export(conn, out)
    assert list(out.iterdir()) == []


def test_readable_with_pandas_and_joinable(db, conn, out):
    pd = pytest.importorskip("pandas")
    with_pause(db, conn, T0, name="線形代数")
    lesson(db, conn, T0 + 1000, 60, name="英語")
    folder = Path(export(conn, out)["dir"])
    sessions = pd.read_csv(folder / "sessions.csv", encoding="utf-8-sig")
    samples = pd.read_csv(folder / "samples.csv", encoding="utf-8-sig")
    merged = samples.merge(sessions[["session_id", "avg_score"]], on="session_id")
    assert len(merged) == len(samples) == 90
    per = samples.groupby("session_id")["score"].mean()
    assert per.to_dict() == pytest.approx(sessions.set_index("session_id")["avg_score"].to_dict())


def test_no_image_columns():
    columns = (export_csv.SESSION_COLUMNS + export_csv.SAMPLE_COLUMNS + export_csv.EVENT_COLUMNS
               + export_csv.PAUSE_COLUMNS + export_csv.PROBLEM_COLUMNS)
    assert not [c for c in columns if re.search(r"image|pixel|frame(?!_count)|jpg|png|video", c)]


# ---------- ReportBridge ----------

@pytest.fixture
def bridge(qapp, db):
    from app.view.report_interface import ReportBridge
    b = ReportBridge(db_path=db)
    b.starts = []
    return b


def choose(bridge, folder):
    if not hasattr(bridge, "starts"):
        bridge.starts = []
    bridge.choose_export_dir = lambda start: bridge.starts.append(start) or (str(folder) if folder else None)


def test_bridge_export_session_and_remembers_folder(bridge, db, conn, out):
    from PyQt5.QtCore import QStandardPaths
    sid = lesson(db, conn, T0, 30, name="英語")
    choose(bridge, out)
    res = json.loads(bridge.export_session(sid))
    assert res["error"] is None and res["data"]["done"] is True and res["data"]["sessions"] == 1
    assert Path(res["data"]["dir"]).parent == out and (Path(res["data"]["dir"]) / "samples.csv").exists()
    assert bridge.starts == [QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)]
    json.loads(bridge.export_all())
    assert bridge.starts[1] == str(out)       # 下次从上次的位置开始


def test_bridge_cancel_writes_nothing(bridge, db, conn, out):
    sid = lesson(db, conn, T0, 30)
    choose(bridge, None)
    assert json.loads(bridge.export_session(sid))["data"] == {"done": False, "dir": None, "sessions": 0}
    assert json.loads(bridge.export_all())["data"]["done"] is False
    assert list(out.iterdir()) == []


def test_bridge_refuses_without_dialog(bridge, db, conn, out):
    choose(bridge, out)
    assert json.loads(bridge.export_all())["error"]          # 没有记录
    rec = lesson(db, conn, T0, 30, finish=False)
    assert json.loads(bridge.export_all())["error"]          # 只有记录中的课
    assert json.loads(bridge.export_session(rec))["error"]
    assert json.loads(bridge.export_session(rec + 100))["error"]
    assert bridge.starts == [] and list(out.iterdir()) == []


def test_bridge_unwritable_folder_reports_error(bridge, db, conn, tmp_path):
    lesson(db, conn, T0, 30)
    choose(bridge, tmp_path / "does-not-exist")
    res = json.loads(bridge.export_all())
    assert res["data"] is None and res["error"]


# ---------- 页面 ----------

def notice(report):
    return json.loads(run_js(report.page, """JSON.stringify({
        hidden: document.getElementById('manage-message').hidden,
        tone: document.getElementById('manage-message').dataset.tone,
        text: document.getElementById('manage-message').textContent,
    })"""))


def press_export_all(report):
    run_js(report.page, "document.body.dataset.manage = 'stale'; document.getElementById('export-all').click()")
    wait_for(report.page, "document.body.dataset.manage", "idle")


def test_page_export_one_and_all(db, conn, out, open_report):
    a = lesson(db, conn, T0, 30, name="英語")
    lesson(db, conn, T0 + 100, 30, name="統計学")
    lesson(db, conn, T0 + 200, 30, finish=False)
    report = open_report()
    choose(report.bridge, out)
    open_manage(report)
    assert run_js(report.page, "document.getElementById('export-all').disabled") is False
    run_js(report.page, "document.body.dataset.manage = 'stale'")
    click_action(report, a, "export")
    n = notice(report)
    folder = next(out.iterdir())
    assert n["hidden"] is False and n["tone"] == "ok"
    assert n["text"] == f"1 件の記録を CSV で書き出しました：{folder}"
    assert run_js(report.page, "document.getElementById('view-manage').hidden") is False

    press_export_all(report)
    n = notice(report)
    assert n["text"].startswith("2 件の記録を CSV で書き出しました：")    # 记录中的课不导出
    assert len(list(out.iterdir())) == 2


def test_page_export_all_disabled_when_only_recording(db, conn, open_report):
    lesson(db, conn, T0, 30, finish=False)
    report = open_report()
    open_manage(report)
    assert run_js(report.page, "document.getElementById('export-all').disabled") is True
    assert table(report)[0]["buttons"][0][:3] == ["export", "CSV", True]


def test_page_export_cancel_and_error(db, conn, out, tmp_path, open_report):
    sid = lesson(db, conn, T0, 30)
    report = open_report()
    open_manage(report)
    choose(report.bridge, None)
    click_action(report, sid, "export")
    assert notice(report)["hidden"] is True and list(out.iterdir()) == []
    choose(report.bridge, tmp_path / "missing")
    click_action(report, sid, "export")
    n = notice(report)
    assert n["hidden"] is False and n["tone"] == "error" and "書き出せませんでした" in n["text"]
    choose(report.bridge, out)
    press_export_all(report)
    assert notice(report)["tone"] == "ok"
