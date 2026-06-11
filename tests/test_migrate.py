import sqlite3

from core import store
from migrate import replay


def _make_old_db(path):
    old = sqlite3.connect(path)
    old.execute("""CREATE TABLE sources (
        id TEXT PRIMARY KEY, title TEXT, source_type TEXT, created_at TEXT,
        n_fragments INTEGER, essence TEXT, raw_text TEXT, status TEXT)""")
    rows = [
        ("src1", "note one", "note", "2024-01-01T10:00:00",
         "Habits compound slowly and then suddenly, which is why consistency beats intensity over long horizons."),
        ("src2", "note two", "note", "2024-02-01T10:00:00",
         "Consistency beats intensity because habits compound slowly over long time horizons before paying off suddenly."),
        ("src3", "empty note", "note", "2024-03-01T10:00:00", "   "),
    ]
    for rid, title, st, ts, raw in rows:
        old.execute("INSERT INTO sources (id, title, source_type, created_at, raw_text) VALUES (?,?,?,?,?)",
                    (rid, title, st, ts, raw))
    old.commit()
    old.close()


def test_replay_counts_and_idempotency(tmp_path):
    old_db = tmp_path / "old.db"
    new_db = tmp_path / "engine.db"
    _make_old_db(old_db)

    s1 = replay(old_db=str(old_db), db_path=new_db, verbose=False)
    assert s1["replayed"] == 2          # empty raw_text skipped by the query
    assert s1["old_sources_total"] == 3
    assert s1["old_sources_nonempty"] == 2
    assert s1["episodes"] == 2

    s2 = replay(old_db=str(old_db), db_path=new_db, verbose=False)
    assert s2["replayed"] == 0
    assert s2["skipped"] == 2
    assert s2["episodes"] == 2          # idempotent


def test_replay_preserves_original_ts_and_order(tmp_path):
    old_db = tmp_path / "old.db"
    new_db = tmp_path / "engine.db"
    _make_old_db(old_db)
    replay(old_db=str(old_db), db_path=new_db, verbose=False)

    conn = store.connect(new_db)
    eps = conn.execute("SELECT ts, id FROM episodes ORDER BY ts").fetchall()
    assert eps[0]["ts"] == "2024-01-01T10:00:00"
    assert eps[1]["ts"] == "2024-02-01T10:00:00"
    assert eps[0]["id"] < eps[1]["id"]  # ULIDs sort chronologically


def test_replay_receipt_sees_prior_note(tmp_path):
    old_db = tmp_path / "old.db"
    new_db = tmp_path / "engine.db"
    _make_old_db(old_db)
    replay(old_db=str(old_db), db_path=new_db, verbose=False)

    conn = store.connect(new_db)
    import json
    second = conn.execute(
        "SELECT receipt_json FROM episodes ORDER BY ts LIMIT 1 OFFSET 1").fetchone()
    receipt = json.loads(second["receipt_json"])
    assert receipt["prior_episode_matches"], "second note should echo the first"
