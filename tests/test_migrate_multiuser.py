"""AUTH.md §5: one-shot migration of the legacy single-user engine.db."""
import sqlite3

import sqlite_vec
from sqlite_vec import serialize_float32

from core import config, store
from core.auth import check_password
from migrate_multiuser import migrate
from tests.test_consolidate import S1


def _make_legacy_db(path):
    """A minimal pre-multi-user engine.db: v1 schema + one consolidated note."""
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript("""
    CREATE TABLE episodes (id TEXT PRIMARY KEY, ts TEXT NOT NULL,
        raw_text TEXT NOT NULL, title TEXT, source TEXT, receipt_json TEXT);
    CREATE TABLE episode_sentences (episode_id TEXT NOT NULL, idx INTEGER NOT NULL,
        text TEXT NOT NULL, PRIMARY KEY (episode_id, idx));
    CREATE TABLE claims (id TEXT PRIMARY KEY, text TEXT NOT NULL,
        strength REAL DEFAULT 1.0, created_at TEXT, last_seen TEXT);
    CREATE TABLE claim_support (claim_id TEXT NOT NULL, episode_id TEXT NOT NULL,
        verbatim_sentence TEXT, PRIMARY KEY (claim_id, episode_id));
    CREATE TABLE concepts (id TEXT PRIMARY KEY, label TEXT, canonical TEXT,
        state TEXT DEFAULT 'active', strength REAL DEFAULT 1.0,
        created_at TEXT, last_activity TEXT);
    CREATE TABLE concept_members (concept_id TEXT NOT NULL, claim_id TEXT NOT NULL,
        weight REAL DEFAULT 1.0, PRIMARY KEY (concept_id, claim_id));
    CREATE TABLE relations (from_id TEXT NOT NULL, to_id TEXT NOT NULL,
        relation TEXT NOT NULL, weight REAL DEFAULT 1.0, created_at TEXT,
        evidence_episode_id TEXT, PRIMARY KEY (from_id, to_id, relation));
    CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        run_id TEXT, type TEXT NOT NULL, payload_json TEXT NOT NULL);
    CREATE TABLE consolidation_runs (id TEXT PRIMARY KEY, started_at TEXT,
        finished_at TEXT, n_episodes INTEGER, batch_id TEXT, status TEXT,
        cost_estimate REAL);
    CREATE TABLE episode_consolidations (episode_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL, consolidated_at TEXT NOT NULL);
    CREATE TABLE replay_map (old_source_id TEXT PRIMARY KEY,
        episode_id TEXT NOT NULL, replayed_at TEXT NOT NULL);
    """)
    dim = config.EMBED_DIM
    conn.execute(f"CREATE VIRTUAL TABLE vec_sentences USING vec0(sent_key TEXT PRIMARY KEY, embedding FLOAT[{dim}])")
    conn.execute(f"CREATE VIRTUAL TABLE vec_claims USING vec0(claim_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])")
    conn.execute(f"CREATE VIRTUAL TABLE vec_concepts USING vec0(concept_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])")
    conn.execute("CREATE VIRTUAL TABLE episodes_fts USING fts5(episode_id UNINDEXED, title, raw_text)")

    ts = "2026-01-01T00:00:00+00:00"
    vec = serialize_float32([1.0] + [0.0] * (dim - 1))
    conn.execute("INSERT INTO episodes VALUES ('ep_1', ?, ?, 'old note', 'mcp', '{}')", (ts, S1))
    conn.execute("INSERT INTO episode_sentences VALUES ('ep_1', 0, ?)", (S1,))
    conn.execute("INSERT INTO vec_sentences VALUES ('ep_1:0', ?)", (vec,))
    conn.execute("INSERT INTO episodes_fts VALUES ('ep_1', 'old note', ?)", (S1,))
    conn.execute("INSERT INTO claims VALUES ('clm_x', ?, 1.5, ?, ?)", (S1, ts, ts))
    conn.execute("INSERT INTO vec_claims VALUES ('clm_x', ?)", (vec,))
    conn.execute("INSERT INTO claim_support VALUES ('clm_x', 'ep_1', ?)", (S1,))
    conn.execute("INSERT INTO concepts VALUES ('cpt_x', 'Memory', 'm', 'active', 1.0, ?, ?)", (ts, ts))
    conn.execute("INSERT INTO concept_members VALUES ('cpt_x', 'clm_x', 1.0)")
    conn.execute("INSERT INTO vec_concepts VALUES ('cpt_x', ?)", (vec,))
    conn.execute("INSERT INTO relations VALUES ('cpt_x', 'cpt_x2', 'bridges', 0.6, ?, NULL)", (ts,))
    conn.execute("INSERT INTO events (ts, run_id, type, payload_json) VALUES (?, 'run_1', 'ENCODED', '{}')", (ts,))
    conn.execute("INSERT INTO consolidation_runs VALUES ('run_1', ?, ?, 1, NULL, 'ok', 0.1)", (ts, ts))
    conn.execute("INSERT INTO episode_consolidations VALUES ('ep_1', 'run_1', ?)", (ts,))
    conn.execute("INSERT INTO replay_map VALUES ('src_1', 'ep_1', ?)", (ts,))
    conn.commit()
    conn.close()


def test_connect_refuses_then_migrate_fixes(tmp_path):
    import pytest
    db = tmp_path / "engine.db"
    _make_legacy_db(db)

    with pytest.raises(RuntimeError, match="legacy"):
        store.connect(db)

    result = migrate(db, "vardan", "first-pass", dry_run=False)
    assert result["episodes"] == 1
    assert result["events"] == 1
    assert (tmp_path / "engine.pre-multiuser.bak").exists()

    conn = store.connect(db)  # accepts the migrated file
    uid = result["user_id"]

    # user row works for login
    user = store.get_user_by_username(conn, "vardan")
    assert user["id"] == uid and user["is_admin"] == 1
    assert check_password("first-pass", user["pw_hash"])

    # every copied row owned by the first user
    for table in ("episodes", "claims", "concepts", "events", "replay_map"):
        owners = {r["user_id"] for r in
                  conn.execute(f"SELECT user_id FROM {table}")}
        assert owners == {uid}, table

    # vec partitions + kNN work post-migration
    emb = [1.0] + [0.0] * (config.EMBED_DIM - 1)
    assert store.knn_claims(conn, uid, emb, k=1)[0]["claim_id"] == "clm_x"
    assert store.knn_claims(conn, "usr_other", emb, k=1) == []

    # FTS rebuilt with user_id
    assert conn.execute(
        "SELECT COUNT(*) AS n FROM episodes_fts WHERE user_id = ?",
        (uid,)).fetchone()["n"] == 1

    # bookkeeping intact: nothing pending consolidation
    assert store.unconsolidated_episodes(conn, uid) == []
    assert store.last_run(conn, uid)["status"] == "ok"


def test_migrate_dry_run_leaves_original(tmp_path):
    db = tmp_path / "engine.db"
    _make_legacy_db(db)
    result = migrate(db, "vardan", "pw", dry_run=True)
    assert result["dry_run"] is True
    assert not (tmp_path / "engine.new").exists()
    # original untouched and still legacy
    import pytest
    with pytest.raises(RuntimeError, match="legacy"):
        store.connect(db)


def test_migrate_refuses_already_migrated(tmp_path):
    import pytest
    db = tmp_path / "engine.db"
    store.connect(db).close()  # fresh multi-user DB
    with pytest.raises(SystemExit, match="already multi-user"):
        migrate(db, "vardan", "pw")
