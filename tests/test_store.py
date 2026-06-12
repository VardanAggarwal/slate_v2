from core import store
from tests.conftest import UID, UID_B


def test_schema_init_creates_all_tables(conn):
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for t in ("users", "episodes", "episode_sentences", "claims", "claim_support",
              "concepts", "concept_members", "relations", "events",
              "consolidation_runs", "episode_consolidations", "replay_map",
              "vec_sentences", "vec_claims", "vec_concepts", "episodes_fts",
              "oauth_clients", "oauth_access_tokens", "oauth_refresh_tokens"):
        assert t in tables, f"missing table {t}"


def test_every_table_carries_user_id(conn):
    for t in ("episodes", "episode_sentences", "claims", "claim_support",
              "concepts", "concept_members", "relations", "events",
              "consolidation_runs", "episode_consolidations", "replay_map"):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
        assert "user_id" in cols, f"{t} lacks user_id"


def test_connect_is_idempotent(tmp_path):
    store.connect(tmp_path / "x.db").close()
    store.connect(tmp_path / "x.db").close()  # second init must not fail


def test_connect_refuses_legacy_db(tmp_path):
    import sqlite3
    import pytest
    legacy = tmp_path / "legacy.db"
    raw = sqlite3.connect(legacy)
    raw.execute("CREATE TABLE episodes (id TEXT PRIMARY KEY, ts TEXT, raw_text TEXT)")
    raw.commit()
    raw.close()
    with pytest.raises(RuntimeError, match="legacy single-user"):
        store.connect(legacy)


def test_event_append_increments_seq(conn):
    s1 = store.append_event(conn, UID, "ENCODED", {"a": 1})
    s2 = store.append_event(conn, UID, "ENCODED", {"a": 2})
    assert s2 == s1 + 1
    rows = store.events_since(conn, UID, 0, types=["ENCODED"])
    assert len(rows) == 2


def test_events_since_scopes_to_user(conn):
    store.append_event(conn, UID, "ENCODED", {"a": 1})
    store.append_event(conn, UID_B, "ENCODED", {"b": 1})
    assert len(store.events_since(conn, UID, 0)) == 1
    assert len(store.events_since(conn, None, 0)) == 2  # rebuild path spans all


def test_ulid_orders_by_timestamp():
    early = store.ulid(ts=1000000000.0)
    late = store.ulid(ts=2000000000.0)
    assert early < late
    assert len(early) == 26


def test_knn_on_empty_tables_returns_nothing(conn):
    emb = [0.0] * 383 + [1.0]
    assert store.knn_claims(conn, UID, emb, k=3) == []
    assert store.knn_sentences(conn, UID, emb, k=3) == []


def test_user_crud(conn):
    uid = store.create_user(conn, "alice", "hash-a", is_admin=True)
    assert uid.startswith("usr_")
    assert store.get_user(conn, uid)["username"] == "alice"
    assert store.get_user_by_username(conn, "alice")["is_admin"] == 1
    assert store.count_users(conn) == 1
    assert store.set_user_password(conn, "alice", "hash-b")
    assert store.get_user_by_username(conn, "alice")["pw_hash"] == "hash-b"
    assert store.delete_user(conn, "alice")
    assert store.count_users(conn) == 0
    assert not store.delete_user(conn, "alice")
