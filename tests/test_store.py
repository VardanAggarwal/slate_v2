import time

from core import store


def test_schema_init_creates_all_tables(conn):
    tables = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
    for t in ("episodes", "episode_sentences", "claims", "claim_support",
              "concepts", "concept_members", "relations", "events",
              "consolidation_runs", "episode_consolidations", "replay_map",
              "vec_sentences", "vec_claims", "vec_concepts", "episodes_fts"):
        assert t in tables, f"missing table {t}"


def test_connect_is_idempotent(tmp_path):
    store.connect(tmp_path / "x.db").close()
    store.connect(tmp_path / "x.db").close()  # second init must not fail


def test_event_append_increments_seq(conn):
    s1 = store.append_event(conn, "ENCODED", {"a": 1})
    s2 = store.append_event(conn, "ENCODED", {"a": 2})
    assert s2 == s1 + 1
    rows = store.events_since(conn, 0, types=["ENCODED"])
    assert len(rows) == 2


def test_ulid_orders_by_timestamp():
    early = store.ulid(ts=1000000000.0)
    late = store.ulid(ts=2000000000.0)
    assert early < late
    assert len(early) == 26


def test_knn_on_empty_tables_returns_nothing(conn):
    emb = [0.0] * 383 + [1.0]
    assert store.knn_claims(conn, emb, k=3) == []
    assert store.knn_sentences(conn, emb, k=3) == []
