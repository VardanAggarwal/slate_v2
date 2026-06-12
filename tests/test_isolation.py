"""AUTH.md §7: user A must never surface user B's data — one leak test per
tool surface, plus vec-partition and FTS-scope tests.

Each test seeds DISTINCT corpora for two users, then drives user A's read
path with queries aimed straight at user B's content. Any hit on B's text is
a cross-user leak (the High-severity risk in AUTH.md §8).
"""
import pytest

from core import store
from core.consolidate import consolidate
from core.encode import encode
from core.recall import (assemble_context, get_claim, get_concept,
                         get_episode, list_episodes, recall)
from tests.conftest import UID, UID_B
from tests.test_consolidate import fake_llm  # noqa: F401

# Corpus A: memory/learning. Corpus B: deliberately unmistakable content.
A1 = "Spaced repetition is the most reliable way to retain knowledge over many years."
B1 = "Project Krakatoa's secret launch budget is forty million dollars hidden in the Q3 ledger."
B2 = "The Krakatoa launch depends on the unannounced partnership with Meridian Robotics."


@pytest.fixture
def two_corpora(conn, fake_llm):
    encode(conn, UID, A1, source="test", title="memory note")
    encode(conn, UID_B, B1, source="test", title="krakatoa budget")
    encode(conn, UID_B, B2, source="test", title="krakatoa partnership")
    consolidate(conn, UID)
    consolidate(conn, UID_B)
    return conn


def _flat(x) -> str:
    return str(x).lower()


def test_recall_never_leaks(two_corpora):
    # A queries for B's exact content — the strongest possible bait.
    hits = recall(two_corpora, UID, "Krakatoa secret launch budget", k=20)
    assert "krakatoa" not in _flat(hits)
    # B's own recall still finds it (the filter scopes, not breaks, search)
    hits_b = recall(two_corpora, UID_B, "Krakatoa secret launch budget", k=20)
    assert "krakatoa" in _flat(hits_b)


def test_assemble_context_never_leaks(two_corpora):
    # query aims at B's content but avoids B's marker words, since the
    # nothing-stored message echoes the topic verbatim
    md = assemble_context(two_corpora, UID, "the hidden project finances")
    assert "krakatoa" not in md.lower()
    assert "meridian" not in md.lower()
    # and B gets its own context back for the same query
    md_b = assemble_context(two_corpora, UID_B, "the hidden project finances")
    assert "krakatoa" in md_b.lower()


def test_get_note_scoped_by_owner(two_corpora):
    b_ep = two_corpora.execute(
        "SELECT id FROM episodes WHERE user_id = ?", (UID_B,)).fetchone()["id"]
    assert get_episode(two_corpora, UID, b_ep) is None       # A can't fetch B's note
    assert get_episode(two_corpora, UID_B, b_ep) is not None  # B can


def test_list_episodes_scoped(two_corpora):
    a_list = list_episodes(two_corpora, UID)
    assert len(a_list) == 1
    assert "krakatoa" not in _flat(a_list)


def test_get_concept_and_claim_scoped(two_corpora):
    b_concept = store.all_concepts(two_corpora, UID_B)[0]
    assert get_concept(two_corpora, UID, b_concept["id"]) is None
    b_claim = two_corpora.execute(
        "SELECT id FROM claims WHERE user_id = ?", (UID_B,)).fetchone()["id"]
    assert get_claim(two_corpora, UID, b_claim) is None
    assert store.get_claim(two_corpora, UID, b_claim) is None


def test_timeline_scoped(two_corpora):
    # The MCP timeline tool greps payload_json within the caller's events.
    b_concept = store.all_concepts(two_corpora, UID_B)[0]["id"]
    rows = two_corpora.execute(
        """SELECT payload_json FROM events
           WHERE user_id = ? AND payload_json LIKE ?""",
        (UID, f"%{b_concept}%")).fetchall()
    assert rows == []


def test_digest_scoped(two_corpora):
    from core.digest import digest
    md_a = digest(two_corpora, UID, since_hours=48)
    assert "krakatoa" not in md_a.lower()
    # each digest reflects only that user's consolidation events: A's lone
    # note made a 1-claim concept, B's two notes a 2-claim concept
    assert "(1 claims)" in md_a
    md_b = digest(two_corpora, UID_B, since_hours=48)
    assert "(2 claims)" in md_b


def test_stats_scoped(two_corpora):
    assert store.stats(two_corpora, UID)["episodes"] == 1
    assert store.stats(two_corpora, UID_B)["episodes"] == 2


def test_vec_partition_restricts_knn(two_corpora):
    """kNN over B's exact sentence embedding from A's partition: no B rows,
    even with k far larger than A's corpus (no cross-partition spill)."""
    from core.encode import get_embedder
    emb = get_embedder().encode([B1], normalize_embeddings=True,
                                show_progress_bar=False)[0]
    for hit in store.knn_sentences(two_corpora, UID, emb, k=50):
        assert "krakatoa" not in (hit["text"] or "").lower()
    for hit in store.knn_claims(two_corpora, UID, emb, k=50):
        assert "krakatoa" not in (hit["text"] or "").lower()
    for hit in store.knn_concepts(two_corpora, UID, emb, k=50):
        assert "krakatoa" not in _flat(hit)
    # and B's partition still matches its own content
    assert any("krakatoa" in (h["text"] or "").lower()
               for h in store.knn_sentences(two_corpora, UID_B, emb, k=5))


def test_fts_scope(two_corpora):
    """episodes_fts carries user_id UNINDEXED; every MATCH must filter on it."""
    rows = two_corpora.execute(
        """SELECT episode_id FROM episodes_fts
           WHERE episodes_fts MATCH 'krakatoa' AND user_id = ?""",
        (UID,)).fetchall()
    assert rows == []
    rows_b = two_corpora.execute(
        """SELECT episode_id FROM episodes_fts
           WHERE episodes_fts MATCH 'krakatoa' AND user_id = ?""",
        (UID_B,)).fetchall()
    assert len(rows_b) == 2


def test_same_claim_text_two_users_distinct_ids(conn, fake_llm):
    """AUTH.md §8: md5(text) PK collision — identical text from two users must
    mint two claims, not share one row."""
    encode(conn, UID, A1, source="test")
    encode(conn, UID_B, A1, source="test")
    consolidate(conn, UID)
    consolidate(conn, UID_B)
    rows = conn.execute(
        "SELECT id, user_id FROM claims ORDER BY user_id").fetchall()
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]
    # strength stays per-user: B's encounter must not bump A's claim
    for r in rows:
        assert conn.execute("SELECT strength FROM claims WHERE id = ?",
                            (r["id"],)).fetchone()["strength"] == 1.0


def test_bridges_and_synthesize_scoped(two_corpora):
    from core.reconstruct import bridges
    assert all("krakatoa" not in _flat(b) for b in bridges(two_corpora, UID))


def test_rebuild_preserves_per_user_separation(two_corpora):
    """rebuild() spans users; afterwards each corpus must still be intact and
    still separate."""
    from core.consolidate import rebuild
    before_a = store.stats(two_corpora, UID)
    before_b = store.stats(two_corpora, UID_B)
    rebuild(two_corpora)
    after_a = store.stats(two_corpora, UID)
    after_b = store.stats(two_corpora, UID_B)
    for k in ("claims", "concepts", "relations"):
        assert after_a[k] == before_a[k]
        assert after_b[k] == before_b[k]
    hits = recall(two_corpora, UID, "Krakatoa secret launch budget", k=20)
    assert "krakatoa" not in _flat(hits)
