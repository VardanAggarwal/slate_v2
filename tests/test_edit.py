"""edit_note: edit = supersede (core/edit.py + the EPISODE_SUPERSEDED applier).

Episodes are immutable, so an edit saves a NEW episode and masks the old one.
These tests cover the blast radius: browse/list, receipt echoes, refine +
consolidation queues, fragments, claim support/orphans, FTS, and rebuild().
"""
import pytest

from core import store, write
from core.consolidate import emit, rebuild
from core.edit import edit_note
from core.encode import encode
from core.recall import get_episode, list_episodes
from tests.conftest import UID

NOTE = (
    "Spaced repetition is the most reliable way to retain knowledge over years. "
    "Cramming the night before an exam produces almost no long-term retention."
)
REVISED = (
    "Spaced repetition is the most reliable way to retain knowledge over years. "
    "Interleaving different topics within a session further improves recall."
)
OTHER = "Memory consolidation happens during sleep when the brain replays recent experiences."

TS = "2026-06-11T00:00:00+00:00"


def _save(conn, text, title=None, refine=False):
    r = encode(conn, UID, text, source="test", title=title)
    if refine:
        write.refine_episode(conn, UID, r["episode_id"])
    return r["episode_id"]


# ── core semantics ─────────────────────────────────────────────────────────────
def test_edit_creates_revision_and_masks_old(conn):
    old_id = _save(conn, NOTE, title="learning note")
    old_ts = store.get_episode(conn, UID, old_id)["ts"]

    receipt = edit_note(conn, UID, old_id, REVISED)
    new_id = receipt["episode_id"]
    assert receipt["superseded_episode_id"] == old_id
    assert receipt["title"] == "learning note"      # kept when not overridden

    new_ep = store.get_episode(conn, UID, new_id)
    assert new_ep["raw_text"] == REVISED
    assert new_ep["source"] == "edit"
    assert new_ep["ts"] == old_ts                    # revision keeps the original date
    # old row untouched (immutable), only masked
    assert store.get_episode(conn, UID, old_id)["raw_text"] == NOTE
    assert store.episode_superseded_by(conn, UID, old_id) == new_id
    assert store.episode_supersedes(conn, UID, new_id) == old_id

    ids = [e["id"] for e in list_episodes(conn, UID)]
    assert new_id in ids and old_id not in ids


def test_edit_lineage_annotations(conn):
    old_id = _save(conn, NOTE)
    new_id = edit_note(conn, UID, old_id, REVISED)["episode_id"]
    assert get_episode(conn, UID, old_id)["superseded_by"] == new_id
    assert get_episode(conn, UID, new_id)["edited_from"] == old_id
    assert "superseded_by" not in get_episode(conn, UID, new_id)


def test_edit_rejects_unknown_and_already_superseded(conn):
    with pytest.raises(ValueError, match="not found"):
        edit_note(conn, UID, "ep_missing", REVISED)
    old_id = _save(conn, NOTE)
    new_id = edit_note(conn, UID, old_id, REVISED)["episode_id"]
    with pytest.raises(ValueError, match=new_id):
        edit_note(conn, UID, old_id, "changed my mind again entirely.")
    # chain: editing the current head works
    edit_note(conn, UID, new_id, "changed my mind again entirely, twice over.")


def test_edit_receipt_does_not_echo_the_note_being_edited(conn):
    old_id = _save(conn, NOTE)
    other_id = _save(conn, REVISED)  # a distinct note sharing sentence 1
    receipt = edit_note(conn, UID, old_id, NOTE + " Plus one extra new thought.")
    eps = {m["episode_id"] for m in receipt["prior_episode_matches"]}
    assert old_id not in eps          # an edit must not resonate with itself
    assert other_id in eps            # but real prior overlap still surfaces


# ── queues, fragments, sentences ──────────────────────────────────────────────
def test_superseded_episode_leaves_work_queues(conn):
    old_id = _save(conn, NOTE)       # unrefined + unconsolidated
    edit_note(conn, UID, old_id, REVISED)
    assert old_id not in [e["id"] for e in store.unfragmented_episodes(conn, UID)]
    assert old_id not in [e["id"] for e in store.unconsolidated_episodes(conn, UID)]
    # the fragmentation marker was claimed, so a late async refine skips it
    assert not store.mark_fragmented(conn, UID, old_id, 0)


def test_edit_deletes_old_fragments(conn):
    old_id = _save(conn, NOTE, refine=True)
    assert store.episode_fragments(conn, UID, old_id)
    edit_note(conn, UID, old_id, REVISED)
    assert store.episode_fragments(conn, UID, old_id) == []


def test_knn_sentences_skips_superseded(conn):
    old_id = _save(conn, NOTE)
    edit_note(conn, UID, old_id, OTHER)  # full rewrite
    emb = store.episode_sentences_with_vectors(
        conn, UID, old_id)[0]["embedding"]
    hits = store.knn_sentences(conn, UID, emb, k=5)
    assert all(h["episode_id"] != old_id for h in hits)


# ── claims ─────────────────────────────────────────────────────────────────────
def test_orphaned_claim_withdrawn_shared_claim_kept(conn):
    old_id = _save(conn, NOTE)
    other_id = _save(conn, OTHER)
    # claim A supported ONLY by the note being edited; claim B shared with another
    emit(conn, UID, "run_test", "CANONICALIZED",
         {"action": "new", "claim_id": "clm_only", "text": "cramming does not retain",
          "episode_id": old_id, "verbatim": NOTE, "cluster": "main", "ts": TS})
    emit(conn, UID, "run_test", "CANONICALIZED",
         {"action": "new", "claim_id": "clm_shared", "text": "sleep consolidates memory",
          "episode_id": old_id, "verbatim": NOTE, "cluster": "main", "ts": TS})
    emit(conn, UID, "run_test", "CANONICALIZED",
         {"action": "support", "claim_id": "clm_shared",
          "episode_id": other_id, "verbatim": OTHER, "cluster": "main", "ts": TS})
    emit(conn, UID, "run_test", "CONCEPT_CREATED",
         {"concept_id": "cpt_t", "label": "t", "canonical": "t",
          "claim_ids": ["clm_only", "clm_shared"], "ts": TS})
    conn.commit()

    edit_note(conn, UID, old_id, REVISED)

    assert store.get_claim(conn, UID, "clm_only") is None
    assert store.concept_member_ids(conn, UID, "cpt_t") == ["clm_shared"]
    shared = store.get_claim(conn, UID, "clm_shared")
    assert shared is not None and shared["status"] == "current"
    assert store.claim_source_episodes(conn, UID, "clm_shared") == [other_id]
    # concept vector recomputed over the survivor, not left dangling
    assert store.get_concept(conn, UID, "cpt_t") is not None


# ── event sourcing ─────────────────────────────────────────────────────────────
def test_rebuild_replays_supersession(conn):
    old_id = _save(conn, NOTE, refine=True)
    emit(conn, UID, "run_test", "CANONICALIZED",
         {"action": "new", "claim_id": "clm_only", "text": "cramming does not retain",
          "episode_id": old_id, "verbatim": NOTE, "cluster": "main", "ts": TS})
    conn.commit()
    new_id = edit_note(conn, UID, old_id, REVISED)["episode_id"]

    rebuild(conn)

    assert store.episode_superseded_by(conn, UID, old_id) == new_id
    assert store.get_claim(conn, UID, "clm_only") is None
    assert store.episode_fragments(conn, UID, old_id) == []
    assert old_id not in [e["id"] for e in list_episodes(conn, UID)]


def test_fts_row_removed_for_old_version(conn):
    old_id = _save(conn, NOTE)
    edit_note(conn, UID, old_id, OTHER)
    rows = conn.execute(
        "SELECT episode_id FROM episodes_fts WHERE episode_id = ?", (old_id,)).fetchall()
    assert rows == []


def test_stats_counts_superseded(conn):
    old_id = _save(conn, NOTE)
    edit_note(conn, UID, old_id, REVISED)
    s = store.stats(conn, UID)
    assert s["episodes_superseded"] == 1
    assert s["episodes"] == 2   # both raw rows kept
