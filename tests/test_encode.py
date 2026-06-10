import json

from core import store
from core.encode import encode, split_sentences

NOTE_A = (
    "Spaced repetition is the most reliable way to retain knowledge over years. "
    "The brain consolidates memories during sleep, replaying the day's experiences. "
    "Without periodic review, even important insights decay into vague impressions."
)

NOTE_B = (
    "Memory consolidation happens during sleep when the brain replays recent experiences. "
    "This is completely unrelated filler about cooking pasta with plenty of salted water."
)


def test_split_sentences_filters_short():
    sents = split_sentences("Short. " + "This sentence is definitely long enough to pass the filter easily.")
    assert len(sents) == 1


def test_encode_writes_episode_and_sentences(conn):
    receipt = encode(conn, NOTE_A, title="memory note", source="test")
    ep = store.get_episode(conn, receipt["episode_id"])
    assert ep is not None
    assert ep["raw_text"] == NOTE_A
    assert ep["title"] == "memory note"
    n_sents = conn.execute("SELECT COUNT(*) AS n FROM episode_sentences").fetchone()["n"]
    assert n_sents == receipt["n_sentences"] > 0
    n_vecs = conn.execute("SELECT COUNT(*) AS n FROM vec_sentences").fetchone()["n"]
    assert n_vecs == n_sents
    # receipt persisted on the episode row
    stored = json.loads(ep["receipt_json"])
    assert stored["n_sentences"] == receipt["n_sentences"]


def test_first_note_is_all_novelty(conn):
    receipt = encode(conn, NOTE_A, source="test")
    assert receipt["n_novelties"] > 0
    assert receipt["echoes"] == []
    assert receipt["prior_episode_matches"] == []


def test_second_similar_note_matches_prior_episode(conn):
    encode(conn, NOTE_A, source="test")
    receipt = encode(conn, NOTE_B, source="test")
    matches = receipt["prior_episode_matches"]
    assert matches, "similar sentence should match the prior episode"
    assert matches[0]["similarity"] >= 0.72


def test_encode_emits_encoded_event(conn):
    encode(conn, NOTE_A, source="test")
    rows = store.events_since(conn, 0, types=["ENCODED"])
    assert len(rows) == 1
    payload = json.loads(rows[0]["payload_json"])
    assert payload["n_sentences"] > 0


def test_encode_rejects_empty_text(conn):
    import pytest
    with pytest.raises(ValueError):
        encode(conn, "   ")
