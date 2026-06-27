from core import store
from core.consolidate import consolidate, emit
from core.encode import encode
from core.recall import (assemble_context, get_claim, get_concept,
                         get_episode, list_episodes, recall)
from tests.conftest import UID
from tests.test_consolidate import S1, S2, S3, _seed, fake_llm  # noqa: F401

TS = "2026-06-11T00:00:00+00:00"


def _emit_claim(conn, cid, text, user_id=UID):
    emit(conn, user_id, "run_test", "CANONICALIZED",
         {"action": "new", "claim_id": cid, "text": text,
          "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": TS})


def test_recall_finds_seeded_claim(conn, fake_llm):
    _seed(conn, UID, S1)
    _seed(conn, UID, S3)
    consolidate(conn, UID)
    hits = recall(conn, UID, "how do I remember things long term", k=5)
    assert hits
    texts = " ".join(h.get("text", "") + h.get("label", "") for h in hits)
    assert "repetition" in texts.lower() or "retain" in texts.lower()


def test_read_api_episode_claims_and_provenance(conn, fake_llm):
    _seed(conn, UID, S1, title="memory note")
    consolidate(conn, UID)

    eps = list_episodes(conn, UID)
    assert len(eps) == 1 and eps[0]["essence"]

    full = get_episode(conn, UID, eps[0]["id"])
    assert full["blueprint"] is not None
    assert full["claims"], "episode should support at least one claim"

    concept = store.all_concepts(conn, UID)[0]
    cfull = get_concept(conn, UID, concept["id"])
    assert cfull["members"]
    support = cfull["members"][0]["support"]
    assert support and support[0]["episode_id"] == eps[0]["id"]  # provenance ✓

    claim = get_claim(conn, UID, cfull["members"][0]["claim_id"])
    assert claim["support"][0]["episode_id"] == eps[0]["id"]
    assert claim["concepts"] == [concept["id"]]


def test_assemble_context_markdown(conn, fake_llm):
    _seed(conn, UID, S1, title="memory note")
    _seed(conn, UID, S2, title="sleep note")
    consolidate(conn, UID)
    md = assemble_context(conn, UID, "memory and sleep")
    assert md.startswith("## Slate context")
    assert "###" in md          # concept-grouped
    assert "2026" in md or "note" in md  # provenance present


def test_assemble_context_empty_store(conn):
    assert "nothing stored" in assemble_context(conn, UID, "anything")
