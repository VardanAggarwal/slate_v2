import pytest

from core import store
from core.consolidate import consolidate, emit
from core.encode import encode
from core.recall import (assemble_context, get_claim, get_concept,
                         get_episode, list_episodes, recall)
from tests.test_consolidate import S1, S2, S3, fake_llm  # noqa: F401

TS = "2026-06-11T00:00:00+00:00"


def _emit_claim(conn, cid, text):
    emit(conn, "run_test", "CANONICALIZED",
         {"action": "new", "claim_id": cid, "text": text,
          "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": TS})


def test_recall_finds_seeded_claim(conn, fake_llm):
    encode(conn, S1, source="test")
    encode(conn, S3, source="test")
    consolidate(conn)
    hits = recall(conn, "how do I remember things long term", k=5)
    assert hits
    texts = " ".join(h.get("text", "") + h.get("label", "") for h in hits)
    assert "repetition" in texts.lower() or "retain" in texts.lower()


def test_recall_two_hop_via_bridge(conn):
    with conn:
        _emit_claim(conn, "clm_mem", S1)   # memory claim
        _emit_claim(conn, "clm_art", "Renaissance painters mixed pigments with egg yolk for tempera.")
        emit(conn, "run_test", "CONCEPT_CREATED",
             {"concept_id": "cpt_mem", "label": "Memory", "canonical": S1,
              "claim_ids": ["clm_mem"], "ts": TS})
        emit(conn, "run_test", "CONCEPT_CREATED",
             {"concept_id": "cpt_art", "label": "Tempera Painting", "canonical": "art",
              "claim_ids": ["clm_art"], "ts": TS})
        emit(conn, "run_test", "BRIDGED",
             {"a": "cpt_mem", "b": "cpt_art", "score": 0.6, "rationale": "test", "ts": TS})

    hits = recall(conn, "techniques for remembering knowledge", k=10)
    ids = {h["id"]: h for h in hits}
    assert "cpt_art" in ids, "bridge should pull the unrelated concept into results"
    assert any("bridge" in s.lower() or "🌉" in s for s in ids["cpt_art"]["signals"])


def test_read_api_episode_claims_and_provenance(conn, fake_llm):
    encode(conn, S1, source="test", title="memory note")
    consolidate(conn)

    eps = list_episodes(conn)
    assert len(eps) == 1 and eps[0]["essence"]

    full = get_episode(conn, eps[0]["id"])
    assert full["blueprint"] is not None
    assert full["claims"], "episode should support at least one claim"

    concept = store.all_concepts(conn)[0]
    cfull = get_concept(conn, concept["id"])
    assert cfull["members"]
    support = cfull["members"][0]["support"]
    assert support and support[0]["episode_id"] == eps[0]["id"]  # provenance ✓

    claim = get_claim(conn, cfull["members"][0]["claim_id"])
    assert claim["support"][0]["episode_id"] == eps[0]["id"]
    assert claim["concepts"] == [concept["id"]]


def test_assemble_context_markdown(conn, fake_llm):
    encode(conn, S1, source="test", title="memory note")
    encode(conn, S2, source="test", title="sleep note")
    consolidate(conn)
    md = assemble_context(conn, "memory and sleep")
    assert md.startswith("## Slate context")
    assert "###" in md          # concept-grouped
    assert "2026" in md or "note" in md  # provenance present


def test_assemble_context_empty_store(conn):
    assert "nothing stored" in assemble_context(conn, "anything")
