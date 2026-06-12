import pytest

from core.consolidate import consolidate, emit
from core.encode import encode
from core.reconstruct import bridges, reconstruct, synthesize
from tests.conftest import UID
from tests.test_consolidate import S1, S2, fake_llm  # noqa: F401

TS = "2026-06-11T00:00:00+00:00"


@pytest.fixture
def fake_gen_llm(monkeypatch, fake_llm):
    """Extend the consolidation fake with generation/judging prompts."""
    import core.llm as llm_mod
    inner = llm_mod.call  # the consolidation fake installed by fake_llm

    def fake_call(prompt, tier="mechanical", max_tokens=2048, system=None, json_out=True):
        base = {"text": "", "provider": "fake", "model": "fake",
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        if prompt.startswith("Reconstruct the author's original note"):
            return {**base, "text": f"{S1} {S2}"}
        if prompt.startswith("Rate how faithfully"):
            return {**base, "json": {"fidelity": 8, "missing": [], "invented": []}}
        if "found a connection between two" in prompt:
            return {**base, "text": "A synthesized document about the connection."}
        return inner(prompt, tier=tier, max_tokens=max_tokens,
                     system=system, json_out=json_out)

    monkeypatch.setattr("core.llm.call", fake_call)


def test_reconstruct_requires_blueprint(conn, fake_gen_llm):
    receipt = encode(conn, UID, S1, source="test")
    with pytest.raises(ValueError, match="no blueprint"):
        reconstruct(conn, UID, receipt["episode_id"])


def test_reconstruct_scores_fidelity_and_compression(conn, fake_gen_llm):
    receipt = encode(conn, UID, S1, source="test", title="memory")
    consolidate(conn, UID)
    result = reconstruct(conn, UID, receipt["episode_id"])
    assert result["fidelity"] == 8
    assert result["reconstruction"]
    assert 0 < result["compression"]["ratio"]


def test_synthesize_uses_bridge_rationale(conn, fake_gen_llm):
    for cid, text, label in (("clm_a", S1, "Memory"), ("clm_b", S2, "Sleep")):
        emit(conn, UID, "run_t", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": TS})
    emit(conn, UID, "run_t", "CONCEPT_CREATED",
         {"concept_id": "cpt_a", "label": "Memory", "canonical": "m",
          "claim_ids": ["clm_a"], "ts": TS})
    emit(conn, UID, "run_t", "CONCEPT_CREATED",
         {"concept_id": "cpt_b", "label": "Sleep", "canonical": "s",
          "claim_ids": ["clm_b"], "ts": TS})
    emit(conn, UID, "run_t", "BRIDGED",
         {"a": "cpt_a", "b": "cpt_b", "score": 0.6,
          "rationale": "memory depends on sleep", "ts": TS})
    conn.commit()

    assert bridges(conn, UID)[0]["a_label"] == "Memory"
    result = synthesize(conn, UID, "cpt_a", "cpt_b")
    assert result["rationale"] == "memory depends on sleep"
    assert result["document"]
