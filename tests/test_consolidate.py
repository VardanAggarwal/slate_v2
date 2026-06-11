import json

import pytest

from core import store
from core.consolidate import (_decay_strengthen, apply_event, consolidate,
                              emit, rebuild)
from core.encode import encode, split_sentences
from core.llm import LLMError

S1 = "Spaced repetition is the most reliable way to retain knowledge over many years."
S2 = "Memory consolidation happens during sleep when the brain replays recent experiences."
S3 = "Constraints often increase creativity rather than limiting what can be made."


@pytest.fixture
def fake_llm(monkeypatch):
    """Deterministic stand-in for core.llm.call keyed off each prompt's header."""
    calls = {"blueprint": 0, "canon": 0, "concept": 0, "bridge": 0}

    def fake_call(prompt, tier="mechanical", max_tokens=2048, system=None, json_out=True):
        base = {"text": "", "provider": "fake", "model": "fake",
                "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        if prompt.startswith("Extract semantic structure"):
            calls["blueprint"] += 1
            text = prompt.split("TEXT:\n", 1)[1]
            sents = split_sentences(text) or [text.strip()]
            return {**base, "json": {
                "title": " ".join(sents[0].split()[:4]), "essence": sents[0],
                "clusters": [{"label": "main", "kernel": sents[0], "claims": sents,
                              "representative_sentences": sents}],
                "assumptions": [], "spine": []}}
        if prompt.startswith("You deduplicate"):
            calls["canon"] += 1
            n = prompt.count("\n   EXISTING:")
            return {**base, "json": {"verdicts": [{"i": i, "same": False} for i in range(n)]}}
        if prompt.startswith("You maintain the concept layer"):
            calls["concept"] += 1
            new_claims = json.loads(
                prompt.split("NEW CLAIMS:\n", 1)[1].split("\n\nEXISTING CONCEPTS", 1)[0])
            if not new_claims:
                return {**base, "json": {"decisions": []}}
            return {**base, "json": {"decisions": [{
                "action": "CREATE", "label": "test concept",
                "canonical": "claims grouped by the fake llm",
                "claim_ids": [c["id"] for c in new_claims]}]}}
        if "drifted near each other" in prompt:
            calls["bridge"] += 1
            return {**base, "json": {"bridge": False, "rationale": ""}}
        raise AssertionError(f"unexpected prompt: {prompt[:80]}")

    monkeypatch.setattr("core.llm.call", fake_call)
    return calls


def _dump_semantic(conn):
    out = {}
    for table, order in (("claims", "id"), ("claim_support", "claim_id, episode_id"),
                         ("concepts", "id"), ("concept_members", "concept_id, claim_id"),
                         ("relations", "from_id, to_id, relation")):
        out[table] = [tuple(r) for r in
                      conn.execute(f"SELECT * FROM {table} ORDER BY {order}")]
    out["vec_claims"] = conn.execute("SELECT COUNT(*) FROM vec_claims").fetchone()[0]
    out["vec_concepts"] = conn.execute("SELECT COUNT(*) FROM vec_concepts").fetchone()[0]
    return out


def test_consolidate_dedupes_claims(conn, fake_llm):
    encode(conn, S1, source="test")
    encode(conn, f"{S1} {S2}", source="test")  # repeats S1, adds S2
    report = consolidate(conn)
    assert report["status"] == "ok"
    n_claims = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    n_support = conn.execute("SELECT COUNT(*) FROM claim_support").fetchone()[0]
    assert n_claims == 2          # 3 raw claim instances → 2 canonical
    assert n_support == 3         # S1 supported by both episodes
    repeated = conn.execute(
        "SELECT strength FROM claims WHERE id IN "
        "(SELECT claim_id FROM claim_support GROUP BY claim_id HAVING COUNT(*) = 2)"
    ).fetchone()
    assert repeated["strength"] > 1.0  # re-encounter bumped strength


def test_consolidate_creates_concept_and_marks_episodes(conn, fake_llm):
    encode(conn, S1, source="test")
    consolidate(conn)
    concepts = store.all_concepts(conn)
    assert len(concepts) == 1
    assert store.concept_member_ids(conn, concepts[0]["id"])
    assert store.unconsolidated_episodes(conn) == []
    assert consolidate(conn)["status"] == "noop"


def test_rebuild_reproduces_semantic_store(conn, fake_llm):
    encode(conn, S1, source="test")
    encode(conn, f"{S2} {S3}", source="test")
    consolidate(conn)
    before = _dump_semantic(conn)
    assert before["claims"]       # non-trivial store
    rebuild(conn)
    assert _dump_semantic(conn) == before


def test_split_applier(conn, fake_llm):
    encode(conn, f"{S1} {S2}", source="test")
    consolidate(conn)
    parent = store.all_concepts(conn)[0]
    members = store.concept_member_ids(conn, parent["id"])
    assert len(members) == 2
    emit(conn, "run_test", "SPLIT", {
        "concept_id": parent["id"],
        "snapshot": {"concept": dict(parent), "member_claim_ids": members},
        "into": [
            {"concept_id": "cpt_childA", "label": "A", "canonical": "a", "claim_ids": [members[0]]},
            {"concept_id": "cpt_childB", "label": "B", "canonical": "b", "claim_ids": [members[1]]},
        ], "ts": "2026-06-11T00:00:00+00:00"})
    conn.commit()
    assert store.get_concept(conn, parent["id"]) is None
    assert store.concept_member_ids(conn, "cpt_childA") == [members[0]]
    assert store.concept_member_ids(conn, "cpt_childB") == [members[1]]


def test_merge_applier_keeps_history(conn, fake_llm):
    ts = "2026-06-11T00:00:00+00:00"
    for cid, text in (("c1", S1), ("c2", S2)):
        emit(conn, "run_test", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": ts})
    emit(conn, "run_test", "CONCEPT_CREATED",
         {"concept_id": "cpt_w", "label": "winner", "canonical": "w",
          "claim_ids": ["c1"], "ts": ts})
    emit(conn, "run_test", "CONCEPT_CREATED",
         {"concept_id": "cpt_l", "label": "loser", "canonical": "l",
          "claim_ids": ["c2"], "ts": ts})
    emit(conn, "run_test", "MERGED", {
        "winner_id": "cpt_w", "loser_id": "cpt_l", "label": None, "canonical": None,
        "winner_snapshot": {"concept": dict(store.get_concept(conn, "cpt_w")),
                            "member_claim_ids": ["c1"]},
        "loser_snapshot": {"concept": dict(store.get_concept(conn, "cpt_l")),
                           "member_claim_ids": ["c2"]}, "ts": ts})
    conn.commit()
    assert store.get_concept(conn, "cpt_l") is None
    assert set(store.concept_member_ids(conn, "cpt_w")) == {"c1", "c2"}
    merged_events = store.events_since(conn, 0, types=["MERGED"])
    payload = json.loads(merged_events[0]["payload_json"])
    assert payload["loser_snapshot"]["concept"]["label"] == "loser"  # history preserved


def test_decay_transitions_state(conn, fake_llm):
    encode(conn, S1, source="test")
    consolidate(conn)
    concept = store.all_concepts(conn)[0]
    with conn:
        store.update_concept(conn, concept["id"], last_activity="2025-01-01T00:00:00+00:00")
    with conn:
        _decay_strengthen(conn, "run_test", [], "2026-06-11T00:00:00+00:00")
    assert store.get_concept(conn, concept["id"])["state"] == "dormant"
    assert store.events_since(conn, 0, types=["DECAYED"])


def test_failed_run_leaves_episodes_unconsolidated(conn, monkeypatch):
    def always_fail(*a, **kw):
        raise LLMError("forced failure")
    monkeypatch.setattr("core.llm.call", always_fail)

    encode(conn, S1, source="test")
    with pytest.raises(LLMError):
        consolidate(conn)  # blueprint falls back local; concept pass raises
    assert len(store.unconsolidated_episodes(conn)) == 1
    run = conn.execute("SELECT status FROM consolidation_runs").fetchone()
    assert run["status"] == "failed"


def test_retry_reuses_blueprint_and_canon_events(conn, fake_llm, monkeypatch):
    encode(conn, S1, source="test")
    original_concept_pass = "core.consolidate._concept_pass_chunk"
    import core.consolidate as consolidate_mod

    real_chunk = consolidate_mod._concept_pass_chunk
    state = {"fail": True}

    def flaky_chunk(*args, **kwargs):
        if state["fail"]:
            state["fail"] = False
            raise LLMError("transient concept-pass failure")
        return real_chunk(*args, **kwargs)

    monkeypatch.setattr(consolidate_mod, "_concept_pass_chunk", flaky_chunk)
    with pytest.raises(LLMError):
        consolidate(conn)

    blueprints_before = fake_llm["blueprint"]
    canon_events = len(store.events_since(conn, 0, types=["CANONICALIZED"]))
    claims_before = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    report = consolidate(conn)  # retry succeeds
    assert report["status"] == "ok"
    assert fake_llm["blueprint"] == blueprints_before          # no re-extraction
    assert len(store.events_since(conn, 0, types=["CANONICALIZED"])) == canon_events
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == claims_before
    assert store.unconsolidated_episodes(conn) == []


def test_llm_calls_never_hold_a_write_transaction(conn, fake_llm, monkeypatch):
    """A save_note arriving mid-consolidation must never wait on a network call:
    every llm.call must happen with no transaction open on the connection."""
    import core.llm as llm_mod
    inner = llm_mod.call

    def guarded(prompt, **kw):
        assert not conn.in_transaction, \
            f"llm.call while holding a write txn: {prompt[:60]!r}"
        return inner(prompt, **kw)

    monkeypatch.setattr("core.llm.call", guarded)
    encode(conn, S1, source="test")
    encode(conn, f"{S2} {S3}", source="test")
    report = consolidate(conn)
    assert report["status"] == "ok"
    assert fake_llm["blueprint"] >= 2 and fake_llm["concept"] >= 1  # guard exercised


def test_stubborn_episode_is_skipped_not_fatal(conn, fake_llm, monkeypatch):
    """A note whose blueprint fails on all providers must not kill the run."""
    import core.llm as llm_mod
    inner = llm_mod.call

    def poisoned(prompt, **kw):
        if prompt.startswith("Extract semantic structure") and S3 in prompt:
            raise LLMError("malformed JSON from every provider")
        return inner(prompt, **kw)

    monkeypatch.setattr("core.llm.call", poisoned)
    # Simulate the slim server image: no sklearn, so no local fallback
    def no_sklearn(text):
        raise ImportError("No module named 'sklearn'")
    monkeypatch.setattr("core.consolidate._blueprint_local", no_sklearn)

    encode(conn, S1, source="test")
    encode(conn, S3, source="test")  # the poisoned note

    report = consolidate(conn)
    assert report["status"] == "ok"
    assert report["episodes"] == 1
    assert len(report["skipped"]) == 1
    remaining = store.unconsolidated_episodes(conn)
    assert len(remaining) == 1       # picked up by the next run
    assert S3 in remaining[0]["raw_text"]


def test_llm_call_retries_malformed_json_once(monkeypatch):
    import core.llm as llm_mod
    attempts = {"n": 0}

    def flaky_claude(prompt, model, max_tokens, system):
        attempts["n"] += 1
        text = "{bad json" if attempts["n"] == 1 else '{"ok": true}'
        return {"text": text, "provider": "claude", "model": model,
                "input_tokens": 1, "output_tokens": 1, "cost": 0.0}

    monkeypatch.setattr(llm_mod, "_call_claude", flaky_claude)
    monkeypatch.setattr(llm_mod.config, "ANTHROPIC_KEY", "test")
    monkeypatch.setattr(llm_mod.config, "LLM_FALLBACK_ORDER", ["claude"])
    result = llm_mod.call("anything")
    assert result["json"] == {"ok": True}
    assert attempts["n"] == 2


def test_episodes_are_immutable(conn):
    import sqlite3
    encode(conn, S1, source="test")
    ep_id = conn.execute("SELECT id FROM episodes").fetchone()["id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE episodes SET title = 'x' WHERE id = ?", (ep_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))
