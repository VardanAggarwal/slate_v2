import json

import pytest

from core import store
from core.consolidate import (_apply_concept_decisions, _decay_strengthen,
                              _prune_safely, apply_event, consolidate, emit,
                              rebuild, rollback_run)
from core.encode import encode, split_sentences
from core.llm import LLMError
from tests.conftest import UID

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
    encode(conn, UID, S1, source="test")
    encode(conn, UID, f"{S1} {S2}", source="test")  # repeats S1, adds S2
    report = consolidate(conn, UID)
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
    encode(conn, UID, S1, source="test")
    consolidate(conn, UID)
    concepts = store.all_concepts(conn, UID)
    assert len(concepts) == 1
    assert store.concept_member_ids(conn, UID, concepts[0]["id"])
    assert store.unconsolidated_episodes(conn, UID) == []
    assert consolidate(conn, UID)["status"] == "noop"


def test_rebuild_reproduces_semantic_store(conn, fake_llm):
    encode(conn, UID, S1, source="test")
    encode(conn, UID, f"{S2} {S3}", source="test")
    consolidate(conn, UID)
    before = _dump_semantic(conn)
    assert before["claims"]       # non-trivial store
    rebuild(conn)
    assert _dump_semantic(conn) == before


def test_split_applier(conn, fake_llm):
    encode(conn, UID, f"{S1} {S2}", source="test")
    consolidate(conn, UID)
    parent = store.all_concepts(conn, UID)[0]
    members = store.concept_member_ids(conn, UID, parent["id"])
    assert len(members) == 2
    emit(conn, UID, "run_test", "SPLIT", {
        "concept_id": parent["id"],
        "snapshot": {"concept": dict(parent), "member_claim_ids": members},
        "into": [
            {"concept_id": "cpt_childA", "label": "A", "canonical": "a", "claim_ids": [members[0]]},
            {"concept_id": "cpt_childB", "label": "B", "canonical": "b", "claim_ids": [members[1]]},
        ], "ts": "2026-06-11T00:00:00+00:00"})
    conn.commit()
    assert store.get_concept(conn, UID, parent["id"]) is None
    assert store.concept_member_ids(conn, UID, "cpt_childA") == [members[0]]
    assert store.concept_member_ids(conn, UID, "cpt_childB") == [members[1]]


def test_merge_applier_keeps_history(conn, fake_llm):
    ts = "2026-06-11T00:00:00+00:00"
    for cid, text in (("c1", S1), ("c2", S2)):
        emit(conn, UID, "run_test", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": ts})
    emit(conn, UID, "run_test", "CONCEPT_CREATED",
         {"concept_id": "cpt_w", "label": "winner", "canonical": "w",
          "claim_ids": ["c1"], "ts": ts})
    emit(conn, UID, "run_test", "CONCEPT_CREATED",
         {"concept_id": "cpt_l", "label": "loser", "canonical": "l",
          "claim_ids": ["c2"], "ts": ts})
    emit(conn, UID, "run_test", "MERGED", {
        "winner_id": "cpt_w", "loser_id": "cpt_l", "label": None, "canonical": None,
        "winner_snapshot": {"concept": dict(store.get_concept(conn, UID, "cpt_w")),
                            "member_claim_ids": ["c1"]},
        "loser_snapshot": {"concept": dict(store.get_concept(conn, UID, "cpt_l")),
                           "member_claim_ids": ["c2"]}, "ts": ts})
    conn.commit()
    assert store.get_concept(conn, UID, "cpt_l") is None
    assert set(store.concept_member_ids(conn, UID, "cpt_w")) == {"c1", "c2"}
    merged_events = store.events_since(conn, UID, 0, types=["MERGED"])
    payload = json.loads(merged_events[0]["payload_json"])
    assert payload["loser_snapshot"]["concept"]["label"] == "loser"  # history preserved


def test_decay_transitions_state(conn, fake_llm):
    encode(conn, UID, S1, source="test")
    consolidate(conn, UID)
    concept = store.all_concepts(conn, UID)[0]
    with conn:
        store.update_concept(conn, UID, concept["id"],
                             last_activity="2025-01-01T00:00:00+00:00")
    with conn:
        _decay_strengthen(conn, UID, "run_test", [], "2026-06-11T00:00:00+00:00")
    assert store.get_concept(conn, UID, concept["id"])["state"] == "dormant"
    assert store.events_since(conn, UID, 0, types=["DECAYED"])


def test_failed_run_leaves_episodes_unconsolidated(conn, monkeypatch):
    def always_fail(*a, **kw):
        raise LLMError("forced failure")
    monkeypatch.setattr("core.llm.call", always_fail)

    encode(conn, UID, S1, source="test")
    with pytest.raises(LLMError):
        consolidate(conn, UID)  # blueprint falls back local; concept pass raises
    assert len(store.unconsolidated_episodes(conn, UID)) == 1
    run = conn.execute("SELECT status FROM consolidation_runs").fetchone()
    assert run["status"] == "failed"


def test_retry_reuses_blueprint_and_canon_events(conn, fake_llm, monkeypatch):
    encode(conn, UID, S1, source="test")
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
        consolidate(conn, UID)

    blueprints_before = fake_llm["blueprint"]
    canon_events = len(store.events_since(conn, UID, 0, types=["CANONICALIZED"]))
    claims_before = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]

    report = consolidate(conn, UID)  # retry succeeds
    assert report["status"] == "ok"
    assert fake_llm["blueprint"] == blueprints_before          # no re-extraction
    assert len(store.events_since(conn, UID, 0, types=["CANONICALIZED"])) == canon_events
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == claims_before
    assert store.unconsolidated_episodes(conn, UID) == []


# ── C14: run-scoped rollback + re-derive from raw ─────────────────────────────
def test_rollback_fully_reverses_a_run(conn, fake_llm):
    """A bad run is undoable: the semantic store returns to its exact pre-run
    state (PRD §Consolidation: bad runs must be undoable)."""
    encode(conn, UID, S1, source="test")
    consolidate(conn, UID)               # run A — the keeper
    before = _dump_semantic(conn)

    encode(conn, UID, S3, source="test")
    run_b = consolidate(conn, UID)["run_id"]   # run B — to be undone
    assert _dump_semantic(conn) != before      # B did change the store

    res = rollback_run(conn, run_b)
    assert res["status"] == "rolled_back"
    assert res["episodes_freed"] == 1
    assert _dump_semantic(conn) == before       # fully reversed to pre-B state


def test_rollback_keeps_events_on_disk_but_unmaterialized(conn, fake_llm):
    """Rolled-back events survive for audit; they are simply not re-applied."""
    encode(conn, UID, S1, source="test")
    run = consolidate(conn, UID)["run_id"]
    n_all = len(store.events_since(conn, UID, 0))                 # full log
    n_active = len(store.events_since(conn, UID, 0, include_rolled_back=False))
    assert n_all == n_active and n_all > 0
    assert any(e["run_id"] == run for e in store.events_since(conn, UID, 0))  # run emitted some

    rollback_run(conn, run)
    assert len(store.events_since(conn, UID, 0)) == n_all        # nothing deleted
    # only write-side events (run_id NULL, e.g. ENCODED) stay active; no run events
    active = store.events_since(conn, UID, 0, include_rolled_back=False)
    assert active and all(e["run_id"] is None for e in active)
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 0  # not materialized


def test_rollback_then_reconsolidate_rederives_from_raw(conn, fake_llm):
    """Re-derivation bypasses a poisoned log: after rollback the freed episode
    is re-blueprinted from its raw text — the rolled-back BLUEPRINTED/CANONICALIZED
    events do NOT short-circuit the `_existing_*` guards."""
    encode(conn, UID, S1, source="test")
    run_b = consolidate(conn, UID)["run_id"]
    assert store.unconsolidated_episodes(conn, UID) == []   # consolidated

    rollback_run(conn, run_b)
    assert len(store.unconsolidated_episodes(conn, UID)) == 1   # freed

    bp_before = fake_llm["blueprint"]
    run_c = consolidate(conn, UID)
    assert run_c["status"] == "ok" and run_c["run_id"] != run_b
    assert fake_llm["blueprint"] > bp_before                # re-derived, not reused
    assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1  # S1 back


def test_rollback_unknown_run_is_noop(conn, fake_llm):
    encode(conn, UID, S1, source="test")
    consolidate(conn, UID)
    before = _dump_semantic(conn)
    res = rollback_run(conn, "run_does_not_exist")
    assert res["status"] == "unknown_run"
    assert _dump_semantic(conn) == before


# ── C6: merge nuance guard. Guard GEOMETRY is in test_wrappers; these check the
#       partition → MERGED-event → applier WIRING with controlled verdicts. ─────
def _seed_two_concepts(conn, ts):
    """Winner cpt_w[w1]; loser cpt_l[ldup, lnuance]."""
    for cid, text in (("w1", S1), ("ldup", S2), ("lnuance", S3)):
        emit(conn, UID, "seed", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": ts})
    emit(conn, UID, "seed", "CONCEPT_CREATED",
         {"concept_id": "cpt_w", "label": "w", "canonical": "w",
          "claim_ids": ["w1"], "ts": ts})
    emit(conn, UID, "seed", "CONCEPT_CREATED",
         {"concept_id": "cpt_l", "label": "l", "canonical": "l",
          "claim_ids": ["ldup", "lnuance"], "ts": ts})
    conn.commit()


def test_merge_guard_folds_dup_keeps_nuance(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_two_concepts(conn, ts)
    monkeypatch.setattr("core.guard.merge", lambda losers, survivors, **kw: [
        {"id": x["id"], "safe_to_drop": x["id"] == "ldup"} for x in losers])
    with conn:
        _apply_concept_decisions(
            conn, UID, "run_m",
            [{"action": "MERGE", "winner_id": "cpt_w", "loser_id": "cpt_l"}],
            {"w1", "ldup", "lnuance"}, {"cpt_w", "cpt_l"}, ts)
    assert set(store.concept_member_ids(conn, UID, "cpt_w")) == {"w1", "ldup"}
    assert store.concept_member_ids(conn, UID, "cpt_l") == ["lnuance"]  # nuance kept
    assert store.get_concept(conn, UID, "cpt_l") is not None            # loser survives


def test_merge_full_fold_deletes_loser(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_two_concepts(conn, ts)
    monkeypatch.setattr("core.guard.merge", lambda losers, survivors, **kw: [
        {"id": x["id"], "safe_to_drop": True} for x in losers])
    with conn:
        _apply_concept_decisions(
            conn, UID, "run_m",
            [{"action": "MERGE", "winner_id": "cpt_w", "loser_id": "cpt_l"}],
            {"w1", "ldup", "lnuance"}, {"cpt_w", "cpt_l"}, ts)
    assert set(store.concept_member_ids(conn, UID, "cpt_w")) == {"w1", "ldup", "lnuance"}
    assert store.get_concept(conn, UID, "cpt_l") is None                # all folded


def test_merge_partition_survives_rebuild(conn, fake_llm, monkeypatch):
    """The fold/kept split is frozen in the event payload, so log-replay
    reproduces the partial merge without re-running the (non-replayable) guard."""
    ts = "2026-01-01T00:00:00+00:00"
    _seed_two_concepts(conn, ts)
    monkeypatch.setattr("core.guard.merge", lambda losers, survivors, **kw: [
        {"id": x["id"], "safe_to_drop": x["id"] == "ldup"} for x in losers])
    with conn:
        _apply_concept_decisions(
            conn, UID, "run_m",
            [{"action": "MERGE", "winner_id": "cpt_w", "loser_id": "cpt_l"}],
            {"w1", "ldup", "lnuance"}, {"cpt_w", "cpt_l"}, ts)
    # break the guard so a re-run would differ — only the payload should drive replay
    monkeypatch.setattr("core.guard.merge", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("guard must not run at apply time")))
    rebuild(conn)
    assert set(store.concept_member_ids(conn, UID, "cpt_w")) == {"w1", "ldup"}
    assert store.concept_member_ids(conn, UID, "cpt_l") == ["lnuance"]


# ── C7: safe-forget prune. Guard geometry in test_wrappers; wiring here. ───────
def _seed_dormant_concept(conn, ts, state="dormant"):
    for cid, text in (("m1", S1), ("m2", S2), ("m3", S3)):
        emit(conn, UID, "seed", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": ts})
    emit(conn, UID, "seed", "CONCEPT_CREATED",
         {"concept_id": "cpt_d", "label": "d", "canonical": "d",
          "claim_ids": ["m1", "m2", "m3"], "ts": ts})
    with conn:
        store.update_concept(conn, UID, "cpt_d", state=state)
    conn.commit()


def test_prune_drops_reconstructable_protects_irreplaceable(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_dormant_concept(conn, ts)
    monkeypatch.setattr("core.guard.forget", lambda members, **kw: [
        {"id": x["id"], "safe_to_drop": x["id"] == "m2"} for x in members])
    with conn:
        _prune_safely(conn, UID, "run_p", ts)
    assert set(store.concept_member_ids(conn, UID, "cpt_d")) == {"m1", "m3"}
    assert store.get_claim(conn, UID, "m2") is None        # orphan → re-derivable, dropped
    assert store.get_claim(conn, UID, "m1") is not None     # irreplaceable, protected


def test_prune_skips_non_dormant(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_dormant_concept(conn, ts, state="active")
    monkeypatch.setattr("core.guard.forget", lambda members, **kw: (_ for _ in ()).throw(
        AssertionError("active concept must not be pruned")))
    with conn:
        _prune_safely(conn, UID, "run_p", ts)
    assert set(store.concept_member_ids(conn, UID, "cpt_d")) == {"m1", "m2", "m3"}


def test_prune_never_empties_concept(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_dormant_concept(conn, ts)
    monkeypatch.setattr("core.guard.forget", lambda members, **kw: [
        {"id": x["id"], "safe_to_drop": True} for x in members])
    with conn:
        _prune_safely(conn, UID, "run_p", ts)
    assert len(store.concept_member_ids(conn, UID, "cpt_d")) == 1  # one representative kept


# ── C8: conflict detection + versioning. Resolver DIRECTION is mocked (it is the
#       LLM's job); the versioning + margin-before-flip logic is what's tested. ─
from core.consolidate import _reconcile  # noqa: E402


def _seed_conflict(conn, ts, a_strength=1.0, b_strength=1.0):
    """a = newer challenger, b = older incumbent, with a 'contradicts' edge."""
    for cid, text in (("clm_a", S1), ("clm_b", S2)):
        emit(conn, UID, "seed", "CANONICALIZED",
             {"action": "new", "claim_id": cid, "text": text,
              "episode_id": "ep_x", "verbatim": text, "cluster": "main", "ts": ts})
    with conn:
        store.bump_claim_strength(conn, UID, "clm_a", ts, a_strength - 1.0)
        store.bump_claim_strength(conn, UID, "clm_b", ts, b_strength - 1.0)
    emit(conn, UID, "seed", "RELATED",
         {"from_id": "clm_a", "to_id": "clm_b", "relation": "contradicts",
          "weight": 1.0, "evidence_episode_id": "ep_x", "ts": ts})
    conn.commit()


def _mock_resolver(monkeypatch, mode, **quals):
    monkeypatch.setattr("core.consolidate._resolve_conflict",
                        lambda newer, older: ({"mode": mode, **quals}, 0.0))


def test_version_supersede_flips_past_margin(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts, a_strength=3.0, b_strength=1.0)   # challenger clears margin
    _mock_resolver(monkeypatch, "supersede")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    a, b = store.get_claim(conn, UID, "clm_a"), store.get_claim(conn, UID, "clm_b")
    assert a["status"] == "current"
    assert b["status"] == "superseded" and b["superseded_by"] == "clm_a"
    assert a["version_group"] and a["version_group"] == b["version_group"]  # neither dropped


def test_version_supersede_blocked_by_margin_holds_as_version(conn, fake_llm, monkeypatch):
    """A single contradicting note must NOT flip the current view (oscillation
    guard): sub-margin challenger is held as a version, incumbent stays current."""
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts, a_strength=1.0, b_strength=1.0)   # no margin
    _mock_resolver(monkeypatch, "supersede")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    a, b = store.get_claim(conn, UID, "clm_a"), store.get_claim(conn, UID, "clm_b")
    assert b["status"] == "current"        # incumbent holds
    assert a["status"] == "version"        # challenger held, not dropped
    assert a["version_group"] == b["version_group"]


def test_version_scope_keeps_both_with_qualifiers(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts)
    _mock_resolver(monkeypatch, "scope",
                   qualifier_newer="when remote", qualifier_older="when in office")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    a, b = store.get_claim(conn, UID, "clm_a"), store.get_claim(conn, UID, "clm_b")
    assert a["status"] == "current" and b["status"] == "current"  # both true, scoped
    assert a["qualifier"] == "when remote" and b["qualifier"] == "when in office"


def test_version_both_stand_holds_challenger(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts)
    _mock_resolver(monkeypatch, "version")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    a, b = store.get_claim(conn, UID, "clm_a"), store.get_claim(conn, UID, "clm_b")
    assert b["status"] == "current" and a["status"] == "version"


def test_version_decision_survives_rebuild(conn, fake_llm, monkeypatch):
    """The current/other split is frozen in the VERSIONED payload, so replay
    reproduces it without re-resolving (the resolver is not deterministic)."""
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts, a_strength=3.0, b_strength=1.0)
    _mock_resolver(monkeypatch, "supersede")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    monkeypatch.setattr("core.consolidate._resolve_conflict",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("resolver must not run at apply time")))
    rebuild(conn)
    assert store.get_claim(conn, UID, "clm_b")["status"] == "superseded"
    assert store.get_claim(conn, UID, "clm_a")["status"] == "current"


def test_contested_surfaced_at_retrieval(conn, fake_llm, monkeypatch):
    from core import recall
    ts = "2026-01-01T00:00:00+00:00"
    _seed_conflict(conn, ts, a_strength=3.0, b_strength=1.0)
    _mock_resolver(monkeypatch, "supersede")
    _reconcile(conn, UID, "run_v", ["clm_a", "clm_b"], ts)
    superseded = recall._score_node(conn, UID, "clm_b", 1.0, "seed")
    current = recall._score_node(conn, UID, "clm_a", 1.0, "seed")
    assert "⚖️ superseded" in superseded["signals"] and superseded["superseded_by"] == "clm_a"
    assert "⚖️ contested" in current["signals"]


# ── C10: store-integrity — derived claim faithful to its source episode ───────
from core.consolidate import _check_integrity  # noqa: E402


def _seed_claim_on_episode(conn, ep_id, claim_id, text, ts):
    emit(conn, UID, "seed", "CANONICALIZED",
         {"action": "new", "claim_id": claim_id, "text": text,
          "episode_id": ep_id, "verbatim": text, "cluster": "main", "ts": ts})
    conn.commit()


def _integrity_flags(conn):
    return [json.loads(e["payload_json"])
            for e in store.events_since(conn, UID, 0, types=["INTEGRITY_FLAGGED"])]


def test_integrity_faithful_claim_passes(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    ep = encode(conn, UID, f"{S1} {S2} {S3}", source="test")["episode_id"]
    _seed_claim_on_episode(conn, ep, "clm_f", S1, ts)   # verbatim of the source
    _check_integrity(conn, UID, "run_i", ["clm_f"], ts)
    assert _integrity_flags(conn) == []                 # grounded → no flag


def test_integrity_ungrounded_claim_flagged(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    ep = encode(conn, UID, f"{S1} {S2} {S3}", source="test")["episode_id"]
    _seed_claim_on_episode(
        conn, ep, "clm_u",
        "Quarterly corporate tax filing deadlines vary by jurisdiction.", ts)
    _check_integrity(conn, UID, "run_i", ["clm_u"], ts)
    flags = _integrity_flags(conn)
    assert len(flags) == 1 and flags[0]["kind"] == "ungrounded"
    assert flags[0]["claim_id"] == "clm_u"


def test_integrity_contradiction_flagged(conn, fake_llm, monkeypatch):
    """Geometry can't see a flipped polarity, so an AMBIGUOUS claim is sent to the
    resolver; a contradiction with the source is flagged. Route + stance are
    controlled (geometry tested in test_predict)."""
    ts = "2026-01-01T00:00:00+00:00"
    ep = encode(conn, UID, f"{S1} {S2} {S3}", source="test")["episode_id"]
    _seed_claim_on_episode(conn, ep, "clm_c", S1, ts)
    monkeypatch.setattr("core.predict.decide", lambda ms, calib=None: {
        "fragments": [{"route": "AMBIGUOUS", "anchor_text": S1, "anchor_id": "x"}]})
    monkeypatch.setattr("core.encode.classify_stance",
                        lambda premise, hyp: "contradiction")
    _check_integrity(conn, UID, "run_i", ["clm_c"], ts)
    flags = _integrity_flags(conn)
    assert len(flags) == 1 and flags[0]["kind"] == "contradicts_source"


def test_integrity_ambiguous_but_faithful_passes(conn, fake_llm, monkeypatch):
    ts = "2026-01-01T00:00:00+00:00"
    ep = encode(conn, UID, f"{S1} {S2} {S3}", source="test")["episode_id"]
    _seed_claim_on_episode(conn, ep, "clm_r", S1, ts)
    monkeypatch.setattr("core.predict.decide", lambda ms, calib=None: {
        "fragments": [{"route": "AMBIGUOUS", "anchor_text": S1, "anchor_id": "x"}]})
    monkeypatch.setattr("core.encode.classify_stance",
                        lambda premise, hyp: "entailment")   # refines, not contradicts
    _check_integrity(conn, UID, "run_i", ["clm_r"], ts)
    assert _integrity_flags(conn) == []


# ── C1: revisit order — most-surprising first ─────────────────────────────────
from core.consolidate import _revisit_order  # noqa: E402


def test_revisit_order_puts_surprise_first():
    eps = [
        {"id": "e1", "receipt_json": json.dumps({"n_novelties": 1, "contradictions": []})},
        {"id": "e2", "receipt_json": json.dumps(
            {"n_novelties": 0, "contradictions": [{"claim_id": "x"}]})},  # AMBIGUOUS
        {"id": "e3", "receipt_json": json.dumps({"n_novelties": 5, "contradictions": []})},
    ]
    order = [e["id"] for e in _revisit_order(eps)]
    assert order == ["e2", "e3", "e1"]  # contradiction first, then by novelty desc


# ── C13: consume retrieval signals — demote exposed-but-never-fetched ─────────
from core.consolidate import _consume_retrieval_signals, RETRIEVAL_EXPOSURE_MIN  # noqa: E402


def _seed_frag(conn, frag_id, episode_id):
    conn.execute(
        "INSERT INTO fragments (id, user_id, episode_id, text) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(id) DO NOTHING", (frag_id, UID, episode_id, "frag text"))


def _claim_on_episode(conn, claim_id, episode_id, ts):
    emit(conn, UID, "seed", "CANONICALIZED",
         {"action": "new", "claim_id": claim_id, "text": S1,
          "episode_id": episode_id, "verbatim": S1, "cluster": "main", "ts": ts})


def _signal(conn, *, fetched, dropped):
    store.append_event(conn, UID, "RETRIEVAL_SIGNAL",
                       {"query": "q", "fetched": fetched, "dropped": dropped,
                        "cut_for_budget": False})


def test_retrieval_signal_demotes_exposed_never_fetched(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    with conn:
        _seed_frag(conn, "frg_d", "ep_d")
        _claim_on_episode(conn, "clm_d", "ep_d", ts)
        for _ in range(RETRIEVAL_EXPOSURE_MIN):       # candidate N×, never fetched
            _signal(conn, fetched=[], dropped=["frg_d"])
    with conn:
        _consume_retrieval_signals(conn, UID, "run_s", ts)
    assert store.get_claim(conn, UID, "clm_d")["background"] == 1


def test_retrieval_signal_keeps_fetched_claim(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    with conn:
        _seed_frag(conn, "frg_f", "ep_f")
        _claim_on_episode(conn, "clm_f", "ep_f", ts)
        for _ in range(RETRIEVAL_EXPOSURE_MIN):
            _signal(conn, fetched=["frg_f"], dropped=[])
    with conn:
        _consume_retrieval_signals(conn, UID, "run_s", ts)
    assert store.get_claim(conn, UID, "clm_f")["background"] == 0   # fetched → kept


def test_retrieval_signal_spares_rare_quiet_claim(conn, fake_llm):
    """Below the exposure floor → not demoted (rare-but-correct guard)."""
    ts = "2026-01-01T00:00:00+00:00"
    with conn:
        _seed_frag(conn, "frg_r", "ep_r")
        _claim_on_episode(conn, "clm_r", "ep_r", ts)
        _signal(conn, fetched=[], dropped=["frg_r"])   # exposed once only
    with conn:
        _consume_retrieval_signals(conn, UID, "run_s", ts)
    assert store.get_claim(conn, UID, "clm_r")["background"] == 0


def test_retrieval_demote_survives_rebuild(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    with conn:
        _seed_frag(conn, "frg_d", "ep_d")
        _claim_on_episode(conn, "clm_d", "ep_d", ts)
        for _ in range(RETRIEVAL_EXPOSURE_MIN):
            _signal(conn, fetched=[], dropped=["frg_d"])
    with conn:
        _consume_retrieval_signals(conn, UID, "run_s", ts)
    rebuild(conn)
    assert store.get_claim(conn, UID, "clm_d")["background"] == 1


# ── C9: background decay — fold a recurrent claim into its theme ───────────────
from core.consolidate import _demote_background, BACKGROUND_MIN_REPEATS  # noqa: E402


def _seed_member_claim(conn, claim_id, n_support, ts):
    """A claim attached to a concept, supported by n_support episodes."""
    emit(conn, UID, "seed", "CANONICALIZED",
         {"action": "new", "claim_id": claim_id, "text": S1,
          "episode_id": "ep_0", "verbatim": S1, "cluster": "main", "ts": ts})
    for i in range(1, n_support):
        emit(conn, UID, "seed", "CANONICALIZED",
             {"action": "support", "claim_id": claim_id,
              "episode_id": f"ep_{i}", "verbatim": S1, "cluster": "main", "ts": ts})
    emit(conn, UID, "seed", "CONCEPT_CREATED",
         {"concept_id": "cpt_bg", "label": "bg", "canonical": "bg",
          "claim_ids": [claim_id], "ts": ts})
    conn.commit()


def test_background_folds_recurrent_member(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_member_claim(conn, "clm_bg", n_support=BACKGROUND_MIN_REPEATS + 1, ts=ts)
    with conn:
        _demote_background(conn, UID, "run_bg", ["clm_bg"], ts)
    assert store.get_claim(conn, UID, "clm_bg")["background"] == 1


def test_background_skips_rare_claim(conn, fake_llm):
    """A claim seen once (rare) is never demoted — the rare-correct guard."""
    ts = "2026-01-01T00:00:00+00:00"
    _seed_member_claim(conn, "clm_rare", n_support=1, ts=ts)
    with conn:
        _demote_background(conn, UID, "run_bg", ["clm_rare"], ts)
    assert store.get_claim(conn, UID, "clm_rare")["background"] == 0


def test_background_requires_a_theme(conn, fake_llm):
    """A recurrent claim with no concept has no theme to fold into → not demoted."""
    ts = "2026-01-01T00:00:00+00:00"
    emit(conn, UID, "seed", "CANONICALIZED",
         {"action": "new", "claim_id": "clm_orphan", "text": S1,
          "episode_id": "ep_0", "verbatim": S1, "cluster": "main", "ts": ts})
    for i in range(1, BACKGROUND_MIN_REPEATS + 1):
        emit(conn, UID, "seed", "CANONICALIZED",
             {"action": "support", "claim_id": "clm_orphan",
              "episode_id": f"ep_{i}", "verbatim": S1, "cluster": "main", "ts": ts})
    conn.commit()
    with conn:
        _demote_background(conn, UID, "run_bg", ["clm_orphan"], ts)
    assert store.get_claim(conn, UID, "clm_orphan")["background"] == 0


def test_background_demoted_at_retrieval(conn, fake_llm):
    from core import recall
    ts = "2026-01-01T00:00:00+00:00"
    _seed_member_claim(conn, "clm_bg", n_support=BACKGROUND_MIN_REPEATS + 1, ts=ts)
    before = recall._score_node(conn, UID, "clm_bg", 1.0, "seed")["score"]
    with conn:
        _demote_background(conn, UID, "run_bg", ["clm_bg"], ts)
    after = recall._score_node(conn, UID, "clm_bg", 1.0, "seed")
    assert after["score"] < before and "🌫️ background" in after["signals"]


def test_background_survives_rebuild(conn, fake_llm):
    ts = "2026-01-01T00:00:00+00:00"
    _seed_member_claim(conn, "clm_bg", n_support=BACKGROUND_MIN_REPEATS + 1, ts=ts)
    with conn:
        _demote_background(conn, UID, "run_bg", ["clm_bg"], ts)
    rebuild(conn)
    assert store.get_claim(conn, UID, "clm_bg")["background"] == 1


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
    encode(conn, UID, S1, source="test")
    encode(conn, UID, f"{S2} {S3}", source="test")
    report = consolidate(conn, UID)
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

    encode(conn, UID, S1, source="test")
    encode(conn, UID, S3, source="test")  # the poisoned note

    report = consolidate(conn, UID)
    assert report["status"] == "ok"
    assert report["episodes"] == 1
    assert len(report["skipped"]) == 1
    remaining = store.unconsolidated_episodes(conn, UID)
    assert len(remaining) == 1       # picked up by the next run
    assert S3 in remaining[0]["raw_text"]


def test_consolidate_all_users_covers_each_corpus(conn, fake_llm):
    from core.consolidate import consolidate_all_users
    from tests.conftest import UID_B
    encode(conn, UID, S1, source="test")
    encode(conn, UID_B, S3, source="test")
    reports = consolidate_all_users(conn)
    by_user = {r["user_id"]: r for r in reports}
    assert by_user[UID]["status"] == "ok" and by_user[UID]["episodes"] == 1
    assert by_user[UID_B]["status"] == "ok" and by_user[UID_B]["episodes"] == 1
    assert store.unconsolidated_episodes(conn, UID) == []
    assert store.unconsolidated_episodes(conn, UID_B) == []


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
    encode(conn, UID, S1, source="test")
    ep_id = conn.execute("SELECT id FROM episodes").fetchone()["id"]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE episodes SET title = 'x' WHERE id = ?", (ep_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM episodes WHERE id = ?", (ep_id,))


# ══════════════════════════════════════════════════════════════════════════════
# C2–C5 — measure()-based consolidation upgrades (synthetic planted geometry).
# These assert the GEOMETRY the upgrades route on (the plan §3 test column),
# deterministically and offline; the LLM/route SEMANTICS stay covered by the
# fake_llm consolidate() tests above.
# ══════════════════════════════════════════════════════════════════════════════
import numpy as np  # noqa: E402

import core.consolidate as Cmod  # noqa: E402
from core import predict  # noqa: E402


def _unit(v):
    return v / np.linalg.norm(v)


def _topic(seed):
    return _unit(np.random.default_rng(seed).standard_normal(384))


def _near(direction, seed, noise=0.04):
    return _unit(direction + noise * np.random.default_rng(seed).standard_normal(384))


def _plant_claims(conn, vecs, ts="2026-01-01T00:00:00Z", prefix="c"):
    """Insert claims with controlled vectors; returns their ids."""
    ids = []
    for i, v in enumerate(vecs):
        cid = f"{prefix}{i}"
        store.insert_claim(conn, UID, cid, f"{prefix} claim {i}", v, ts)
        ids.append(cid)
    return ids


# ── C2: dedup routes a planted dup → same, a distinct claim → new ─────────────
def test_c2_dedup_route_collapses_dup_keeps_distinct(conn):
    topic = _topic(1)
    # a populated region (> SPAN_K) so the spread-relative path engages
    members = [_near(topic, 100 + i) for i in range(10)]
    ids = _plant_claims(conn, members, prefix="m")

    dup = [{"text": "near dup of m0", "verbatim": None, "cluster": ""}]
    dup_emb = [_near(topic, 100, noise=0.005)]          # ~ identical to m0
    decided, uncertain = Cmod._dedup_route(conn, UID, dup, dup_emb,
                                           Cmod.DEDUP_CALIBRATION)
    assert decided and decided[0][1] == "same"           # collapsed onto a member
    assert decided[0][2] in ids

    far = [{"text": "unrelated", "verbatim": None, "cluster": ""}]
    far_emb = [_topic(999)]                               # orthogonal topic
    decided2, _ = Cmod._dedup_route(conn, UID, far, far_emb, Cmod.DEDUP_CALIBRATION)
    assert decided2 and decided2[0][1] == "new"          # genuinely distinct → new


# ── C3: nearest-cluster membership — in-topic plausible, off-topic not ────────
def test_c3_membership_z_separates_in_and_off_topic(conn):
    topic = _topic(2)
    members = [_near(topic, 200 + i) for i in range(10)]
    mids = _plant_claims(conn, members, prefix="t")
    cid = "cpt_t"
    store.insert_concept(conn, UID, cid, "topic", "the topic", "2026-01-01T00:00:00Z")
    for m in mids:
        store.add_concept_member(conn, UID, cid, m)

    z_in = Cmod._membership_z(conn, UID, _near(topic, 200, 0.04), "in", cid)
    z_off = Cmod._membership_z(conn, UID, _topic(888), "off", cid)
    assert z_in is not None and z_off is not None
    assert z_in <= Cmod.CONCEPT_MEMBERSHIP_Z < z_off     # in-topic fits, off-topic doesn't


# ── C4: spread test — bimodal splits, cohesive doesn't; medoid is central ─────
def test_c4_spread_is_bimodal():
    a, b = _topic(3), _topic(4)                          # two separated blobs
    V = np.vstack([_near(a, 300 + i) for i in range(5)] +
                  [_near(b, 400 + i) for i in range(5)])
    assert Cmod._spread_is_bimodal(V) is True
    one = np.vstack([_near(a, 500 + i) for i in range(10)])
    assert Cmod._spread_is_bimodal(one) is False


def test_c4_medoid_is_a_central_member():
    a = _topic(5)
    tight = [_near(a, 600 + i, noise=0.03) for i in range(8)]
    outlier = _unit(a + 0.9 * _topic(6))                 # far drifted member
    V = np.vstack(tight + [outlier])
    med = Cmod._medoid_vec(V)
    # the medoid is one of the tight members, never the outlier
    assert not np.array_equal(med, V[-1])
    assert float(med @ outlier) < float(med @ V[0])


# ── C5: bridge residual band — related→bridge, near-dup/unrelated→no ──────────
def test_c5_bridge_residual_band():
    a = _topic(7)
    region_a = np.vstack([_near(a, 700 + i) for i in range(6)])

    # near-duplicate concept: medoid ~ inside region_a → residual below band
    dup_medoid = _near(a, 700, noise=0.01)
    r_dup = float(predict.residuals_against(dup_medoid, region_a)[0])
    assert r_dup < Cmod.BRIDGE_RES_LOW

    # unrelated concept: orthogonal medoid → residual above band
    r_far = float(predict.residuals_against(_topic(770), region_a)[0])
    assert r_far > Cmod.BRIDGE_RES_HIGH

    # related-but-distinct: partial overlap → residual inside the band
    related = _unit(0.5 * a + 0.5 * _topic(771))
    r_rel = float(predict.residuals_against(related, region_a)[0])
    assert Cmod.BRIDGE_RES_LOW <= r_rel <= Cmod.BRIDGE_RES_HIGH


# ── C12: re-cluster fragments onto concepts + push baselines down ─────────────
from core import write  # noqa: E402

C12_NOTES = [
    "Spaced repetition is the most reliable way to retain knowledge over years. "
    "Active recall beats passive rereading for durable long-term memory.",
    "Memory consolidation happens during sleep when the brain replays experiences. "
    "Deep sleep is when the hippocampus hands memories to the cortex.",
    "Constraints often increase creativity rather than limiting what can be made. "
    "A tight brief forces sharper choices than a blank canvas does.",
]


def _seed_fragmented_corpus(conn):
    """Encode several notes and run the refine pass so fragments exist for C12."""
    for note in C12_NOTES:
        encode(conn, UID, note, source="test")
    write.refine_pending(conn, UID)


def test_c12_reclusters_fragments_and_pushes_baselines(conn, fake_llm):
    _seed_fragmented_corpus(conn)
    assert len(store.fragment_pool(conn, UID)) >= predict.WARMUP_MIN_CORPUS
    assert store.get_baselines(conn, UID) == {}        # nothing pushed pre-consolidation

    consolidate(conn, UID)

    concept_ids = {c["concept_id"] for c in store.concept_pool(conn, UID)}
    assert concept_ids                                  # concepts formed
    frags = store.fragment_pool(conn, UID)
    # every fragment is now scoped to a consolidated concept region (was anchor-bootstrap)
    assert all(f["cluster"] in concept_ids for f in frags)

    base = store.get_baselines(conn, UID)
    assert set(base) == {"clusters", "prior"}
    assert base["clusters"] and all(k in concept_ids for k in base["clusters"])
    assert len(base["prior"]) == 2                      # (mu, sd) prior-over-clusters


def test_c12_baselines_feed_the_next_write(conn, fake_llm, monkeypatch):
    """A write AFTER consolidation reads the pushed-down baselines instead of
    recomputing them over the corpus."""
    _seed_fragmented_corpus(conn)
    consolidate(conn, UID)
    pushed = store.get_baselines(conn, UID)
    assert pushed

    seen = {}
    real = write.route_fragments

    def spy(*a, **kw):
        seen["baselines"] = kw.get("baselines")
        return real(*a, **kw)

    monkeypatch.setattr(write, "route_fragments", spy)
    ep = encode(conn, UID, "Interleaving topics while studying improves retention.",
                source="test")["episode_id"]
    write.refine_episode(conn, UID, ep)
    assert seen["baselines"] == pushed                  # the pushed snapshot, not a recompute


def test_c12_recluster_survives_rebuild(conn, fake_llm):
    _seed_fragmented_corpus(conn)
    consolidate(conn, UID)
    before = {f["id"]: f["cluster"] for f in store.fragment_pool(conn, UID)}
    assert any(v is not None for v in before.values())
    rebuild(conn)
    after = {f["id"]: f["cluster"] for f in store.fragment_pool(conn, UID)}
    assert after == before                              # RECLUSTERED replays after FRAGMENTED


def test_c12_recluster_reverts_on_rollback(conn, fake_llm):
    _seed_fragmented_corpus(conn)
    bootstrap = {f["id"]: f["cluster"] for f in store.fragment_pool(conn, UID)}
    r = consolidate(conn, UID)
    reclustered = {f["id"]: f["cluster"] for f in store.fragment_pool(conn, UID)}
    assert reclustered != bootstrap                     # consolidation moved them onto concepts

    rollback_run(conn, r["run_id"])
    rebuild(conn)
    after = {f["id"]: f["cluster"] for f in store.fragment_pool(conn, UID)}
    assert after == bootstrap                            # rolled-back RECLUSTERED is excluded
