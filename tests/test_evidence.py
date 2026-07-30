"""Evidence lane (docs/evidence-lane-plan.md) — E1/E2/E3/E4/E5 + the P0 guard."""
import json

import pytest

from core import config, evidence, store, write
from core.consolidate import consolidate, rebuild
from core.encode import encode
from tests.conftest import UID
from tests.test_consolidate import S1, S2, _seed, fake_llm  # noqa: F401

# A "source" excerpt that warrants S1's general claim. Deliberately close in wording:
# the sweep only reaches a stance call above ECHO_THRESHOLD (0.72), and a genuinely
# distant paraphrase of S1 sits near the MiniLM NN-cosine median (~0.42–0.59). That
# gap is a property of the retrieval layer, not of this test — see the sweep docstring.
SOURCE_TEXT = ("Across 29 trials, spaced repetition was the most reliable method for "
               "retaining knowledge over multi-year intervals (Cepeda et al., 2006). "
               "Massed study matched it only at immediate test, never at follow-up.")


@pytest.fixture
def lane_on(monkeypatch):
    monkeypatch.setattr(config, "EVIDENCE_LANE", True)


def _seed_evidence(conn, uid=UID, text=SOURCE_TEXT, title="Spacing trial"):
    """Save a source the way save_evidence does, then fragment it — the nightly
    refine_pending is what makes a research episode consolidate into claims."""
    receipt = encode(conn, uid, text, source="research", title=title,
                     citation={"url": "https://example.org/spacing",
                               "title": title, "retrieved_at": "2026-07-30"})
    write.refine_episode(conn, uid, receipt["episode_id"])
    return receipt


# ── E1: citation column ───────────────────────────────────────────────────────
def test_citation_column_exists_on_a_fresh_db(conn):
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(episodes)")}
    assert "citation_json" in cols


def test_citation_column_is_added_to_an_existing_db(tmp_path, monkeypatch):
    """The house pattern: _ADD_COLUMNS, not a migration script."""
    db = tmp_path / "old.db"
    c = store.connect(db)
    c.execute("ALTER TABLE episodes DROP COLUMN citation_json")   # simulate a pre-E1 DB
    c.commit()
    c.close()
    c = store.connect(db)
    cols = {r["name"] for r in c.execute("PRAGMA table_info(episodes)")}
    c.close()
    assert "citation_json" in cols


def test_citation_never_enters_raw_text_or_sentences(conn):
    r = _seed_evidence(conn)
    ep = store.get_episode(conn, UID, r["episode_id"])
    assert ep["raw_text"] == SOURCE_TEXT           # citation is NOT prepended
    assert "example.org" not in ep["raw_text"]
    sents = [s["text"] for s in conn.execute(
        "SELECT text FROM episode_sentences WHERE episode_id = ?", (r["episode_id"],))]
    assert not any("example.org" in s for s in sents)   # no junk sentence minted
    assert store.episode_citation(conn, UID, r["episode_id"])["url"] == \
        "https://example.org/spacing"


# ── E2: stance direction ──────────────────────────────────────────────────────
def test_stance_direction_flips_for_research(monkeypatch):
    """Premise = the warrant, hypothesis = what is on trial. Assert BOTH directions:
    the whole point is that the unflipped call scores the same pair differently."""
    from core import encode as enc
    seen = []

    def fake(premise, hypothesis):
        seen.append((premise, hypothesis))
        return "entailment" if premise == SOURCE_TEXT else "neutral"

    monkeypatch.setattr(enc, "classify_stance", fake)
    assert enc.stance_for(S1, SOURCE_TEXT, is_evidence=True) == "entailment"
    assert seen[-1] == (SOURCE_TEXT, S1)      # evidence ⊨ claim
    assert enc.stance_for(S1, SOURCE_TEXT, is_evidence=False) == "neutral"
    assert seen[-1] == (S1, SOURCE_TEXT)      # note: stored claim is the premise


def test_stance_for_is_neutral_without_claim_text():
    from core import encode as enc
    assert enc.stance_for("", "anything", is_evidence=True) == "neutral"


def test_research_receipt_carries_source_and_stance(conn, monkeypatch):
    from core import encode as enc
    monkeypatch.setattr(enc, "classify_stance", lambda p, h: "entailment")
    encode(conn, UID, SOURCE_TEXT, source="research")
    r = encode(conn, UID, SOURCE_TEXT, source="research")   # now echoes itself
    assert r["source"] == "research"
    assert all("stance" in e for e in r["echoes"])


def test_note_receipt_has_no_stance_key(conn):
    r = encode(conn, UID, S1, source="mcp")
    assert r["source"] == "mcp"
    assert all("stance" not in e for e in r["echoes"])


# ── E7: prior matches know whose words they are ───────────────────────────────
def test_prior_match_reports_episode_source(conn):
    _seed_evidence(conn, UID, text=S2, title="A paper")
    r = encode(conn, UID, S2, source="mcp")     # a NOTE matching a research episode
    assert r["prior_episode_matches"]
    assert r["prior_episode_matches"][0]["episode_source"] == "research"


# ── E3: membership kind ───────────────────────────────────────────────────────
def test_evidence_claims_join_as_kind_evidence(conn, fake_llm, lane_on):
    _seed(conn, UID, S1)
    consolidate(conn, UID)
    before = conn.execute("SELECT embedding FROM vec_concepts").fetchall()

    _seed_evidence(conn)
    consolidate(conn, UID)

    kinds = {r["kind"] for r in conn.execute(
        "SELECT DISTINCT kind FROM concept_members WHERE user_id = ?", (UID,))}
    assert "evidence" in kinds
    # The evidence claim is a MEMBER but never a REPRESENTATIVE: every centre path
    # allowlists kind, so the concept vector must be byte-identical.
    after = conn.execute("SELECT embedding FROM vec_concepts").fetchall()
    assert [bytes(r["embedding"]) for r in after][:len(before)] == \
        [bytes(r["embedding"]) for r in before], "evidence leaked into a centre path"


def test_evidence_kind_is_off_when_the_lane_is_off(conn, fake_llm):
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    kinds = {r["kind"] for r in conn.execute(
        "SELECT DISTINCT kind FROM concept_members WHERE user_id = ?", (UID,))}
    assert "evidence" not in kinds


def test_a_claim_the_user_also_wrote_stays_theirs(conn, fake_llm, lane_on):
    """Origin is joined, not stored: support from BOTH a note and a source reads as
    the user's — a source's words never become the user's, and vice versa."""
    _seed(conn, UID, S1)
    _seed_evidence(conn, UID, text=S1, title="A paper restating it")
    consolidate(conn, UID)
    claim_id = conn.execute("SELECT id FROM claims WHERE user_id = ? LIMIT 1",
                            (UID,)).fetchone()["id"]
    assert store.claim_support_count(conn, UID, claim_id) == 2
    assert not store.is_evidence_claim(conn, UID, claim_id)


# ── E4: the sweep ─────────────────────────────────────────────────────────────
@pytest.fixture
def stance_counter(monkeypatch):
    """Count stance calls — the watermark's only observable property."""
    calls = {"n": 0}

    def fake(premise, hypothesis):
        calls["n"] += 1
        return "entailment"

    monkeypatch.setattr(evidence, "classify_stance", fake)
    return calls


def test_sweep_is_a_noop_without_evidence(conn, lane_on, stance_counter):
    r = evidence.sweep(conn, UID)
    assert r["status"] == "noop"
    assert stance_counter["n"] == 0


def test_sweep_attaches_evidence_to_a_claim(conn, fake_llm, lane_on, stance_counter):
    _seed(conn, UID, S1)
    consolidate(conn, UID)
    _seed_evidence(conn)
    consolidate(conn, UID)   # runs the sweep at the end of the run

    rows = conn.execute("SELECT * FROM evidence_attachments WHERE user_id = ?",
                        (UID,)).fetchall()
    assert rows, "the source should attach to the claim it warrants"
    assert rows[0]["stance"] == "entailment"
    assert rows[0]["similarity"] >= config.ECHO_THRESHOLD


def test_second_sweep_does_zero_stance_calls(conn, fake_llm, lane_on, stance_counter):
    """The watermark, stated as a test: an unchanged corpus costs nothing."""
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)
    baseline = stance_counter["n"]
    with conn:
        second = evidence.sweep(conn, UID)
    assert stance_counter["n"] == baseline, "re-sweep re-evaluated settled pairs"
    assert second["pairs"] == 0
    assert second["changed_claims"] == 0


def test_a_new_claim_only_evaluates_the_affected_pairs(conn, fake_llm, lane_on,
                                                       stance_counter):
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)
    before = stance_counter["n"]

    _seed(conn, UID, S2)          # one new claim, no new evidence
    consolidate(conn, UID)
    with conn:
        r = evidence.sweep(conn, UID)
    # Work is bounded by the CHANGED claims, never by (all evidence × all claims).
    assert r["new_evidence"] == 0
    assert stance_counter["n"] - before <= r["changed_claims"] * 5


def test_sweep_refuses_a_per_pair_billed_provider(conn, lane_on, monkeypatch):
    monkeypatch.setattr(config, "STANCE_PROVIDER", "haiku")
    with pytest.raises(RuntimeError, match="bills per pair"):
        evidence.sweep(conn, UID)


def test_sweep_is_off_when_the_lane_is_off(conn):
    assert evidence.sweep(conn, UID)["status"] == "off"


def test_pair_cap_defers_instead_of_silently_truncating(conn, fake_llm, lane_on,
                                                        stance_counter):
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    conn.execute("DELETE FROM evidence_attachments WHERE user_id = ?", (UID,))
    store.set_evidence_watermark(conn, UID, 0, store.now_iso())
    with conn:
        r = evidence.sweep(conn, UID, max_pairs=0)
    assert r["deferred"], "a capped run must NAME what it dropped"
    # watermark not advanced → the deferred work is picked up next run
    assert store.evidence_watermark(conn, UID) == 0


def test_attachments_survive_rebuild(conn, fake_llm, lane_on, stance_counter):
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)
    before = [tuple(r) for r in conn.execute(
        "SELECT evidence_episode_id, sentence_idx, claim_id, stance "
        "FROM evidence_attachments ORDER BY 1,2,3")]
    assert before

    rebuild(conn)   # truncates the derived table, replays the log

    after = [tuple(r) for r in conn.execute(
        "SELECT evidence_episode_id, sentence_idx, claim_id, stance "
        "FROM evidence_attachments ORDER BY 1,2,3")]
    assert after == before


def test_the_event_carries_claim_text_not_just_the_id(conn, fake_llm, lane_on,
                                                      stance_counter):
    """Claim ids are md5 of the text — a pinned id dangles on re-canonicalisation,
    so the durable record has to be the text (same reason MERGED stores snapshots)."""
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)
    events = store.events_since(conn, UID, 0, types=["EVIDENCE_ATTACHED"])
    payloads = [json.loads(e["payload_json"]) for e in events]
    attached = [a for p in payloads for a in p["attachments"]]
    assert attached
    assert all(a.get("claim_text") for a in attached)


def test_attachment_refreshes_last_seen_but_not_strength(conn, fake_llm, lane_on,
                                                         stance_counter):
    """Decay: no exemption for evidence — refresh on usage or a new attachment.
    last_seen only; bumping strength per attachment would degree-boost the target."""
    _seed(conn, UID, S1)
    ev = _seed_evidence(conn)
    consolidate(conn, UID)
    ids = store.claims_for_episode(conn, UID, ev["episode_id"])
    if not ids:
        pytest.skip("the fake blueprint minted no claim for the source")
    before = store.get_claim(conn, UID, ids[0])
    conn.execute("UPDATE claims SET last_seen = '2000-01-01' WHERE id = ?", (ids[0],))
    store.touch_claim(conn, UID, ids[0], "2026-07-30T00:00:00+00:00")
    after = store.get_claim(conn, UID, ids[0])
    assert after["last_seen"] == "2026-07-30T00:00:00+00:00"
    assert after["strength"] == before["strength"]


def test_member_ratios_reports_the_volume_asymmetry(conn, fake_llm, lane_on,
                                                    stance_counter):
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    ratios = evidence.member_ratios(conn, UID)
    assert all("n_evidence" in r and "n_self" in r for r in ratios)


# ── E5: three labels ──────────────────────────────────────────────────────────
def test_standalone_evidence_reads_as_might(conn, lane_on):
    from core.recall import EVIDENCE_STANDALONE, label_evidence
    results = [{"type": "claim", "id": "clm_x", "origin": "evidence"}]
    label_evidence(conn, UID, results)
    assert results[0]["origin_label"] == EVIDENCE_STANDALONE


def test_a_contradiction_never_renders_as_support(conn, lane_on):
    """The one genuinely harmful outcome: ⚡ collapsed into 📎."""
    from core.recall import EVIDENCE_LABELS, label_evidence
    ts = store.now_iso()
    r = encode(conn, UID, SOURCE_TEXT, source="research")
    ep = r["episode_id"]
    assert r["n_sentences"] >= 2, "fixture needs two sentences to hold two verdicts"
    store.insert_claim(conn, UID, "clm_self", S1, [0.0] * config.EMBED_DIM, ts)
    conn.execute("INSERT INTO claim_support (claim_id, user_id, episode_id) "
                 "VALUES ('clm_ev', ?, ?)", (UID, ep))
    # both verdicts present for the same pairing — contradiction must win
    store.add_evidence_attachment(conn, UID, ep, 0, "clm_self", "entailment", 0.9, ts)
    store.add_evidence_attachment(conn, UID, ep, 1, "clm_self", "contradiction", 0.8, ts)
    results = [{"type": "claim", "id": "clm_self"},
               {"type": "claim", "id": "clm_ev", "origin": "evidence"}]
    label_evidence(conn, UID, results)
    assert results[1]["origin_label"] == EVIDENCE_LABELS["contradiction"]
    assert results[1]["verdict"]["stance"] == "contradiction"


def test_origin_label_does_not_collide_with_a_concept_label(conn, fake_llm, lane_on):
    """A concept headline's `label` is its NAME. The provenance badge must not
    overwrite it."""
    from core.recall import recall
    _seed(conn, UID, S1)
    consolidate(conn, UID)
    concepts = [r for r in recall(conn, UID, S1) if r["type"] == "concept"]
    assert concepts and concepts[0]["label"] == "test concept"


def test_recall_labels_are_absent_when_the_lane_is_off(conn, fake_llm):
    from core.recall import recall
    _seed(conn, UID, S1)
    consolidate(conn, UID)
    for r in recall(conn, UID, S1):
        assert "origin_label" not in r
        assert "origin" not in r


# ── E5: budget partition in the prod retrieval head ───────────────────────────
def test_no_evidence_means_a_byte_identical_render(conn, fake_llm, lane_on):
    """The gate: with no research episode the partition must not exist at all."""
    from core import resonance
    _seed(conn, UID, S1)
    _seed(conn, UID, S2)
    consolidate(conn, UID)
    with_lane = resonance.resonance_context(conn, UID, S1)
    config.EVIDENCE_LANE = False
    try:
        without = resonance.resonance_context(conn, UID, S1)
    finally:
        config.EVIDENCE_LANE = True
    assert with_lane == without


def test_flag_off_leaves_the_retrieval_path_untouched(conn, fake_llm, lane_on,
                                                      stance_counter):
    """EVIDENCE_LANE defaults to OFF, so this is the rollback story: with the flag
    down, a corpus that CONTAINS evidence renders exactly as it did before the lane
    existed — no section, no partition, no relabelling."""
    from core import resonance
    from core.recall import recall
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)

    config.EVIDENCE_LANE = False
    try:
        md = resonance.resonance_context(conn, UID, S1)
        hits = recall(conn, UID, S1)
    finally:
        config.EVIDENCE_LANE = True
    assert "### Evidence" not in md
    assert all("origin" not in h and "origin_label" not in h for h in hits)


def test_evidence_gets_its_own_section_and_slice(conn, fake_llm, lane_on,
                                                 stance_counter):
    from core import resonance
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)

    md = resonance.resonance_context(conn, UID, S1, max_chars=6000)
    assert "### Evidence" in md
    # The source's text is under Evidence, never inside the user's own Specifics.
    ev_part = md.split("### Evidence", 1)[1]
    self_part = md.split("### Evidence", 1)[0]
    assert "Cepeda" in ev_part
    assert "Cepeda" not in self_part.split("### Specifics", 1)[-1]


def test_the_evidence_slice_is_bounded(conn, fake_llm, lane_on, stance_counter):
    """Evidence gets a SHARE of B, it does not bid freely — that is what keeps
    self-lane coverage measurable against the existing gold sets."""
    from core import resonance
    _seed(conn, UID, S1)
    for i in range(6):                       # lots of evidence, one note
        _seed_evidence(conn, title=f"source {i}")
    consolidate(conn, UID)
    with conn:
        evidence.sweep(conn, UID)

    max_chars = 4000
    md = resonance.resonance_context(conn, UID, S1, max_chars=max_chars)
    if "### Evidence" not in md:
        pytest.skip("no evidence fragment reached the field")
    ev_chars = len(md.split("### Evidence", 1)[1])
    assert ev_chars <= max_chars * config.EVIDENCE_SHARE + 400


def test_a_source_with_no_verdict_says_might_not_does(conn, fake_llm, lane_on):
    """Asserting a relationship that was never computed would be a fabricated
    citation, so a standalone source may only say what it MIGHT do."""
    from core import resonance
    _seed(conn, UID, S1)
    _seed_evidence(conn)
    consolidate(conn, UID)          # no sweep run → no verdicts exist
    conn.execute("DELETE FROM evidence_attachments WHERE user_id = ?", (UID,))
    md = resonance.resonance_context(conn, UID, S1)
    if "### Evidence" not in md:
        pytest.skip("no evidence fragment reached the field")
    assert "might back you" in md
    assert "refutes" not in md.split("### Evidence", 1)[1]


# ── Deploy config: the lane must not ship disabled ────────────────────────────
def test_compose_enables_the_lane_in_production():
    """`.env.example` ships EVIDENCE_LANE=0 for local/library use, so if the flag
    lived only in .env a routine `cp .env.example .env` during a rebuild would
    silently ship the feature off. Compose's `environment:` overrides `env_file` and
    is version-controlled, which is why the production default lives there."""
    from pathlib import Path
    compose = (Path(__file__).parent.parent / "docker-compose.yml").read_text()
    assert "EVIDENCE_LANE=${EVIDENCE_LANE:-1}" in compose, \
        "production must default the evidence lane ON (see DEPLOY.md §8)"
    env_block = compose.split("environment:", 1)[1]
    assert "EVIDENCE_LANE" in env_block.split("restart:", 1)[0], \
        "must be under `environment:` (which overrides env_file), not `env_file`"


def test_health_reports_the_lane_and_stance_together():
    """Both are needed to read a deploy: the lane ON with a degraded stance provider
    means every verdict collapses to 'neutral' and 📎/⚡ never fire."""
    import server
    body = server.health()
    assert "evidence_lane" in body and "stance" in body
    assert set(body["stance"]) >= {"provider", "ok", "detail"}


# ── Gate A runner ─────────────────────────────────────────────────────────────
def test_gate_a_reads_the_verdict_off_the_shipped_receipt_fields():
    """The runner must score what the receipt RENDERS, not a parallel reimplementation
    — otherwise the gate can pass while the user sees something else."""
    from eval.evidence_gate import BACKS, ORPHAN, REFUTES, RELATES, _observed
    assert _observed({"contradictions": [{}], "echoes": [{"stance": "entailment"}]}) \
        == REFUTES                      # a refutation is never masked by co-occurring support
    assert _observed({"contradictions": [], "echoes": [{"stance": "entailment"},
                                                       {"stance": "neutral"}]}) == BACKS
    assert _observed({"contradictions": [], "echoes": [{"stance": "neutral"}]}) == RELATES
    assert _observed({"contradictions": [], "echoes": []}) == ORPHAN


def test_gate_a_gold_is_well_formed():
    import json
    from pathlib import Path
    rows = [json.loads(l) for l in
            (Path(__file__).parent.parent / "eval" / "gold_evidence.jsonl")
            .read_text().splitlines() if l.strip()]
    assert len(rows) >= 10, "the plan's gate is 10-15 hand-fed sources"
    assert {r["id"] for r in rows}.__len__() == len(rows)
    assert all(r["expect"] in ("backs", "refutes", "relates", "orphan") for r in rows)
    assert all(r["source_url"] and r["source_title"] for r in rows)
    # both failure modes must be measurable: near-field (reaches the classifier) and
    # far-field (real source phrasing that tests the ECHO_THRESHOLD reach limit)
    assert {r["field"] for r in rows} == {"near", "far"}
    assert any(r["expect"] == "refutes" for r in rows)


# ── P0 ────────────────────────────────────────────────────────────────────────
def test_stance_health_flags_nli_without_torch(monkeypatch):
    import builtins
    from core.encode import stance_health
    monkeypatch.setattr(config, "STANCE_PROVIDER", "nli")
    real_import = builtins.__import__

    def no_st(name, *a, **kw):
        if name == "sentence_transformers":
            raise ImportError("No module named 'sentence_transformers'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_st)
    h = stance_health()
    assert h["ok"] is False
    assert "STANCE_PROVIDER=hf" in h["detail"]


def test_stance_health_flags_hf_without_a_token(monkeypatch):
    from core.encode import stance_health
    monkeypatch.setattr(config, "STANCE_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_TOKEN", "")
    assert stance_health()["ok"] is False


def test_stance_health_probe_catches_a_present_but_dead_credential(monkeypatch):
    """A token that exists but cannot buy a call must read as BROKEN.

    Checking HF_TOKEN non-empty reported ok=True on prod while every call returned
    402 Payment Required and classify_stance degraded every pair to "neutral" —
    the original silent failure showing green on /health.
    """
    from core import encode as enc

    monkeypatch.setattr(config, "STANCE_PROVIDER", "hf")
    monkeypatch.setattr(config, "HF_TOKEN", "looks-fine-but-broke")
    monkeypatch.setattr(enc, "classify_stance", lambda p, h: "neutral")  # degraded
    h = enc.stance_health()
    assert h["ok"] is False
    assert "quota" in h["detail"] and "neutral" in h["detail"]

    # and it passes when the provider genuinely works
    monkeypatch.setattr(enc, "classify_stance", lambda p, h: "contradiction")
    assert enc.stance_health()["ok"] is True
    # probe=False must NOT be mistaken for a working provider
    monkeypatch.setattr(enc, "classify_stance", lambda p, h: "neutral")
    assert "unprobed" in enc.stance_health(probe=False)["detail"]


def test_openrouter_stance_pins_the_fallback_chain(monkeypatch):
    """The free rung failing must NOT escalate to the paid API — that escalation at
    sweep volume is the documented way this account got drained."""
    from core import encode as enc, llm

    seen = {}
    monkeypatch.setattr(config, "STANCE_PROVIDER", "openrouter")
    monkeypatch.setattr(config, "LLM_FALLBACK_ORDER", ["openrouter", "claude", "gemini"])

    def fake_call(prompt, tier=None, max_tokens=None, system=None):
        seen["order"] = list(config.LLM_FALLBACK_ORDER)
        return {"json": {"stance": "contradiction"}, "cost": 0.0}

    monkeypatch.setattr(llm, "call", fake_call)
    assert enc.classify_stance("a", "b") == "contradiction"
    assert seen["order"] == ["openrouter"]                       # pinned during the call
    assert config.LLM_FALLBACK_ORDER == ["openrouter", "claude", "gemini"]  # and restored


def test_sweep_guard_allows_pinned_openrouter_but_not_haiku(monkeypatch):
    from core.evidence import _stance_budget_guard

    monkeypatch.setattr(config, "STANCE_PROVIDER", "haiku")
    with pytest.raises(RuntimeError, match="bills per pair"):
        _stance_budget_guard()

    for p in ("openrouter", "hf", "nli"):
        monkeypatch.setattr(config, "STANCE_PROVIDER", p)
        _stance_budget_guard()                                   # must not raise
