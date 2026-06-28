"""Frag+concept HYBRID retrieval (core/hybrid.py, plan §2 / §6 P4).

Offline geometry/plumbing over the real local embedder + a temp DB: the LLM
answer/judge is R9 (the SR@B harness owns that). Asserts the orchestration the
blend must get right — both paths surfaced, budget honoured, empty-path fallback,
user isolation — without tuning the split (that's the C12 SR@B sweep).

Fragments come from encode→refine; the concept path needs a claim, which the test
DB has none of (consolidation builds those), so the relevant tests insert one
directly via store.
"""
from core import hybrid, store
from core.encode import encode, get_embedder
from core.write import refine_episode
from tests.conftest import UID, UID_B
from tests.test_retrieve import NOTE_AI, NOTE_FARMING, NOTE_RELIGION, _seed

CLAIM_AI = "Memory is the moat for AI because it creates a monetisable switching cost."


def _insert_claim(conn, uid, text, ts="2026-06-01T00:00:00"):
    """Plant one claim (+ vector) so the concept path is non-empty — stands in for
    a consolidation run the integration DB doesn't have."""
    emb = get_embedder().encode([text], normalize_embeddings=True,
                                show_progress_bar=False)[0]
    cid = "clm_test_ai"
    store.insert_claim(conn, uid, cid, text, emb, ts)
    ep = encode(conn, uid, "seed note for support", source="test")["episode_id"]
    store.add_claim_support(conn, uid, cid, ep, text)
    return cid


def test_hybrid_blends_both_paths(conn):
    """Both paths present → one header, a Synthesis section (claim) AND a Source
    spans section (fragment), so the blend carries tail + specificity together."""
    _seed(conn)
    _insert_claim(conn, UID, CLAIM_AI)
    ctx = hybrid.hybrid_context(conn, UID, "AI memory paradigm", max_chars=10_000)
    assert ctx.count("## Slate context:") == 1          # single top header
    assert "### Synthesis" in ctx and "### Source spans" in ctx
    assert "moat" in ctx.lower()                         # the planted claim
    assert "memory" in ctx.lower()                       # a fragment span


def test_hybrid_falls_back_to_fragments_when_no_concepts(conn):
    """No claims/concepts (the default test DB) → pure fragment context, no empty
    Synthesis section, and the full budget went to fragments."""
    _seed(conn)
    ctx = hybrid.hybrid_context(conn, UID, "AI memory paradigm")
    assert "### Synthesis" not in ctx
    assert "nothing stored" not in ctx
    assert "memory" in ctx.lower()


def test_hybrid_falls_back_to_concepts_when_no_fragments(conn):
    """Claim present but no fragments → the concept path carries it; no Source-spans
    section, no padding to budget with nothing."""
    _insert_claim(conn, UID, CLAIM_AI)
    ctx = hybrid.hybrid_context(conn, UID, "AI memory moat")
    assert "### Source spans" not in ctx
    assert "moat" in ctx.lower()


def test_hybrid_runs_fragment_path_once_when_concepts_empty(conn, monkeypatch):
    """No claims → fragments take the whole budget, but the fragment path (heavy
    assembly + R8 signals=ON) must run EXACTLY ONCE, not be sized then re-run at full
    budget — a double run double-emits R8, inflating C13 exposure counts."""
    _seed(conn)
    calls = {"n": 0}
    real = hybrid.retrieve.assemble_context

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(hybrid.retrieve, "assemble_context", counting)
    ctx = hybrid.hybrid_context(conn, UID, "AI memory paradigm")
    assert calls["n"] == 1
    assert "memory" in ctx.lower()


def test_hybrid_nothing_stored(conn):
    """Both paths empty → the explicit sentinel, never a padded guess."""
    msg = hybrid.hybrid_context(conn, UID, "quantum gardening on mars")
    assert "nothing stored" in msg.lower()


def test_hybrid_respects_char_budget(conn):
    """A tiny budget truncates and says so (sufficiency @ budget, synthesis-first)."""
    _seed(conn)
    _insert_claim(conn, UID, CLAIM_AI)
    small = hybrid.hybrid_context(conn, UID, "AI memory paradigm", max_chars=300)
    assert len(small) <= 360                             # budget + truncation marker
    assert "truncated" in small


def test_hybrid_is_user_isolated(conn):
    """Neither path crosses users (AUTH.md §1/§3): UID_B sees nothing of UID's data."""
    _seed(conn, UID)
    _insert_claim(conn, UID, CLAIM_AI)
    msg = hybrid.hybrid_context(conn, UID_B, "AI memory paradigm")
    assert "nothing stored" in msg.lower()
