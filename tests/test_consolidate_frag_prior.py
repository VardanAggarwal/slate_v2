"""Fragment→consolidate PRIOR (plan §3): Write's fragment routing informs C2 dedup
and C3 membership, WITHOUT becoming the source of truth.

The bridge is `consolidate._anchor_concept_prior`: claim → nearest same-episode
fragment (over the medoid vectors Write already built) → that fragment's AMBIGUOUS
anchor → the anchor's episode → its claims → their concept. It is a HINT only — the
predictor (C2) / LLM (C3) still decide, and it degrades to a no-op when an episode has
no materialized fragments. These tests pin both the bridge and the two injection sites.
"""
from core import consolidate, store
from core.encode import encode, get_embedder
from tests.conftest import UID


def _emb(text):
    return get_embedder().encode([text], normalize_embeddings=True,
                                 show_progress_bar=False)[0]


def _setup_anchor_chain(conn, uid=UID):
    """epB holds claim clmB in concept cnc_x; epA's span anchors (AMBIGUOUS) onto
    epB's fragment. So the prior for a claim from epA is cnc_x."""
    epB = encode(conn, uid, "Memory is the moat for AI products.", source="test")["episode_id"]
    fragB = store.fragment_id_for(uid, epB, 0, 0)
    store.insert_fragment(conn, uid, fragB, epB, "Memory is the moat.", 0, 0,
                          "NOVEL", 0.0, 0.0, 1.0, None, None, False, False,
                          "2026-06-01T00:00:00", medoid_idx=0)
    store.insert_claim(conn, uid, "clm_b", "Memory is the moat for AI.",
                       _emb("Memory is the moat for AI."), "2026-06-01T00:00:00")
    store.add_claim_support(conn, uid, "clm_b", epB, None)
    store.insert_concept(conn, uid, "cnc_x", "AI memory moat",
                         "Memory is the moat", "2026-06-01T00:00:00")
    store.add_concept_member(conn, uid, "cnc_x", "clm_b")

    epA = encode(conn, uid, "The switching cost from accumulated memory is the moat.",
                 source="test")["episode_id"]
    fragA = store.fragment_id_for(uid, epA, 0, 0)
    store.insert_fragment(conn, uid, fragA, epA, "Switching cost is the moat.", 0, 0,
                          "AMBIGUOUS", 1.0, 0.5, 1.0, fragB, None, False, False,
                          "2026-06-02T00:00:00", medoid_idx=0)
    return epA, epB


# ── the bridge ────────────────────────────────────────────────────────────────
def test_prior_returns_anchored_concept(conn):
    epA, _ = _setup_anchor_chain(conn)
    assert consolidate._anchor_concept_prior(conn, UID, epA, _emb("anything")) == "cnc_x"


def test_prior_none_without_fragments(conn):
    _setup_anchor_chain(conn)
    bare = encode(conn, UID, "Unrelated note about gardening on weekends.",
                  source="test")["episode_id"]
    assert consolidate._anchor_concept_prior(conn, UID, bare, _emb("x")) is None


def test_prior_none_for_novel_span(conn):
    _, epB = _setup_anchor_chain(conn)
    # epB's only fragment is NOVEL (no anchor) → nothing to inherit.
    assert consolidate._anchor_concept_prior(conn, UID, epB, _emb("x")) is None
    assert consolidate._anchor_concept_prior(conn, UID, None, _emb("x")) is None


# ── C2 dedup injection ──────────────────────────────────────────────────────────
def test_c2_prior_catches_dup_global_knn_missed(conn, monkeypatch):
    """knn under-ranks the dup (simulated empty), but the source span's anchored
    concept still surfaces clmB → routes 'same', not a blind 'new'. The fix to
    concept fragmentation/duplication that the prior is for."""
    epA, _ = _setup_anchor_chain(conn)
    monkeypatch.setattr(store, "knn_claims", lambda *a, **k: [])  # global knn whiffs
    c = {"text": "Memory is the moat for AI.", "verbatim": None, "cluster": ""}
    emb = _emb(c["text"])

    decided, uncertain = consolidate._dedup_route(
        conn, UID, [c], [emb], consolidate.DEDUP_CALIBRATION, episode_id=epA)
    assert decided and decided[0][1] == "same" and decided[0][2] == "clm_b"

    # No episode → no prior → the blind 'new' the old code would emit.
    d2, _ = consolidate._dedup_route(
        conn, UID, [c], [emb], consolidate.DEDUP_CALIBRATION, episode_id=None)
    assert d2[0][1] == "new"


# ── C3 membership injection ──────────────────────────────────────────────────────
def test_c3_prior_surfaces_concept_global_knn_missed(conn, monkeypatch):
    """knn returns no candidate concepts, yet the anchored concept is offered to the
    attach/split LLM — so the claim can join its theme instead of spawning a duplicate."""
    epA, _ = _setup_anchor_chain(conn)
    store.insert_claim(conn, UID, "clm_new", "The moat is the switching cost.",
                       _emb("The moat is the switching cost."), "2026-06-02T00:00:00")
    store.add_claim_support(conn, UID, "clm_new", epA, None)

    monkeypatch.setattr(store, "knn_concepts", lambda *a, **k: [])  # global knn whiffs
    captured = {}

    def fake_llm(prompt, **k):
        captured["prompt"] = prompt
        return {"json": {"decisions": []}, "cost": 0.0}

    monkeypatch.setattr(consolidate.llm, "call", fake_llm)
    consolidate._concept_pass_chunk(conn, UID, "run_test", ["clm_new"],
                                    "2026-06-02T00:00:00")
    assert "AI memory moat" in captured["prompt"]   # cnc_x label, surfaced by the prior
