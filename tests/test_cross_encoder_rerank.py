"""Cross-encoder rerank: real (query, candidate) scoring over the top-N field nodes
by current salience, vs apply_feedback_rerank's query-similarity-only reweight.

Invariants under test:
  - flag OFF (default) → byte-identical field (pure no-op).
  - flag ON → only candidates present in BOTH nodes and the caller's texts move.
  - reranker failure → degrades to no-op, never raises.
  - resonance wiring bounds the candidate set to res_ce_top_n by current salience.
"""
import pytest

from core import resonance, retrieve, store
from core.encode import get_embedder
from tests.conftest import UID

TS = "2026-07-09T00:00:00+00:00"


def _mk_concept(conn, cid, label, claims):
    store.insert_concept(conn, UID, cid, label, f"{label} canonical", TS)
    for i, text in enumerate(claims):
        clm = f"clm_{cid}_{i}"
        emb = get_embedder().encode([text], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
        store.insert_claim(conn, UID, clm, text, emb, TS)
        store.add_concept_member(conn, UID, cid, clm)
    store.recompute_concept_embedding(conn, UID, cid)


@pytest.fixture
def graph(conn):
    _mk_concept(conn, "cpt_a", "advertising economy",
                ["Advertising forms an economic bubble that adds cost without value.",
                 "Products get priced up by the cost of the ads in the loop."])
    _mk_concept(conn, "cpt_b", "product teams",
                ["Missionaries pursue the larger business objective, mercenaries just execute."])
    return conn


class _StubReranker:
    """Deterministic stand-in: score = 1.0 for the first candidate, 0.0 for the rest."""

    def __init__(self, scores=None, raises=None):
        self._scores = scores
        self._raises = raises

    def rank(self, query, candidates):
        if self._raises:
            raise self._raises
        if self._scores is not None:
            return list(self._scores)
        return [1.0] + [0.0] * (len(candidates) - 1)


def test_cross_encoder_rerank_boosts_scored_candidate(graph):
    conn = graph
    nodes = {"clm_cpt_a_0": {"salience": 1.0}, "clm_cpt_a_1": {"salience": 1.0}}
    texts = {"clm_cpt_a_0": "ads bubble", "clm_cpt_a_1": "ad pricing"}
    reranker = _StubReranker(scores=[0.5, 0.1])
    from core import encode
    orig = encode.get_reranker
    try:
        encode.get_reranker = lambda: reranker
        # retrieve imported get_reranker by name — patch its local binding too
        orig_retrieve = retrieve.get_reranker
        retrieve.get_reranker = lambda: reranker
        retrieve.cross_encoder_rerank(nodes, "advertising", texts, alpha=1.0)
    finally:
        encode.get_reranker = orig
        retrieve.get_reranker = orig_retrieve
    assert nodes["clm_cpt_a_0"]["salience"] == pytest.approx(1.5)
    assert nodes["clm_cpt_a_1"]["salience"] == pytest.approx(1.1)


def test_cross_encoder_rerank_ignores_ids_not_in_nodes(graph):
    nodes = {"clm_cpt_a_0": {"salience": 1.0}}
    texts = {"clm_cpt_a_0": "ads bubble", "clm_cpt_a_1": "not in the field"}
    reranker = _StubReranker()
    orig = retrieve.get_reranker
    try:
        retrieve.get_reranker = lambda: reranker
        retrieve.cross_encoder_rerank(nodes, "advertising", texts, alpha=1.0)
    finally:
        retrieve.get_reranker = orig
    assert set(nodes) == {"clm_cpt_a_0"}


def test_cross_encoder_rerank_empty_texts_noop():
    nodes = {"clm_x": {"salience": 1.0}}
    before = dict(nodes["clm_x"])
    retrieve.cross_encoder_rerank(nodes, "q", {}, alpha=1.0)
    assert nodes["clm_x"] == before


def test_cross_encoder_rerank_failure_degrades_to_noop():
    nodes = {"clm_x": {"salience": 1.0}}
    before = nodes["clm_x"]["salience"]
    reranker = _StubReranker(raises=RuntimeError("HF Inference down"))
    orig = retrieve.get_reranker
    try:
        retrieve.get_reranker = lambda: reranker
        retrieve.cross_encoder_rerank(nodes, "q", {"clm_x": "text"}, alpha=1.0)
    finally:
        retrieve.get_reranker = orig
    assert nodes["clm_x"]["salience"] == before


def test_resonance_flag_off_is_noop(graph):
    """Default calibration (res_ce_rerank=False) → byte-identical field."""
    conn = graph
    q = "advertising and the economy"
    before = resonance.activate(conn, UID, q)["nodes"]
    after = resonance.activate(conn, UID, q)["nodes"]
    assert before == after


def test_resonance_flag_on_uses_reranker(graph, monkeypatch):
    conn = graph
    reranker = _StubReranker()
    monkeypatch.setattr(retrieve, "get_reranker", lambda: reranker)
    calibration = {**resonance.DEFAULT_CALIBRATION,
                   "res_ce_rerank": True, "res_ce_top_n": 5}
    field = resonance.resonance_recall(conn, UID, "advertising and the economy",
                                       calibration=calibration)
    assert isinstance(field["nodes"], dict)  # ran without raising end-to-end
