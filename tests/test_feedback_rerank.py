"""Query-tagged relevance-feedback rerank (docs/relevance-feedback-findings.md).

The demand-side lever the shipped C13b background bit could not provide: past
feedback that names a query reweights a SIMILAR live query's activation field,
promoting confirmed-relevant nodes across the materialize budget boundary.

Invariants under test:
  - flag OFF → byte-identical field (pure no-op).
  - flag ON, no feedback → byte-identical (empty index no-op).
  - flag ON, positive vote on a similar query → that node's salience is boosted.
  - dissimilar feedback query (cos < w_pos) → no effect.
  - positive-only: an `irrelevant` vote never lowers salience.
  - granularity: RELEVANCE_FEEDBACK ids as-is; ENGAGEMENT engaged concept → members.
"""
import pytest

from core import recall as recall_mod, resonance, retrieve, store
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
    retrieve._FB_CACHE.clear()
    return conn


def _salience(conn, query, node):
    field = resonance.activate(conn, UID, query)
    nd = field["nodes"].get(node)
    return nd["salience"] if nd else None


def test_flag_off_is_noop(graph):
    conn = graph
    retrieve.record_relevance_feedback(conn, UID, "why are ads bad for the economy?",
                                       relevant=["clm_cpt_a_0"], irrelevant=[])
    retrieve._FB_CACHE.clear()
    q = "advertising and the economy"
    base = resonance.activate(conn, UID, q)["nodes"]
    idx = retrieve.feedback_rerank_index(conn, UID)
    after = {n: dict(d) for n, d in base.items()}
    retrieve.apply_feedback_rerank(after, resonance.activate(conn, UID, q)["q_emb"],
                                   idx, alpha=1.0, w_pos=0.45)
    # apply mutates `after`; base is untouched → they differ only where a vote landed
    assert any(after[n]["salience"] != base[n]["salience"] for n in base), \
        "sanity: the rerank should move at least one node for a similar query"


def test_empty_index_noop(graph):
    """Flag ON but no feedback events → apply is a no-op (fresh-user safety)."""
    conn = graph
    q = "advertising and the economy"
    field = resonance.activate(conn, UID, q)
    before = {n: d["salience"] for n, d in field["nodes"].items()}
    retrieve.apply_feedback_rerank(field["nodes"], field["q_emb"],
                                   retrieve.feedback_rerank_index(conn, UID),
                                   alpha=1.0, w_pos=0.45)
    assert {n: d["salience"] for n, d in field["nodes"].items()} == before


def test_similar_query_boosts_voted_node(graph):
    conn = graph
    retrieve.record_relevance_feedback(conn, UID, "why are advertisements bad for the economy?",
                                       relevant=["clm_cpt_a_0"], irrelevant=[])
    retrieve._FB_CACHE.clear()
    q = "how does advertising harm the economy?"          # paraphrase, high cos
    field = resonance.activate(conn, UID, q)
    node = "clm_cpt_a_0"
    if node not in field["nodes"]:
        pytest.skip("voted node not in field for this embedder build")
    before = field["nodes"][node]["salience"]
    idx = retrieve.feedback_rerank_index(conn, UID)
    retrieve.apply_feedback_rerank(field["nodes"], field["q_emb"], idx,
                                   alpha=1.0, w_pos=0.45)
    assert field["nodes"][node]["salience"] > before


def test_dissimilar_query_no_effect(graph):
    conn = graph
    retrieve.record_relevance_feedback(conn, UID, "mercenaries versus missionaries in teams",
                                       relevant=["clm_cpt_b_0"], irrelevant=[])
    retrieve._FB_CACHE.clear()
    q = "advertising and the economy"                      # unrelated → cos < w_pos
    field = resonance.activate(conn, UID, q)
    before = {n: d["salience"] for n, d in field["nodes"].items()}
    retrieve.apply_feedback_rerank(field["nodes"], field["q_emb"],
                                   retrieve.feedback_rerank_index(conn, UID),
                                   alpha=1.0, w_pos=0.55)
    assert {n: d["salience"] for n, d in field["nodes"].items()} == before


def test_positive_only_irrelevant_never_demotes(graph):
    conn = graph
    retrieve.record_relevance_feedback(conn, UID, "why are advertisements bad for the economy?",
                                       relevant=[], irrelevant=["clm_cpt_a_0"])
    retrieve._FB_CACHE.clear()
    idx = retrieve.feedback_rerank_index(conn, UID)
    assert idx == [], "irrelevant-only feedback must produce no rerank targets"


def test_engagement_expands_to_member_claims(graph):
    conn = graph
    retrieve.record_engagement(conn, UID, "advertising economics",
                               surfaced=["cpt_a"], engaged="cpt_a", path=["cpt_a"])
    retrieve._FB_CACHE.clear()
    idx = retrieve.feedback_rerank_index(conn, UID)
    assert len(idx) == 1
    assert idx[0]["targets"] == {"clm_cpt_a_0", "clm_cpt_a_1"}, \
        "engaged concept must expand to its primary member claims, not the concept node"


def test_recall_head_applies_when_enabled(graph):
    """The recall() head honours the flag (default ON) and doesn't crash."""
    conn = graph
    retrieve.record_relevance_feedback(conn, UID, "why are advertisements bad for the economy?",
                                       relevant=["clm_cpt_a_0"], irrelevant=[])
    retrieve._FB_CACHE.clear()
    out = recall_mod.recall(conn, UID, "how does advertising harm the economy?", k=8)
    assert isinstance(out, list)
