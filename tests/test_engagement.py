"""ENGAGEMENT event + recorder (docs/broad-lift-traversal-findings.md, Phase 0).

The demand-signal capture layer: `retrieve.record_engagement` logs which surfaced
node a user drilled into and the concept-normalised path they walked. Log-only —
`apply_event` ignores it (read straight from the event log), like RETRIEVAL_SIGNAL
/ RELEVANCE_FEEDBACK. The broad-lift CONSUMERS of this signal (traversal edges,
arc synthesis, walked-path query-claims, engagement salience) were all measured
inert and are NOT shipped — see docs/broad-lift-traversal-findings.md and
scratchpad/dead-lanes-full.patch.
"""
import json

import pytest

from core import retrieve, store
from core.consolidate import apply_event
from core.encode import get_embedder
from tests.conftest import UID

TS = "2026-07-08T00:00:00+00:00"


def _mk_concept(conn, cid, label, claims):
    """Concept + primary member claims, embedded from text."""
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
    _mk_concept(conn, "cpt_a", "compromise", ["Compromise erodes the self over time."])
    _mk_concept(conn, "cpt_b", "identity", ["Identity must be actively preserved."])
    _mk_concept(conn, "cpt_c", "society", ["Society rewards conformity over individuality."])
    return conn


def test_record_engagement_normalises_path_to_concepts(graph):
    conn = graph
    # claim lifts to home concept; episode leaf drops; consecutive dups collapse
    retrieve.record_engagement(
        conn, UID, "how does compromise erode identity?",
        surfaced=["cpt_a", "clm_cpt_b_0"], engaged="clm_cpt_a_0",
        path=["clm_cpt_a_0", "cpt_a", "clm_cpt_b_0", "ep_zzz", "cpt_c"])
    evs = store.events_since(conn, UID, 0, types=["ENGAGEMENT"])
    assert len(evs) == 1
    p = json.loads(evs[0]["payload_json"])
    assert p["path"] == ["cpt_a", "cpt_b", "cpt_c"]
    assert p["engaged"] == "cpt_a"
    # log-only: applier ignores the event, nothing materialised
    apply_event(conn, UID, "ENGAGEMENT", p)
