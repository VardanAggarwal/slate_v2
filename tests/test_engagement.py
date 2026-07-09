"""ENGAGEMENT event + recorder + felt-quality consumers.

Capture: `retrieve.record_engagement` logs which surfaced node a user drilled
into and the concept-normalised path they walked. Log-only — `apply_event`
ignores it, like RETRIEVAL_SIGNAL / RELEVANCE_FEEDBACK.

Consumers SHIPPED (felt-quality, not broad — broad was measured inert):
- ENGAGEMENT_SALIENCE: engaged node = implicit relevant vote in `_relevance_net`
  → C13 demotion exemption + C13b UNBACKGROUNDED promote.
- USAGE_REFRESH: usage recency (ENGAGEMENT + RETRIEVAL_SIGNAL fetches) feeds the
  step-7 decay clock so read-but-never-written concepts don't drift to dormant.

The broad-lift consumers (traversal edges, arc synthesis, walked-path
query-claims) stay unshipped — measured inert on Coverage@B; see
docs/broad-lift-traversal-findings.md and scratchpad/dead-lanes-full.patch.
"""
import json

import pytest

from core import consolidate, retrieve, store
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


# ── Consumer: engaged node = implicit relevant vote (ENGAGEMENT_SALIENCE) ──────
def test_engaged_node_unbackgrounds_member_claims(graph):
    """Drilling a concept after a recall is the 'needed' vote — C13b promotes its
    previously-demoted member claims back to foreground."""
    conn = graph
    store.set_claim_background(conn, UID, "clm_cpt_a_0", 1)
    retrieve.record_engagement(conn, UID, "q", surfaced=["cpt_a"],
                               engaged="cpt_a", path=["cpt_a"])
    with conn:
        consolidate._consume_relevance_feedback(conn, UID, "run_e", TS)
    assert store.get_claim(conn, UID, "clm_cpt_a_0")["background"] == 0


def test_engaged_claim_exempt_from_never_fetched_demotion(graph):
    """C13 would demote an exposed-never-fetched claim; an engagement vote on its
    concept exempts it (no demote/promote oscillation)."""
    conn = graph
    conn.execute("INSERT INTO fragments (id, user_id, episode_id, text) "
                 "VALUES ('frg_e', ?, 'ep_e', 'frag text')", (UID,))
    store.add_claim_support(conn, UID, "clm_cpt_a_0", "ep_e", "verbatim")
    for _ in range(consolidate.RETRIEVAL_EXPOSURE_MIN):
        store.append_event(conn, UID, "RETRIEVAL_SIGNAL",
                           {"query": "q", "fetched": [], "dropped": ["frg_e"],
                            "cut_for_budget": False})
    retrieve.record_engagement(conn, UID, "q", surfaced=["cpt_a"],
                               engaged="clm_cpt_a_0", path=["clm_cpt_a_0"])
    with conn:
        consolidate._consume_retrieval_signals(conn, UID, "run_e", TS)
    assert store.get_claim(conn, UID, "clm_cpt_a_0")["background"] == 0


def test_surfaced_but_not_engaged_is_not_demoted(graph):
    """Non-engagement must never count as an irrelevant vote (rare-but-correct)."""
    conn = graph
    retrieve.record_engagement(conn, UID, "q", surfaced=["cpt_a", "cpt_b"],
                               engaged="cpt_a", path=["cpt_a"])
    with conn:
        consolidate._consume_relevance_feedback(conn, UID, "run_e", TS)
    assert store.get_claim(conn, UID, "clm_cpt_b_0")["background"] == 0


# ── Consumer: usage recency feeds the decay clock (USAGE_REFRESH) ──────────────
def test_usage_refreshes_decay_clock(graph):
    """A concept recalled/walked recently stays active even when last_activity
    (write-side) is ancient; an unused twin with the same last_activity decays."""
    conn = graph
    old = "2026-01-01T00:00:00+00:00"
    conn.execute("UPDATE concepts SET last_activity = ? WHERE id IN ('cpt_a','cpt_b') "
                 "AND user_id = ?", (old, UID))
    retrieve.record_engagement(conn, UID, "q", surfaced=["cpt_a"],
                               engaged="cpt_a", path=["cpt_a"])   # ts = now
    with conn:
        consolidate._decay_strengthen(conn, UID, "run_d", [], TS)
    decayed = {json.loads(e["payload_json"])["concept_id"]
               for e in store.events_since(conn, UID, 0, types=["DECAYED"])}
    assert "cpt_a" not in decayed          # read recently → clock refreshed
    assert "cpt_b" in decayed              # same age, no usage → decays


def test_fetched_fragments_refresh_decay_clock(graph):
    """RETRIEVAL_SIGNAL fetches bridge frag → episode → claims → home concept."""
    conn = graph
    old = "2026-01-01T00:00:00+00:00"
    conn.execute("UPDATE concepts SET last_activity = ? WHERE id IN ('cpt_a','cpt_b') "
                 "AND user_id = ?", (old, UID))
    conn.execute("INSERT INTO fragments (id, user_id, episode_id, text) "
                 "VALUES ('frg_u', ?, 'ep_u', 'frag text')", (UID,))
    store.add_claim_support(conn, UID, "clm_cpt_a_0", "ep_u", "verbatim")
    store.append_event(conn, UID, "RETRIEVAL_SIGNAL",
                       {"query": "q", "fetched": ["frg_u"], "dropped": [],
                        "cut_for_budget": False})                 # ts = now
    with conn:
        consolidate._decay_strengthen(conn, UID, "run_d", [], TS)
    decayed = {json.loads(e["payload_json"])["concept_id"]
               for e in store.events_since(conn, UID, 0, types=["DECAYED"])}
    assert "cpt_a" not in decayed
    assert "cpt_b" in decayed
