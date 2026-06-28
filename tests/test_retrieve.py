"""Fragment-backed retrieval (core/retrieve.py, plan §2 P2.5).

Integration over the real local embedder + a temp DB: encode notes → refine into
fragments → retrieve. Asserts the plumbing and the R-step behaviours that DON'T
need an LLM (the host LLM read/synthesis is R9, out of scope — the SR@B harness
judges that). Deterministic offline: stance is "off" (conftest), embedder local.
"""
import numpy as np

from core import retrieve, store
from core.encode import encode
from core.write import refine_episode
from tests.conftest import UID, UID_B

# Three notes on clearly distinct topics, each multi-sentence so refine emits
# several fragments and assembly has something to select/stop over.
NOTE_AI = (
    "I believe the next paradigm shift in AI will come from memory. "
    "Memory creates a switching cost that becomes the moat for monetisation. "
    "Today's chat memories are shallow snippets, not a hippocampus-like store."
)
NOTE_FARMING = (
    "Sustainable agriculture has three aspects: social, economical, environmental. "
    "Sustainability comes from reducing the inherent risk in farming. "
    "Natural farming cuts the cost of cultivation while keeping yield comparable."
)
NOTE_RELIGION = (
    "I have grown from atheist to anti-theist about religion. "
    "Religion establishes an external agency with control over your life. "
    "Giving up that agency hides from reality and cripples growth, like a crutch."
)


def _seed(conn, uid=UID):
    eps = []
    for note in (NOTE_AI, NOTE_FARMING, NOTE_RELIGION):
        ep = encode(conn, uid, note, source="test")["episode_id"]
        refine_episode(conn, uid, ep)
        eps.append(ep)
    return eps


def test_fragment_recall_surfaces_the_relevant_topic(conn):
    """A query about AI-memory selects fragments from the AI note first, not the
    farming/religion notes — the assembly seed ranks by query relevance."""
    _seed(conn)
    chosen = retrieve.fragment_recall(conn, UID, "What is the next paradigm shift in AI?")
    assert chosen, "expected fragments for an on-corpus query"
    top = chosen[0]
    assert "AI" in top["text"] or "memory" in top["text"].lower()
    # the single most relevant span is more relevant than the median pick
    assert top["relevance"] >= max(c["relevance"] for c in chosen) - 1e-9


def test_fragment_recall_empty_without_fragments(conn):
    """No fragments stored → return nothing, never a crash (PRD: don't pad)."""
    assert retrieve.fragment_recall(conn, UID, "anything at all") == []


def test_assemble_context_groups_by_episode_with_provenance(conn):
    """Context is verbatim spans grouped under a source header (title/date)."""
    _seed(conn)
    ctx = retrieve.assemble_context(conn, UID, "natural farming and sustainability")
    assert ctx.startswith("## Slate context:")
    assert "###" in ctx                       # at least one episode-group header
    assert "sustainab" in ctx.lower() or "farming" in ctx.lower()


def test_assemble_context_respects_char_budget(conn):
    """A tiny budget truncates and says so — most-relevant-first, so the cut drops
    the least-useful spans last (sufficiency @ budget)."""
    _seed(conn)
    big = retrieve.assemble_context(conn, UID, "AI memory paradigm", max_chars=10_000)
    small = retrieve.assemble_context(conn, UID, "AI memory paradigm", max_chars=300)
    assert len(small) <= len(big)
    assert len(small) <= 360                  # budget + the short truncation marker
    assert "truncated" in small


def test_assemble_context_nothing_stored_message(conn):
    """Empty store → the explicit 'nothing stored' sentinel, not a padded guess."""
    msg = retrieve.assemble_context(conn, UID, "quantum gardening")
    assert "nothing stored" in msg.lower()


def test_chosen_fragments_are_distinct(conn):
    """R6 dedup: the assembly never picks the same span twice; max-marginal-residual
    won't re-add what's already represented (residual≈0 against the chosen set)."""
    _seed(conn)
    chosen = retrieve.fragment_recall(conn, UID, "AI memory paradigm")
    texts = [c["text"] for c in chosen]
    assert len(texts) == len(set(texts))
    ids = [c["frag_id"] for c in chosen]
    assert len(ids) == len(set(ids))


def test_retrieval_is_user_isolated(conn):
    """Graph/vector hops never cross users (AUTH.md §1/§3): UID_B sees nothing of
    UID's corpus."""
    _seed(conn, UID)
    assert retrieve.fragment_recall(conn, UID_B, "AI memory paradigm") == []


# ── R1 decompose ────────────────────────────────────────────────────────────────
def test_decompose_splits_a_multi_part_query(conn):
    """A two-topic query splits into ≥2 sub-queries at the planted boundary; a
    single-clause query stays whole."""
    q = ("What is the next paradigm shift in AI? "
         "How does natural farming reduce the cost of cultivation?")
    subs = retrieve.decompose_query(q)
    assert len(subs) >= 2
    assert retrieve.decompose_query("What is the AI memory moat?") == \
        ["What is the AI memory moat?"]


def test_decompose_seed_covers_both_parts(conn):
    """With decompose ON, a multi-topic query surfaces fragments from BOTH topics —
    the union seed covers each part, not just the dominant one."""
    _seed(conn)
    q = "What is the AI memory paradigm and how does natural farming cut cost?"
    chosen = retrieve.fragment_recall(conn, UID, q, decompose=True)
    blob = " ".join(c["text"].lower() for c in chosen)
    assert ("ai" in blob or "memory" in blob) and ("farm" in blob or "cultivation" in blob)


# ── R7 answerability triage ─────────────────────────────────────────────────────
def test_triage_returns_nothing_when_no_anchor(conn):
    """An off-corpus query (nothing clears the anchor floor) returns nothing rather
    than padding with nearest-but-irrelevant spans. Driven deterministically here by
    raising the triage floor above any real candidate's relevance."""
    _seed(conn)
    cal = {**retrieve.DEFAULT_CALIBRATION, "triage_min_rel": 0.99}
    assert retrieve.fragment_recall(conn, UID, "AI memory paradigm", calibration=cal) == []
    # the on-corpus query still answers under the default floor
    assert retrieve.fragment_recall(conn, UID, "AI memory paradigm")


# ── R3 borrow — topic → query-residual → match to off-topic span ──────────────────
def _bcand(fid, vec, cluster, q):
    """A synthetic candidate: normalised embedding + its cosine-to-query relevance."""
    v = vec / np.linalg.norm(vec)
    return {"frag_id": fid, "id": fid, "text": fid, "embedding": v,
            "cluster": cluster, "similarity": round(float(v @ q), 4)}


def _borrow_pool(q):
    e = np.eye(8)
    topic = [_bcand(f"a{i}", e[0] + 0.01 * i * e[1], "A", q) for i in range(4)]
    fit = _bcand("b", e[2], "B", q)      # off-topic, ALIGNS with the uncovered nuance
    miss = _bcand("c", e[1], "C", q)     # off-topic, unrelated to the nuance
    return sorted(topic + [fit, miss], key=lambda c: -c["similarity"])


def test_borrow_matches_query_residual_to_off_topic_span():
    """The borrowed span is the NON-topic candidate aligned with the query's
    uncovered-nuance direction (e2), not the unrelated off-topic one (e1)."""
    q = np.eye(8)[0] + 0.8 * np.eye(8)[2]
    q = q / np.linalg.norm(q)
    got = retrieve._borrow_nuance(q, _borrow_pool(q), min_rel=0.25)
    assert got is not None and got["frag_id"] == "b" and got["borrowed"]


def test_borrow_noop_when_topic_covers_query():
    """A query fully inside its topic has no residual to fill → nothing borrowed."""
    q = np.eye(8)[0]
    assert retrieve._borrow_nuance(q, _borrow_pool(q), min_rel=0.25) is None


def test_borrow_skips_when_no_off_topic_fits_the_gap(conn):
    """Borrow ON over the real corpus is additive and never duplicates a chosen
    span; it fires only when an off-topic span fits the query's residual."""
    _seed(conn)
    base = retrieve.fragment_recall(conn, UID, "AI memory paradigm")
    borrowed = retrieve.fragment_recall(conn, UID, "AI memory paradigm", borrow=True)
    ids = [c["frag_id"] for c in borrowed]
    assert len(ids) == len(set(ids))            # no dup
    assert len(borrowed) >= len(base)           # additive (≤1 extra)


# ── R8 retrieval signals ──────────────────────────────────────────────────────
def test_record_retrieval_signal_logs_fetched_and_dropped(conn):
    """R8 emits one RETRIEVAL_SIGNAL event partitioning the seed into fetched vs
    dropped — the C13 input — and never materializes a semantic row."""
    import json
    _seed(conn)
    emb = retrieve._embed_query("AI memory paradigm")
    seed = store.fragment_candidates(conn, UID, emb, k=retrieve.SEED_K)
    chosen = retrieve.fragment_recall(conn, UID, "AI memory paradigm")
    retrieve.record_retrieval_signal(conn, UID, "AI memory paradigm",
                                     fetched=chosen, seed=seed, truncated=False)
    rows = store.events_since(conn, UID, types=["RETRIEVAL_SIGNAL"])
    assert len(rows) == 1
    p = json.loads(rows[0]["payload_json"])
    assert set(p["fetched"]) == {c["frag_id"] for c in chosen}
    assert set(p["fetched"]).isdisjoint(p["dropped"])
    assert len(p["fetched"]) + len(p["dropped"]) == len(seed)


def test_assemble_context_emits_a_signal_by_default(conn):
    """The answerer entrypoint IS a retrieval, so it leaves exactly one signal
    (PRD: every retrieval feeds consolidation); off for speculative reads."""
    _seed(conn)
    retrieve.assemble_context(conn, UID, "AI memory paradigm")
    assert len(store.events_since(conn, UID, types=["RETRIEVAL_SIGNAL"])) == 1
    retrieve.assemble_context(conn, UID, "AI memory paradigm", signals=False)
    assert len(store.events_since(conn, UID, types=["RETRIEVAL_SIGNAL"])) == 1
