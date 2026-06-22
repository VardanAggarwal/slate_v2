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
