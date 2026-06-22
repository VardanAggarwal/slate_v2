"""Fragment-backed retrieval (plan §2, P2.5) — the fragment layer's FIRST consumer.

Wires Write's fragment spine into RETRIEVE. Until now fragments were an orphan
branch: Write produced them, but `recall.py` (spreading activation) reads only the
v2 claims/concepts graph, so `z_echo` / the per-cluster calibration had no path to
SR@B and P3 (the calibration loop) had no objective. This module closes that loop —
it reads the working fragment layer and assembles a budgeted context from it.

100% local (knn seed + the assembly wrapper). No LLM, no network. The host LLM
reads the returned context and answers (PRD R9 — out of scope here).

Pipeline (PRD §Retrieve; the single-query subset of plan R-steps):
  seed   — store.fragment_candidates(query): nearest fragments, over-fetched.
  select — core/assembly.assemble(cand, Y=query, weights=relevance): the iterative
           residual-vs-assembly wrapper. Greedy max-marginal-residual: at each step
           add the candidate that adds the most NEW information per relevance, and
           STOP when the best marginal residual falls below the gain floor — the
           point that MAXIMISES the answer, not the one that fills B (PRD R5).
           Dedup is free: a near-duplicate of what's chosen has residual≈0 (R6).
  format — verbatim fragment spans grouped by source episode + provenance, cut to
           the char budget. Spans are verbatim raw, so fidelity is near-lossless
           (write-eval: 9.1/10 reconstruction) — the opposite end of the
           compression↔fidelity frontier from the v2 claims path.

Every function takes an explicit user_id and filters on it (AUTH.md §1/§3).
The STOP/budget POLICY is the scopeable calibration profile, fitted at
consolidation and pushed down — never baked here (mirrors recall/assembly).
"""
from __future__ import annotations

from core import assembly, store
from core.encode import get_embedder

# Seed breadth: how many nearest fragments to hand the assembly loop. Wider than
# the final assembly so the greedy selector has room to pick for COVERAGE, not just
# raw proximity. Selection/stop is the assembly wrapper's job, not this number's.
SEED_K = 40
# Retrieve's assembly cap: a generous item ceiling so the gain floor (sufficiency)
# or the char budget — not an arbitrary item count — is what stops assembly. The
# v2 recall used a hard k=8/12; here k is a budget, not the stop criterion.
MAX_ITEMS = 24
DEFAULT_CALIBRATION = {"gain_floor": assembly.GAIN_FLOOR, "max_items": MAX_ITEMS,
                       "per_cluster": {}}


def _embed_query(query: str):
    return get_embedder().encode([query], normalize_embeddings=True,
                                 show_progress_bar=False)[0]


def fragment_recall(conn, user_id: str, query: str, *, seed_k: int = SEED_K,
                    k: int | None = None, calibration: dict | None = None) -> list[dict]:
    """Rank + select fragments for a query via the assembly wrapper.

    Returns the CHOSEN fragments in assembly order (most-informative first), each
    enriched with its marginal `gain` (unique residual it added) and `share`
    (its fraction of total assembled information — the budget split, PRD R4).
    `stopped`-on-saturation vs filled-k is reflected by len(result) < the cap.
    Empty list when memory holds nothing for the query (PRD: return nothing, not
    padding)."""
    calibration = calibration or DEFAULT_CALIBRATION
    emb = _embed_query(query)
    cand = store.fragment_candidates(conn, user_id, emb, k=seed_k)
    if not cand:
        return []

    query_row = {"id": "__query__", "text": query, "embedding": emb}
    weights = [c["similarity"] for c in cand]  # relevance; assemble clips ≥0
    # k=None lets assemble read `max_items` from the calibration profile (the
    # item ceiling); the gain floor (saturation) or the char budget stops earlier.
    res = assembly.assemble(cand, seed=[query_row], weights=weights, k=k,
                            calibration=calibration)

    out = []
    for pos, idx in enumerate(res["chosen"]):
        c = cand[idx]
        out.append({**c, "rank": pos, "gain": res["gains"][pos],
                    "share": res["shares"][pos], "relevance": round(weights[idx], 4)})
    return out


def assemble_context(conn, user_id: str, topic: str, max_chars: int = 6000,
                     *, calibration: dict | None = None) -> str:
    """Fragment-backed analog of recall.assemble_context — the answerer entrypoint.

    Assembles verbatim fragment spans for `topic`, grouped by source episode with
    provenance, sized to the char budget. Most-informative-first so a budget cut
    drops the least-useful spans last (PRD: surface enough to answer within B)."""
    chosen = fragment_recall(conn, user_id, topic, calibration=calibration)
    if not chosen:
        return f"_Slate has nothing stored about “{topic}” yet._"

    # Group chosen spans by source episode, preserving assembly (relevance) order
    # across groups: a group's position is its best fragment's rank.
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for f in chosen:
        ep = f["episode_id"]
        if ep not in groups:
            groups[ep] = []
            order.append(ep)
        groups[ep].append(f)

    lines = [f"## Slate context: {topic}\n"]
    for ep in order:
        frags = groups[ep]
        head = frags[0]
        title = head["title"] or "untitled"
        when = (head["ts"] or "")[:10]
        lines.append(f"### {title} _({when})_")
        for f in frags:
            flag = " ⚠️ contested" if f.get("direction") == "contradict" else ""
            lines.append(f"- {f['text']}{flag}")
        lines.append("")

    out, total = [], 0
    for line in lines:  # most-relevant-first budget cut
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — narrow the query for more)_")
            break
        out.append(line)
    return "\n".join(out)


__all__ = ["fragment_recall", "assemble_context", "SEED_K", "MAX_ITEMS",
           "DEFAULT_CALIBRATION"]
