"""Frag+concept HYBRID retrieval (plan §2 / §6 P4 — "the path to the R-gate").

The SR@B matrix (docs/retrieve-workflow-eval.md) showed the two single paths are
COMPLEMENTARY, not redundant:

  - `core/retrieve.py` (fragments)  wins the FACTUAL/specific queries — verbatim
    spans carry the detail — but loses the conceptual TAIL (20% vs claims' 67%).
  - `core/recall.py` (claims/concepts) wins the hard tail — the concept graph
    synthesises across notes — but misses concrete detail the spans nail.

Neither dominates, so a replacement of one by the other can't close the R-gate. This
module gives EACH path a slice of the same budget B: concepts for the tail, fragments
for specificity. It is a thin orchestrator — no new geometry — over the two existing
answerer entrypoints, so each path keeps its own provenance/format/dedup.

100% local (both sub-paths are knn + the assembly wrapper). The host LLM reads the
returned context and answers (PRD R9 — out of scope here).

POLICY — `concept_share` is the budget split (concepts vs fragments). It is a C12
calibration target, NOT pre-tuned: the value below is a sane default (fragments are
the proven specificity winner on the common queries, so they keep the majority; the
tail gets a guaranteed minority slice). The fitted value comes from the frozen SR@B
sweep and is pushed down — never baked as the answer here (mirrors `value_floor`).
"""
from __future__ import annotations

from core import recall, retrieve

# Default budget split. Fragments keep the majority (they win the common factual
# queries); concepts get a guaranteed slice so the tail gets synthesis. FIT TARGET —
# the frozen SR@B sweep replaces this; do not hand-tune it past a sane default.
CONCEPT_SHARE = 0.4

_HEADER_PREFIX = "## Slate context:"
_EMPTY_MARK = "nothing stored"


def _is_empty(body: str) -> bool:
    """A sub-path returned its 'nothing stored' sentinel (PRD: don't pad)."""
    return _EMPTY_MARK in body


def _strip_header(body: str) -> str:
    """Drop a sub-context's own `## Slate context: …` top line (+ the blank after it)
    so the hybrid can wrap both bodies under a single header."""
    lines = body.split("\n")
    if lines and lines[0].startswith(_HEADER_PREFIX):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def hybrid_context(conn, user_id: str, topic: str, max_chars: int = 6000, *,
                   concept_share: float | None = None,
                   calibration: dict | None = None) -> str:
    """Budget-split blend of the concept path (tail) and the fragment path
    (specificity). Each sub-path gets its share of `max_chars`; if one is empty its
    share is handed to the other (no wasted budget); if both are empty, the sentinel.

    `concept_share` (0..1) is the fraction of the budget reserved for concepts/claims;
    read from the calibration profile or the module default — never decided here."""
    if concept_share is None:
        concept_share = (calibration or {}).get("concept_share", CONCEPT_SHARE)
    concept_share = min(max(concept_share, 0.0), 1.0)

    concept_budget = int(max_chars * concept_share)
    frag_budget = max_chars - concept_budget

    # Concept path first (knn, emits no R8 signals). If it comes back empty the fragment
    # path takes the WHOLE budget — so we size the fragment assembly ONCE up front rather
    # than running it at frag_budget and re-running it at full budget. The old re-run not
    # only wasted the heavier assembly pass, it emitted R8 signals TWICE (signals default
    # ON), inflating C13's exposure counts. Fragments now run exactly once.
    concept = recall.assemble_context(conn, user_id, topic, max_chars=concept_budget)
    c_empty = _is_empty(concept)

    frag = retrieve.assemble_context(conn, user_id, topic,
                                     max_chars=(max_chars if c_empty else frag_budget),
                                     calibration=calibration)
    f_empty = _is_empty(frag)

    if c_empty and f_empty:
        return f"_Slate has nothing stored about “{topic}” yet._"
    if c_empty:
        return frag                       # already assembled at the full budget
    if f_empty:
        # Concept path carries it alone; re-run at the full budget (cheap knn, no signals).
        return recall.assemble_context(conn, user_id, topic, max_chars=max_chars)

    # Both present: synthesis first (frames + protects the tail under a final cut),
    # then verbatim source spans for the concrete detail.
    body = (f"## Slate context: {topic}\n\n"
            f"### Synthesis — concepts & claims\n"
            f"{_strip_header(concept)}\n\n"
            f"### Source spans — verbatim\n"
            f"{_strip_header(frag)}")

    if len(body) <= max_chars:
        return body
    out, total = [], 0
    for line in body.split("\n"):  # final guard: keep total ≤ B, synthesis-first
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — narrow the query for more)_")
            break
        out.append(line)
    return "\n".join(out)


__all__ = ["hybrid_context", "CONCEPT_SHARE"]
