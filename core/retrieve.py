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

import numpy as np

from core import assembly, calibration as calib, predict, scan, store
from core.encode import get_embedder, split_sentences

# Seed breadth: how many nearest fragments to hand the assembly loop. Wider than
# the final assembly so the greedy selector has room to pick for COVERAGE, not just
# raw proximity. Selection/stop is the assembly wrapper's job, not this number's.
SEED_K = 40
# Retrieve's assembly cap: a generous item ceiling so the gain floor (sufficiency)
# or the char budget — not an arbitrary item count — is what stops assembly. The
# v2 recall used a hard k=8/12; here k is a budget, not the stop criterion.
MAX_ITEMS = 24
# R7 answerability triage: if NOTHING in memory anchors the query (best candidate's
# relevance below this), return nothing rather than padding with the nearest-but-
# irrelevant spans (PRD §Retrieve "return nothing rather than padding"). Default
# sits between the corpus's ambient cosine (~0.10) and answer cosine (~0.66) seen in
# the R0 spike — conservative; a fit-target like the rest, never SR@B-tuned here.
TRIAGE_MIN_REL = 0.15
# R3 borrow: a borrowed cross-theme span must ALIGN with the query's uncovered-nuance
# direction at least this much (cosine) — a genuine fit, not noise. Fit-target.
BORROW_MIN_REL = 0.25
# `value_floor` is the R2/R7 relevance-aware STOP — the P3 calibration target that
# kills the observed over-injection (assembly pads to budget, hurting SR@B@B/2; see
# docs/retrieve-workflow-eval.md). LEFT None (disabled) here on purpose: it must be
# FITTED against frozen SR@B by C12 and pushed down, never pre-tuned in source.
# `decompose`/`borrow` (R1/R3) are now wired into the pipeline as profile flags: ON
# by default so a multi-part query covers every part (R1) and a query whose own theme
# can't fully cover it borrows one aligned cross-theme span (R3). Both stay
# calibration-controllable — a fitted profile can flip either off — but they are no
# longer structurally unreachable from the hybrid/MCP path.
DEFAULT_CALIBRATION = {"gain_floor": assembly.GAIN_FLOOR, "max_items": MAX_ITEMS,
                       "value_floor": assembly.VALUE_FLOOR,
                       "triage_min_rel": TRIAGE_MIN_REL,
                       "borrow_min_rel": BORROW_MIN_REL,
                       "decompose": True, "borrow": True,
                       # `concept_share` is the hybrid budget split (concepts vs fragments;
                       # mirrors `hybrid.CONCEPT_SHARE`). Declared here so the single fitted
                       # profile can carry/push it like `value_floor` — it's a C12 fit-target,
                       # not pre-tuned. hybrid falls back to its own const if a caller omits it.
                       "concept_share": 0.4,
                       "per_cluster": {}}


def _embed_query(query: str):
    return get_embedder().encode([query], normalize_embeddings=True,
                                 show_progress_bar=False)[0]


def decompose_query(query: str, *, calibration: dict | None = None) -> list[str]:
    """R1 — split a multi-part query into independent sub-queries via the sequential
    scan wrapper (topic boundaries BETWEEN the query's sentences). A single-clause
    query returns `[query]` unchanged; a compound query yields one sub-query per
    planted part, so R4 can later allocate budget across them. Pure/offline."""
    # min_chars=1: queries are short, so don't drop a brief clause the way the
    # write-side sentence floor (SENT_MIN_CHARS=40) would.
    sents = split_sentences(query, min_chars=1)
    if len(sents) <= 1:
        return [query.strip()] if query.strip() else []
    E = get_embedder().encode(sents, normalize_embeddings=True, show_progress_bar=False)
    segs = scan.segment(np.asarray(E, dtype=float), calibration=calibration, scope="query")
    # scan MERGES same-topic sentences; at query scale it often can't measure a
    # boundary (too few points) and returns one segment. The seed is unioned and
    # assembly runs once over it, so OVER-splitting is harmless while UNDER-splitting
    # loses a part — so when scan can't separate multiple sentences, fall back to one
    # sub-query per sentence (the finest independent units).
    if len(segs) <= 1:
        return [s.strip() for s in sents if s.strip()]
    subs = [" ".join(sents[i] for i in seg).strip() for seg in segs]
    return [s for s in subs if s]


def _borrow_nuance(query_emb, pool: list[dict], *,
                   min_rel: float = BORROW_MIN_REL) -> dict | None:
    """R3 — borrow a transferable nuance from ANOTHER theme. Mechanism (PRD line 168,
    "a low-residual *fit* across a theme boundary"):

      1. TOPIC — the query's on-topic theme = the cluster of its most-relevant
         candidate (`pool` is similarity-descending).
      2. RESIDUAL — reconstruct the query from its OWN topic's spans; the leftover
         `residual_direction` is the part of the query the topic can't cover — the
         uncovered nuance, as a direction.
      3. MATCH — among NON-topic candidates, the one whose embedding best aligns with
         that residual direction is the transferable nuance. Borrow it only if the
         alignment clears `min_rel` (a real fit, not noise).

    Returns the borrowed candidate (flagged `borrowed`, `gain`=the fit) or None.
    No-op when there is no theme boundary (topic cluster missing) or the topic
    already covers the query (no residual to fill)."""
    if not pool:
        return None
    topic = pool[0].get("cluster")
    if topic is None:                       # no theme to anchor → nothing to cross
        return None
    topic_embs = [np.asarray(c["embedding"], dtype=float)
                  for c in pool if c.get("cluster") == topic]
    non_topic = [c for c in pool if c.get("cluster") != topic]
    if not topic_embs or not non_topic:
        return None

    resid = predict.residual_direction(np.asarray(query_emb, dtype=float),
                                       np.vstack(topic_embs))
    rn = float(np.linalg.norm(resid))
    if rn < 1e-6:                           # topic fully reconstructs the query
        return None
    resid = resid / rn

    fits = [(c, float(np.asarray(c["embedding"], dtype=float) @ resid)) for c in non_topic]
    best, fit = max(fits, key=lambda t: t[1])
    if fit < min_rel:                       # no off-topic span fits the uncovered gap
        return None
    return {**best, "gain": round(fit, 4), "value": round(fit, 4), "share": 0.0,
            "relevance": round(best["similarity"], 4), "borrowed": True}


def record_retrieval_signal(conn, user_id: str, query: str, *, fetched: list[dict],
                            seed: list[dict], truncated: bool,
                            run_id: str | None = None) -> None:
    """R8 — append a RETRIEVAL_SIGNAL event (fetched / dropped / cut-for-budget) for
    consolidation C13 to consume (promote dropped-but-needed, demote
    salient-but-never-fetched). Append-only — NOT materialized into the semantic
    store, so consolidate.apply_event ignores it; it is read straight from the log."""
    fetched_ids = {f["frag_id"] for f in fetched}
    store.append_event(conn, user_id, "RETRIEVAL_SIGNAL", {
        "query": query,
        "fetched": [f["frag_id"] for f in fetched],
        "dropped": [c["frag_id"] for c in seed if c["frag_id"] not in fetched_ids],
        "cut_for_budget": bool(truncated),
    }, run_id=run_id)


def _home_concept(conn, user_id: str, claim_id: str) -> str | None:
    """A claim's HOME concept (primary membership) — mirrors consolidate._claim_concept,
    inlined here so the recorder has no consolidate dependency."""
    row = conn.execute(
        "SELECT concept_id FROM concept_members WHERE claim_id = ? AND user_id = ? "
        "AND kind = 'primary' LIMIT 1", (claim_id, user_id)).fetchone()
    return row["concept_id"] if row else None


def _concept_walk(conn, user_id: str, ids: list[str]) -> list[str]:
    """Normalise a traversal to CONCEPTS — the only pivot the walk analysis found:
    concepts pass through, claims lift to their home concept, episodes drop.
    Consecutive duplicates collapse (drilling two claims of one concept is one stop)."""
    out: list[str] = []
    for nid in ids or []:
        if nid.startswith("cpt_"):
            c = nid
        elif nid.startswith("clm_"):
            c = _home_concept(conn, user_id, nid)
        else:                                   # ep_/frag_ leaves carry no pivot
            c = None
        if c and (not out or out[-1] != c):
            out.append(c)
    return out


def record_engagement(conn, user_id: str, query: str, *, surfaced: list[str],
                      engaged: str | None, path: list[str],
                      spawned_write: bool = False, run_id: str | None = None) -> None:
    """Traversal signal — the user's own drill-down after a recall (the demand-side
    concept walk consolidation was missing; docs/broad-lift-traversal-plan.md §0.1).
    `surfaced` = what recall offered, `engaged` = the node drilled first, `path` = the
    walk order. Path/engaged are normalised to CONCEPTS (claim → home concept, leaves
    dropped) before append. Append-only, log-only — NOT materialized into the semantic
    store (apply_event ignores it); consolidation reads it straight from the log,
    mirroring RETRIEVAL_SIGNAL / RELEVANCE_FEEDBACK."""
    walk = _concept_walk(conn, user_id, list(path or []))
    eng = _concept_walk(conn, user_id, [engaged] if engaged else [])
    store.append_event(conn, user_id, "ENGAGEMENT", {
        "query": query,
        "surfaced": list(surfaced or []),
        "engaged": eng[0] if eng else None,
        "path": walk,
        "spawned_write": bool(spawned_write),
    }, run_id=run_id)


def record_relevance_feedback(conn, user_id: str, query: str, *,
                              relevant: list[str], irrelevant: list[str],
                              run_id: str | None = None) -> None:
    """Explicit relevance feedback from the host after it USED recalled context:
    `relevant` / `irrelevant` are claim/concept/episode ids the assistant judged
    helped (or didn't) answer `query`. Append-only RELEVANCE_FEEDBACK event,
    consumed by consolidation C13b — the explicit 'needed' signal usage logging
    (RETRIEVAL_SIGNAL / C13) cannot provide, so promotion was deferred without it.
    Not materialized into the semantic store directly; read straight from the log."""
    store.append_event(conn, user_id, "RELEVANCE_FEEDBACK", {
        "query": query,
        "relevant": list(relevant or []),
        "irrelevant": list(irrelevant or []),
    }, run_id=run_id)


def _seed_pool(conn, user_id: str, query: str, *, seed_k: int,
               decompose: bool, calibration: dict | None):
    """Seed candidate fragments for `query`. With R1 decompose ON, union the seeds
    of each independent sub-query (dedup by frag_id, keep the max similarity) so a
    multi-part query covers every part; OFF, a single knn seed. Returns
    (candidates, query_embedding)."""
    emb = _embed_query(query)
    if not decompose:
        return store.fragment_candidates(conn, user_id, emb, k=seed_k), emb
    subs = decompose_query(query, calibration=calibration)
    if len(subs) <= 1:
        return store.fragment_candidates(conn, user_id, emb, k=seed_k), emb
    # Embed every sub-query in ONE encode call — under the prod HF API each
    # _embed_query is a network round-trip, so a per-sub loop would be N of them.
    sub_embs = get_embedder().encode(subs, normalize_embeddings=True,
                                     show_progress_bar=False)
    merged: dict[str, dict] = {}
    for se in sub_embs:
        for c in store.fragment_candidates(conn, user_id, se, k=seed_k):
            cur = merged.get(c["frag_id"])
            if cur is None or c["similarity"] > cur["similarity"]:
                merged[c["frag_id"]] = c
    return list(merged.values()), emb


def fragment_recall(conn, user_id: str, query: str, *, seed_k: int = SEED_K,
                    k: int | None = None, calibration: dict | None = None,
                    decompose: bool | None = None, borrow: bool | None = None,
                    signals: bool = False, run_id: str | None = None,
                    extra_seed: list[dict] | None = None,
                    extra_candidates: list[dict] | None = None) -> list[dict]:
    """Rank + select fragments for a query via the assembly wrapper.

    Returns the CHOSEN fragments in assembly order (most-informative first), each
    enriched with its marginal `gain` (unique residual it added) and `share`
    (its fraction of total assembled information — the budget split, PRD R4).
    `stopped`-on-saturation vs filled-k is reflected by len(result) < the cap.
    Empty list when memory holds nothing for the query, OR when nothing in memory
    anchors it (R7 triage — PRD: return nothing, never pad).

    `decompose` (R1) seeds from independent sub-queries; `borrow` (R3) appends one
    cross-theme nuance; `signals` (R8) logs fetched/dropped for consolidation.
    `decompose`/`borrow` default to the calibration profile (ON by default — see
    DEFAULT_CALIBRATION); pass an explicit bool to override the profile for one call.

    `extra_seed` (hierarchical retrieve): rows prepended to the assembly's initial
    context Y alongside the query — e.g. the BACKGROUND concept centroids. A fragment
    that merely restates what the background already covers then reads residual≈0 and
    is suppressed, so only fragments that DEEPEN beyond the background survive (the
    'define concepts, ignore as background, focus on the novel' rule). Triage/borrow
    still measure relevance against the bare query, not the augmented seed."""
    # C12: a caller override wins; otherwise load the user's fitted profile over
    # the in-code defaults (just the defaults until a fit is pushed).
    calibration = calibration or calib.merged(conn, DEFAULT_CALIBRATION, user_id)
    # Resolve R1/R3 toggles: an explicit bool arg wins; else the profile flag; else off.
    if decompose is None:
        decompose = bool(calibration.get("decompose", False))
    if borrow is None:
        borrow = bool(calibration.get("borrow", False))
    cand, emb = _seed_pool(conn, user_id, query, seed_k=seed_k,
                           decompose=decompose, calibration=calibration)
    if not cand:
        return []

    # R7 answerability triage: if the best candidate doesn't clear the anchor floor,
    # the query is off-corpus — return nothing instead of padding (PRD §Retrieve).
    triage = calibration.get("triage_min_rel", TRIAGE_MIN_REL)
    top_sim = max(c["similarity"] for c in cand)
    if top_sim < triage:
        if signals:
            record_retrieval_signal(conn, user_id, query, fetched=[], seed=cand,
                                    truncated=False, run_id=run_id)
        return []

    # Spread-relative relevance floor (R2 per-candidate "drop-if-echo"): keep only
    # candidates at least `rel_keep_frac` as relevant as the best, so the VOI loop's
    # max-marginal-RESIDUAL operates WITHIN the on-topic set instead of padding with
    # novel-but-irrelevant spans. On broad queries relevance is uniformly low (the R0
    # symmetric-encoder limit), so without this the diversity term pulls in garbage.
    # Off (None) → no filtering, the legacy whole-pool behaviour. Calibration-owned.
    rel_keep = calibration.get("rel_keep_frac")
    if rel_keep:
        floor = max(triage, top_sim * float(rel_keep))
        kept = [c for c in cand if c["similarity"] >= floor]
        if kept:
            cand = kept

    # Graph-reached candidates (the graph-walk): union AFTER the relevance filter so
    # they survive it — they were chosen by GRAPH connectivity, not query cosine, and
    # are exactly the lexically-distant notes the query knn under-ranks (R0 fix). Dedup
    # by frag_id, keep the higher similarity.
    if extra_candidates:
        have = {c["frag_id"] for c in cand}
        for xc in extra_candidates:
            if xc["frag_id"] not in have:
                cand.append({**xc, "via_graph": True})
                have.add(xc["frag_id"])

    query_row = {"id": "__query__", "text": query, "embedding": emb}
    weights = [c["similarity"] for c in cand]  # relevance; assemble clips ≥0
    # Seed Y = the query, plus any BACKGROUND rows (concept centroids) so fragments
    # already covered by the background are suppressed (hierarchical retrieve).
    seed_rows = [query_row] + list(extra_seed or [])
    # k=None lets assemble read `max_items` from the calibration profile (the
    # item ceiling); the gain floor (saturation) or the char budget stops earlier.
    res = assembly.assemble(cand, seed=seed_rows, weights=weights, k=k,
                            calibration=calibration)

    out = []
    for pos, idx in enumerate(res["chosen"]):
        c = cand[idx]
        out.append({**c, "rank": pos, "gain": res["gains"][pos],
                    "value": res["values"][pos], "share": res["shares"][pos],
                    "relevance": round(weights[idx], 4)})
    if borrow:
        min_rel = calibration.get("borrow_min_rel", BORROW_MIN_REL)
        chosen_ids = {c["frag_id"] for c in out}
        nuance = _borrow_nuance(emb, cand, min_rel=min_rel)
        if nuance and nuance["frag_id"] not in chosen_ids:
            out.append({**nuance, "rank": len(out)})
    if signals:
        # cut_for_budget ≈ assembly did NOT stop on saturation (was item/budget-bound)
        record_retrieval_signal(conn, user_id, query, fetched=out, seed=cand,
                                truncated=not res.get("stopped", False), run_id=run_id)
    return out


def assemble_context(conn, user_id: str, topic: str, max_chars: int = 6000,
                     *, calibration: dict | None = None, signals: bool = True,
                     run_id: str | None = None) -> str:
    """Fragment-backed analog of recall.assemble_context — the answerer entrypoint.

    Assembles verbatim fragment spans for `topic`, grouped by source episode with
    provenance, sized to the char budget. Most-informative-first so a budget cut
    drops the least-useful spans last (PRD: surface enough to answer within B).

    `signals` default ON: this IS a retrieval, and per the PRD every retrieval leaves
    fetched/dropped signals for consolidation (R8). Off for speculative/preview reads."""
    chosen = fragment_recall(conn, user_id, topic, calibration=calibration,
                             signals=signals, run_id=run_id)
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


__all__ = ["fragment_recall", "assemble_context", "decompose_query",
           "record_retrieval_signal", "record_relevance_feedback",
           "record_engagement", "SEED_K",
           "MAX_ITEMS", "TRIAGE_MIN_REL", "BORROW_MIN_REL", "DEFAULT_CALIBRATION"]
