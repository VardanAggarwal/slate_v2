"""Hierarchical retrieve — the predictor-native replacement for the budget-split
hybrid (`hybrid.hybrid_context`) AND the spreading-activation graph walk
(`recall.recall`).

First principles (PRD §Retrieve + the user's own "RAG Limitations and hierarchical
memory" note, 2026-03-12):

    a query = BACKGROUND (a concept / cluster) + NUANCE (within or across it).

    1. BACKGROUND — match the query to its nearest concept(s). Their centroids are
       the frame; their distilled `canonical` essence is the compressed background
       answer (cheap, covers the "tail" the old concept path owned).
    2. SUBTRACT  — seed the assembly's growing context Y with those concept
       centroids. A fragment that merely RESTATES the background then reads
       residual≈0 against Y and is dropped; only fragments that DEEPEN beyond the
       concept survive. This is the note's "define concepts, ignore them as
       background, focus on what is novel."
    3. NUANCE    — the assembly VOI loop (core/assembly via retrieve.fragment_recall)
       picks verbatim fragments by marginal residual×relevance and STOPS when
       saturated. Within-cluster vs across-cluster nuance falls out of the SAME
       residual: every fragment in the corpus competes in one assembly, so a
       cross-theme fragment that fills the query's uncovered direction wins on its
       own merit — no graph hop needed. THIS is what retires spreading activation:
       the fixed-decay SPREAD_*/MIN_ACTIVATION walk is replaced by one residual-VOI
       assembly with background suppression.

The old `concept_share` budget split is now EMERGENT, not a fixed knob: background
concepts take only the essence they need; fragments take whatever survives
subtraction. 100% local (concept knn + fragment knn + the assembly wrapper); the
host LLM reads the returned context and answers (PRD R9 — out of scope here).
"""
from __future__ import annotations

import numpy as np

from core import calibration as calib, predict, retrieve, store

# How many nearest concepts to consider as background, and the cosine a concept must
# clear to seed it (below this it's not really the query's theme — don't frame with
# noise). Both are calibration fit-targets (declared in retrieve.DEFAULT_CALIBRATION
# additions below), never SR@B-tuned in source.
SEED_CONCEPTS = 5
CONCEPT_MIN_REL = 0.18    # a concept must align at least this much to be background
MAX_BG_CONCEPTS = 3       # cap the frame so it doesn't crowd the nuance
# Fraction of the char budget the background frame may take before the rest goes to
# nuance fragments. A soft cap: the frame is compact essence, so this rarely binds,
# but it guarantees fragments always get the majority of B.
BG_BUDGET_CAP = 0.35
# Top member claims to enumerate under each background concept. The concept's
# canonical is the gist; its member claims are the BREADTH (the enumeration a
# "summarise / list all my X" query needs) — distilled, with provenance.
BG_CLAIMS_PER_CONCEPT = 4
# Relevance-gated STOP for the nuance assembly (R2/R7). The background frame already
# carries the breadth, so the fragment leg should add only RELEVANT depth, not pad
# the budget with novel-but-tangential spans. value_floor stops on residual×relevance
# (the same quantity the pick maximises), killing the over-injection that flat
# max-residual produces on broad queries. A calibration fit-target (P3 fitted 0.25 on
# the narrow gold); set as the hierarchical default so the nuance leg stays focused.
NUANCE_VALUE_FLOOR = 0.25
# Subtract the background from the nuance by seeding the assembly with the concept
# centroids? Helps DEPTH queries (don't repeat the frame) but hurts BREADTH ones
# (drops the frame's own instances). OFF by default — the frame carries breadth; the
# assembly's own dedup against CHOSEN fragments is enough to avoid intra-nuance repeats.
SUPPRESS_BACKGROUND = False
# Bridge-walk (replaces recall.py's SPREAD_*/MIN_ACTIVATION fixed-decay graph walk):
# from the seed concepts, hop one step over the concept/bridge graph and ADD a
# neighbour only when it is BOTH well-connected (edge weight) AND novel vs the
# background already selected (residual-VOI — not fixed decay). Its fragments enter
# the nuance seed even though the query knn under-ranked them (R0 fix for lexically-
# distant-but-topically-linked notes), and its essence enriches the frame.
#
# MEASURED 2026-06-24 (frozen-14 + broad gold): the walk is NET-NEGATIVE as a default.
# On narrow depth queries it injects cross-theme fragments that DILUTE the focused
# answer (frozen B2k 57%→50%, tail 40%→20%); on broad queries it reached some linked
# notes but flipped NO query (33%→33%), because the missing facts sit in notes the
# concept graph doesn't edge to (a consolidation-completeness limit). KEY FINDING: this
# refutes the hypothesis that retiring spreading activation costs ~7% on narrow — a
# faithful residual-VOI graph walk makes narrow WORSE, so hybrid's edge is its richer
# concept-path FORMATTING (claims grouped w/ provenance), not graph hops. OFF by
# default; kept calibration-gated for per-query-type routing experiments.
WALK_CONCEPTS = False
WALK_MAX_EXPAND = 3        # cap added neighbours per query (budget on the hop)
WALK_VOI_FLOOR = 0.30      # edge_score × novelty must clear this to add a neighbour
WALK_FRAGS_PER_CONCEPT = 3  # fragments pulled from each graph-reached concept

# Calibration keys this module reads (merged over retrieve.DEFAULT_CALIBRATION).
DEFAULT_CALIBRATION = {"seed_concepts": SEED_CONCEPTS,
                       "concept_min_rel": CONCEPT_MIN_REL,
                       "max_bg_concepts": MAX_BG_CONCEPTS,
                       "bg_budget_cap": BG_BUDGET_CAP,
                       "bg_claims_per_concept": BG_CLAIMS_PER_CONCEPT,
                       "suppress_background": SUPPRESS_BACKGROUND,
                       # nuance leg gets a relevance-gated stop by default
                       "value_floor": NUANCE_VALUE_FLOOR,
                       # keep only nuance candidates ≥ half as relevant as the best,
                       # so diversity operates within the on-topic set (kills the
                       # broad-query over-injection). Spread-relative; calibration-owned.
                       "rel_keep_frac": 0.5,
                       # bridge-walk knobs (residual-VOI graph hop; retires SPREAD_*)
                       "walk_concepts": WALK_CONCEPTS,
                       "walk_max_expand": WALK_MAX_EXPAND,
                       "walk_voi_floor": WALK_VOI_FLOOR,
                       "walk_frags_per_concept": WALK_FRAGS_PER_CONCEPT}


def _background_concepts(conn, user_id: str, q_emb, *, calibration: dict) -> list[dict]:
    """The query's nearest concept(s), above the relevance floor, with their centroid
    embeddings — the BACKGROUND frame. Empty when the query has no home theme (the
    concept layer can't frame it; nuance fragments carry it alone)."""
    k = int(calibration.get("seed_concepts", SEED_CONCEPTS))
    floor = float(calibration.get("concept_min_rel", CONCEPT_MIN_REL))
    cap = int(calibration.get("max_bg_concepts", MAX_BG_CONCEPTS))
    n_claims = int(calibration.get("bg_claims_per_concept", BG_CLAIMS_PER_CONCEPT))
    bg = []
    for h in store.knn_concepts(conn, user_id, q_emb, k=k):
        if h["similarity"] < floor:
            continue
        emb = store.concept_embedding(conn, user_id, h["id"])
        if emb is None:
            continue
        # The concept's strongest member claims = the BREADTH the frame carries
        # (the enumeration a "summarise/list" query needs), distilled with provenance.
        claims = [{"text": m["text"], "title": (m["support"][0]["title"]
                   if m.get("support") else None)}
                  for m in _concept_members(conn, user_id, h["id"], n_claims)]
        bg.append({"id": h["id"], "label": h["label"],
                   "canonical": h.get("canonical") or "", "state": h.get("state"),
                   "similarity": round(h["similarity"], 4),
                   "claims": claims, "embedding": np.asarray(emb, dtype=float)})
        if len(bg) >= cap:
            break
    return bg


def _concept_members(conn, user_id: str, concept_id: str, n: int) -> list[dict]:
    """Top-n member claims of a concept by strength, each with its first provenance
    (title) — the distilled breadth under a background concept. Direct SQL, no LLM."""
    rows = conn.execute(
        """SELECT cl.id, cl.text FROM concept_members cm
           JOIN claims cl ON cl.id = cm.claim_id
           WHERE cm.concept_id = ? AND cm.user_id = ?
           ORDER BY cl.strength DESC LIMIT ?""", (concept_id, user_id, n)).fetchall()
    out = []
    for r in rows:
        src = conn.execute(
            """SELECT e.title FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
               WHERE cs.claim_id = ? AND cs.user_id = ? ORDER BY e.ts LIMIT 1""",
            (r["id"], user_id)).fetchone()
        out.append({"text": r["text"],
                    "support": [{"title": src["title"]}] if src else []})
    return out


def _walk_expand(conn, user_id: str, seed_bg: list[dict], q_emb, *,
                 calibration: dict) -> list[dict]:
    """Residual-VOI bridge-walk — the predictor-native replacement for SPREAD_*.

    One hop from the seed concepts over the concept/bridge graph. A neighbour is
    ADDED only when `edge_score × novelty` clears the VOI floor, where novelty is the
    neighbour centroid's residual against the already-selected background — so a
    strongly-connected but REDUNDANT neighbour is skipped (the fixed-decay walk could
    not tell). Greedy, capped at `walk_max_expand`. Returns the added concept dicts
    (with embedding + the fragments pulled from each), for both frame and nuance."""
    if not calibration.get("walk_concepts", WALK_CONCEPTS) or not seed_bg:
        return []
    cap = int(calibration.get("walk_max_expand", WALK_MAX_EXPAND))
    floor = float(calibration.get("walk_voi_floor", WALK_VOI_FLOOR))
    n_frags = int(calibration.get("walk_frags_per_concept", WALK_FRAGS_PER_CONCEPT))
    seed_ids = [c["id"] for c in seed_bg]
    neighbours = store.related_concepts(conn, user_id, seed_ids)
    if not neighbours:
        return []

    selected = [np.asarray(c["embedding"], dtype=float) for c in seed_bg]
    added = []
    for nb in neighbours:
        if len(added) >= cap:
            break
        emb = store.concept_embedding(conn, user_id, nb["concept_id"])
        if emb is None:
            continue
        emb = np.asarray(emb, dtype=float)
        # novelty = residual of the neighbour centroid against the selected background
        novelty = float(predict.residuals_against(emb[None, :], np.vstack(selected))[0])
        if nb["edge_score"] * novelty < floor:
            continue
        c = store.get_concept(conn, user_id, nb["concept_id"])
        if not c:
            continue
        eps = store.concept_episode_ids(conn, user_id, nb["concept_id"])
        frs = store.fragments_for_episodes(conn, user_id, eps, q_emb)
        frs.sort(key=lambda f: -f["similarity"])
        added.append({"id": nb["concept_id"], "label": c["label"],
                      "canonical": c["canonical"] or "", "state": c["state"],
                      "similarity": None, "via": nb["relation"],
                      "bridge": nb["bridge"], "claims": [],
                      "embedding": emb, "frags": frs[:n_frags]})
        selected.append(emb)
    return added


def hierarchical_recall(conn, user_id: str, query: str, *,
                        calibration: dict | None = None, signals: bool = False,
                        run_id: str | None = None) -> dict:
    """Returns {background: [concept dicts], fragments: [chosen frag dicts]}.

    background = the concept frame (compressed tail); fragments = verbatim spans that
    DEEPEN beyond the frame (the assembly already subtracted the background by seeding
    Y with the concept centroids). Either may be empty."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    q_emb = retrieve._embed_query(query)
    bg = _background_concepts(conn, user_id, q_emb, calibration=calibration)
    # Bridge-walk: expand the background along the concept graph (residual-VOI). The
    # reached concepts enrich the frame AND inject their fragments into the nuance seed
    # — the lexically-distant notes the query knn missed (R0 fix).
    walked = _walk_expand(conn, user_id, bg, q_emb, calibration=calibration)
    extra_candidates = [f for w in walked for f in w["frags"]]
    # Suppress the background from the nuance (depth queries) only when enabled — by
    # default the frame carries the breadth and we don't subtract its own instances.
    bg_seed = None
    if calibration.get("suppress_background", SUPPRESS_BACKGROUND):
        bg_seed = [{"id": c["id"], "text": c["canonical"] or c["label"],
                    "embedding": c["embedding"]} for c in (bg + walked)]
    frags = retrieve.fragment_recall(conn, user_id, query, calibration=calibration,
                                     signals=signals, run_id=run_id,
                                     extra_seed=bg_seed,
                                     extra_candidates=extra_candidates or None)
    return {"background": bg, "walked": walked, "fragments": frags}


def hierarchical_context(conn, user_id: str, topic: str, max_chars: int = 6000, *,
                         calibration: dict | None = None, signals: bool = True,
                         run_id: str | None = None) -> str:
    """Answerer entrypoint — the hierarchical analog of recall/retrieve
    assemble_context. Background concept essences (the frame) first, then verbatim
    nuance fragments grouped by source episode, all sized to the char budget."""
    res = hierarchical_recall(conn, user_id, topic, calibration=calibration,
                              signals=signals, run_id=run_id)
    bg, frags = res["background"], res["fragments"]
    walked = res.get("walked", [])
    if not bg and not frags:
        return f"_Slate has nothing stored about “{topic}” yet._"

    cap = float((calibration or {}).get("bg_budget_cap", BG_BUDGET_CAP))
    bg_budget = int(max_chars * cap)

    lines = [f"## Slate context: {topic}\n"]
    # ── BACKGROUND frame (compressed; capped so nuance keeps the majority of B) ──
    if bg:
        frame, used = ["### Background"], 0
        stop = False
        for c in bg:
            head = (f"- **{c['label']}** — {c['canonical']}" if c["canonical"]
                    else f"- **{c['label']}**")
            block = [head]
            for m in c.get("claims", []):
                prov = f" _({m['title']})_" if m.get("title") else ""
                block.append(f"  - {m['text']}{prov}")
            for row in block:
                if used + len(row) > bg_budget and len(frame) > 1:
                    stop = True
                    break
                frame.append(row)
                used += len(row) + 1
            if stop:
                break
        # bridged concepts reached by the walk — the non-obvious cross-theme links
        for w in walked:
            if w.get("canonical"):
                tag = "🌉 " if w.get("bridge") else ""
                row = f"- {tag}**{w['label']}** — {w['canonical']}"
                if used + len(row) <= bg_budget:
                    frame.append(row)
                    used += len(row) + 1
        lines.extend(frame)
        lines.append("")

    # ── NUANCE (verbatim spans that survived background subtraction) ──
    if frags:
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for f in frags:
            ep = f["episode_id"]
            if ep not in groups:
                groups[ep] = []
                order.append(ep)
            groups[ep].append(f)
        lines.append("### Specifics")
        for ep in order:
            head = groups[ep][0]
            title = head.get("title") or "untitled"
            when = (head.get("ts") or "")[:10]
            lines.append(f"#### {title} _({when})_")
            for f in groups[ep]:
                flag = " ⚠️ contested" if f.get("direction") == "contradict" else ""
                borrowed = " ↗ borrowed" if f.get("borrowed") else ""
                lines.append(f"- {f['text']}{flag}{borrowed}")
            lines.append("")

    out, total = [], 0
    for line in lines:  # most-relevant-first budget cut
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — narrow the query for more)_")
            break
        out.append(line)
    return "\n".join(out)


__all__ = ["hierarchical_recall", "hierarchical_context", "DEFAULT_CALIBRATION",
           "SEED_CONCEPTS", "CONCEPT_MIN_REL", "MAX_BG_CONCEPTS"]
