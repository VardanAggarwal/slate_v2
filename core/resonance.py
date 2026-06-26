"""Resonance retrieve — navigation by activation over the consolidated graph.

The critical-note path (docs/retrieve-resonance-design.md). Retrieval-only: reads
the graph consolidation already builds (claims/concepts + membership + relations/
bridges) and changes nothing upstream. It is the concept path's (`recall.recall`)
successor, fixing its three structural gaps:

  1. CONFLUENCE — a query is decomposed into probes; each probe lights up the graph
     independently; activation SUMS across probes (recall.py takes max, discarding
     convergence) and takes the MAX within a probe (so a cycle can't inflate it). A
     node many probes reach outranks one a single probe reaches — "the parts that
     light up most because too many signals reach there."
  2. PE-GATED SPREAD — flow across an edge is gated by how DISTINCTIVE the target is
     given what is already lit (residual against the lit set), not a fixed SPREAD_*
     decay. This is the fragment path's max-marginal-residual applied along edges; it
     is also what gives real multi-hop deepening, and it de-noises ("exploration
     without de-noising returns only noise").
  3. VERBATIM MATERIALISE — navigation chooses WHERE (bright nodes); the answer is
     assembled from their backing verbatim fragments via the existing assembly
     wrapper, so the concept path's synthesis reach and the fragment path's fidelity
     come from ONE walk (vs hybrid's blind concatenation).

Three de-noisers separate a real query-hub from a generic corpus-hub (which lights
up for ANY query): fan-out normalisation (source side), the PE conductance gate
(path side), and a distinctiveness prior (node side; inverse-degree — cheap/robust,
upgrade to corpus-residual later).

100% local (knn seed + graph SQL + the residual primitive + the assembly wrapper).
The host LLM reads the returned context and answers (PRD R9 — out of scope here).
Every accessor takes an explicit user_id and filters on it (AUTH.md §1/§3).
"""
from __future__ import annotations

import math
import re

import numpy as np

from core import assembly, calibration as calib, predict, retrieve, store

# Clause delimiters — when sentence-decomposition yields one probe (a single-sentence
# query), split it into independent CLAUSE probes so confluence still fires: each
# clause lights up a different graph region, and convergence is the salience signal.
# Retrieval-side only — nothing is stored (no AI slop into memory).
_CLAUSE_RE = re.compile(r",|;|\s+\b(?:and|or|vs|versus|across|between|about|regarding)\b\s+",
                        flags=re.IGNORECASE)

# ── Seeding ───────────────────────────────────────────────────────────────────
SEED_CLAIMS = 12        # nearest claims per probe (multi-source injection sites)
SEED_CONCEPTS = 5       # nearest concepts per probe
SEED_FLOOR = 0.15       # a seed must clear this cosine to inject activation
TRIAGE_MIN_REL = 0.15   # whole query off-corpus → return nothing (PRD R7), like the others

# ── Spread ────────────────────────────────────────────────────────────────────
H_MAX = 2               # hop budget (the cap; VOI stop usually fires first)
FANOUT_EXP = 0.5        # divide a source's outflow by degree**this — a hub can't broadcast
MIN_ACTIVATION = 0.04   # drop activation below this (prune the frontier)
NEIGHBOR_CAP = 25       # max neighbours expanded per node (bounds a hub's fan-out)
LIT_FLOOR = 0.10        # a node counts as "lit" (joins the conductance set) above this
VOI_EPS = 0.05          # stop a hop when new salience < eps × salience-so-far

# ── Scoring ───────────────────────────────────────────────────────────────────
DIST_GAMMA = 0.5        # distinctiveness = 1/(1+ln(1+degree))**gamma  (denoiser #3)
MEMBERSHIP_W = 1.0      # claim↔concept edge weight (PE conductance does the gating)
BRIDGE_BOOST = 1.3      # a 'bridges' relation is the non-obvious cross-theme link

# ── Materialise ───────────────────────────────────────────────────────────────
MATERIALIZE_NODES = 12  # how many bright nodes to pull verbatim fragments from
FRAGS_PER_NODE = 3      # top fragments (by query cosine) per bright node
ASSEMBLE_GAIN_FLOOR = 0.20
ASSEMBLE_MAX_ITEMS = 24
# Distilled breadth frame: the brightest CONCEPT nodes contribute their canonical +
# top member claims (the enumeration a "summarise/list all my X" query needs) — the
# breadth verbatim spans alone can't carry. Concept SELECTION is confluence-salience
# (not knn rank, the hierarchical path's frame), so navigation picks the frame.
FRAME_CONCEPTS = 3      # brightest concept nodes to enumerate as the frame
FRAME_CLAIMS_PER = 4    # member claims per framed concept
FRAME_BUDGET_CAP = 0.35 # max share of B the frame may take (rest → verbatim nuance)
# Gap-2 (within-theme nuance reach): once navigation lands on a bright region, the
# query-cosine slice misses the answer-relevant-but-query-DISSIMILAR nuance (e.g. the
# "religion is a crutch" analogy — flagged is_novel_peak but cut by FRAGS_PER_NODE).
# Two fixes: (a) always include a bright node's is_novel_peak/is_centre fragments
# regardless of query cosine; (b) read the single brightest note(s) DEEPLY (more
# fragments) — "navigate, then read the note", the depth grep wins the tail on.
INCLUDE_PEAKS = True    # surface bright nodes' novel-peak / centre fragments
PEAKS_PER_NODE = 2      # cap on cosine-independent peak/centre picks per node
DEEP_TOP_N = 1          # read this many of the brightest notes deeply
DEEP_FRAGS = 8          # fragment cap for a deep-read note (vs FRAGS_PER_NODE)
BREADTH_FRAGS_PER_NODE = 1  # fragments per non-deep node (one each → coverage)
# Coverage layer: the breadth slice draws from graph-bright NODES, so a query-relevant
# note that isn't a bright node is missed (Mode-3 broad: e.g. the debugging note for an
# AI-memory query, the seed-savers note for a work query — both at fragment-rank ~20 by
# query cosine, just outside the navigated window). This layer unions the top distinct
# EPISODES by raw query cosine into the breadth pool, so reachable-but-not-bright notes
# get a slot. Bounded; breadth-tagged (draws the breadth budget, never depth). Cannot
# reach R0-wall notes (rank ~250+, vocabulary-disjoint) — those need consolidation.
# DEFAULT OFF (0) — VALIDATED net-negative (2026-06-24, self-judged ON vs OFF on broad
# gold). It does NOT beat Mode 3 and HURTS entity-aggregation queries: the top-cosine
# distinct episodes it adds are generic high-similarity notes that DISPLACE the sparse
# fact-notes (b_work 1/4→0/4). The genuinely-missing facts are R0 rank-250+ unreachable.
# Keep off; left as a flag for experiments. Mode 3 needs consolidation, not retrieval.
COVERAGE_NOTES = 0
# Budget-partition (Path 1): depth (deep-read+peaks of the top note) and breadth
# (coverage across many notes) each get a reserved slice of the specifics budget, so
# deep-read can't starve breadth (the broad-query regression) nor vice versa. Leftover
# from one slice flows to the other, so it self-balances WITHOUT classifying the query
# deep-vs-broad (which query geometry can't separate on this corpus — R0 + saturated
# concepts). A deep query has few real breadth notes → depth takes the leftover; a
# broad query fills breadth → coverage.
DEPTH_SHARE = 0.30      # share of specifics reserved for depth. 0.30 measured best:
                        # narrow 85.7/tail80 AND broad recovers to its 33% ceiling;
                        # 0.5 over-reserves depth and starves broad coverage (16.7%);
                        # 0.0 loses the narrow-tail deep-read win (78.6%).

DEFAULT_CALIBRATION = {
    "res_seed_claims": SEED_CLAIMS, "res_seed_concepts": SEED_CONCEPTS,
    "res_seed_floor": SEED_FLOOR, "res_triage_min_rel": TRIAGE_MIN_REL,
    "res_h_max": H_MAX, "res_fanout_exp": FANOUT_EXP,
    "res_min_activation": MIN_ACTIVATION, "res_neighbor_cap": NEIGHBOR_CAP,
    "res_lit_floor": LIT_FLOOR, "res_voi_eps": VOI_EPS,
    "res_dist_gamma": DIST_GAMMA, "res_materialize_nodes": MATERIALIZE_NODES,
    "res_frags_per_node": FRAGS_PER_NODE,
    "res_frame_concepts": FRAME_CONCEPTS, "res_frame_claims_per": FRAME_CLAIMS_PER,
    "res_frame_budget_cap": FRAME_BUDGET_CAP,
    "res_include_peaks": INCLUDE_PEAKS, "res_peaks_per_node": PEAKS_PER_NODE,
    "res_deep_top_n": DEEP_TOP_N, "res_deep_frags": DEEP_FRAGS,
    "res_breadth_frags_per_node": BREADTH_FRAGS_PER_NODE, "res_depth_share": DEPTH_SHARE,
    "res_coverage_notes": COVERAGE_NOTES,
    "gain_floor": ASSEMBLE_GAIN_FLOOR, "max_items": ASSEMBLE_MAX_ITEMS,
    "value_floor": None, "per_cluster": {},
    # ablation switches — flip to isolate each mechanism (see design doc test plan)
    "res_sum_probes": False,     # confluence: REFUTED on both narrow + paragraph gold
                                 # (sum buries the sharp node under generic-vocab grazing;
                                 # max wins 87.5 vs 62.5 on paragraph gold). True → sum.
    "res_pe_gate": True,         # False → fixed conductance (no PE de-noising)
    "res_distinctiveness": True, # False → drop the inverse-degree node prior
    "res_clause_probes": True,   # False → only sentence-level probes (confluence rarely fires)
}


def _probes(query: str, calibration: dict) -> list[str]:
    """Probe set for a query. Sentence decomposition first (reuses Write's scan); if
    that yields a single probe (a one-sentence query), fall back to CLAUSE splitting
    so a long single sentence still injects from multiple sites — the precondition for
    confluence. Each probe is an independent injection site; nothing is persisted."""
    subs = retrieve.decompose_query(query, calibration=calibration) or [query]
    if len(subs) >= 2 or not calibration.get("res_clause_probes", True):
        return subs
    parts = [p.strip(" .?!") for p in _CLAUSE_RE.split(query)]
    parts = [p for p in parts if len(p) >= 4 and " " in p or len(p) >= 8]
    return parts if len(parts) >= 2 else subs


# ── graph adjacency + embeddings (cached per call) ──────────────────────────────
class _Graph:
    """On-demand adjacency + node embeddings over the consolidated store, cached for
    one retrieval. Nodes are claim ids ('clm_…') and concept ids ('cpt_…')."""

    def __init__(self, conn, user_id: str):
        self.conn, self.user_id = conn, user_id
        self._emb: dict[str, np.ndarray | None] = {}
        self._adj: dict[str, list[tuple[str, float]]] = {}
        self._deg: dict[str, int] = {}

    def emb(self, node: str) -> np.ndarray | None:
        if node not in self._emb:
            if node.startswith("cpt_"):
                v = store.concept_embedding(self.conn, self.user_id, node)
            else:
                v = store.claim_embedding(self.conn, self.user_id, node)
            self._emb[node] = None if v is None else np.asarray(v, dtype=float)
        return self._emb[node]

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        """[(other_node, edge_weight)] over membership + relation edges. Degree is the
        full count (recorded before the NEIGHBOR_CAP truncation) for distinctiveness."""
        if node in self._adj:
            return self._adj[node]
        conn, uid = self.conn, self.user_id
        edges: dict[str, float] = {}
        if node.startswith("clm_"):
            for r in conn.execute(
                    "SELECT concept_id FROM concept_members WHERE claim_id=? AND user_id=?",
                    (node, uid)):
                edges[r["concept_id"]] = max(edges.get(r["concept_id"], 0), MEMBERSHIP_W)
        elif node.startswith("cpt_"):
            for r in conn.execute(
                    "SELECT claim_id FROM concept_members WHERE concept_id=? AND user_id=?",
                    (node, uid)):
                edges[r["claim_id"]] = max(edges.get(r["claim_id"], 0), MEMBERSHIP_W)
        for r in conn.execute(
                """SELECT from_id, to_id, relation, weight FROM relations
                   WHERE user_id=? AND (from_id=? OR to_id=?)""", (uid, node, node)):
            other = r["to_id"] if r["from_id"] == node else r["from_id"]
            w = min(1.0, r["weight"] or 1.0) * (BRIDGE_BOOST if r["relation"] == "bridges" else 1.0)
            edges[other] = max(edges.get(other, 0), w)
        self._deg[node] = len(edges)
        adj = sorted(edges.items(), key=lambda kv: -kv[1])[:NEIGHBOR_CAP]
        self._adj[node] = adj
        return adj

    def degree(self, node: str) -> int:
        if node not in self._deg:
            self.neighbours(node)
        return self._deg.get(node, 0)


# ── core: build the activation field ────────────────────────────────────────────
def activate(conn, user_id: str, query: str, *, calibration: dict | None = None) -> dict:
    """Decompose → multi-source seed → PE-gated spread. Returns
    {nodes: {node: {strength, confluence, salience}}, q_emb, probes, top_sim}."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    C = calibration.get
    sum_probes = bool(C("res_sum_probes", True))
    pe_gate = bool(C("res_pe_gate", True))
    use_dist = bool(C("res_distinctiveness", True))
    fanout_exp = float(C("res_fanout_exp", FANOUT_EXP))
    min_act = float(C("res_min_activation", MIN_ACTIVATION))
    lit_floor = float(C("res_lit_floor", LIT_FLOOR))
    h_max = int(C("res_h_max", H_MAX))
    voi_eps = float(C("res_voi_eps", VOI_EPS))
    dist_gamma = float(C("res_dist_gamma", DIST_GAMMA))

    g = _Graph(conn, user_id)
    q_emb = retrieve._embed_query(query)
    probes = _probes(query, calibration)
    p_embs = predict._default_embed(probes)  # (P, d), one encode call

    # act[node][probe] = activation. Per-probe so SUM across probes = confluence; we
    # take MAX within a probe so a cycle/multi-path can't double-count one probe.
    act: dict[str, dict[int, float]] = {}

    def deposit(node: str, pi: int, val: float):
        if val < min_act:
            return False
        cur = act.setdefault(node, {})
        if val > cur.get(pi, 0.0):
            cur[pi] = val
            return True
        return False

    seeded: set[str] = set()  # nodes injected directly (vs reached by spread) — the via signal
    for pi, pe in enumerate(p_embs):
        for h in store.knn_claims(conn, user_id, pe, k=int(C("res_seed_claims", SEED_CLAIMS))):
            if h["similarity"] > float(C("res_seed_floor", SEED_FLOOR)):
                deposit(h["claim_id"], pi, h["similarity"])
                seeded.add(h["claim_id"])
        for h in store.knn_concepts(conn, user_id, pe, k=int(C("res_seed_concepts", SEED_CONCEPTS))):
            if h["similarity"] > float(C("res_seed_floor", SEED_FLOOR)):
                deposit(h["id"], pi, h["similarity"])
                seeded.add(h["id"])

    def strength(node: str) -> float:
        vals = act[node].values()
        return sum(vals) if sum_probes else max(vals)

    top_sim = max((max(v.values()) for v in act.values()), default=0.0)

    frontier = set(act.keys())
    sal_so_far = sum(strength(n) for n in act)
    for _ in range(h_max):
        if not frontier:
            break
        # freeze the lit set (conductance context) at the start of the hop
        lit = [n for n in act if strength(n) >= lit_floor]
        lit_mat = None
        if pe_gate and len(lit) >= 2:
            embs = [g.emb(n) for n in lit]
            embs = [e for e in embs if e is not None]
            if len(embs) >= 2:
                lit_mat = np.vstack(embs)

        next_frontier: set[str] = set()
        new_sal = 0.0
        for u in frontier:
            u_act = dict(act[u])  # snapshot (don't read this hop's own deposits)
            adj = g.neighbours(u)
            if not adj:
                continue
            damp = 1.0 / (g.degree(u) ** fanout_exp) if fanout_exp else 1.0
            for v, w in adj:
                v_emb = g.emb(v)
                if v_emb is None:
                    continue
                if lit_mat is not None:
                    cond = float(predict.residuals_against(v_emb[None, :], lit_mat)[0])
                else:
                    cond = 1.0  # warmup / gate off: edge weight only
                factor = w * cond * damp
                if factor <= 0:
                    continue
                for pi, val in u_act.items():
                    dv = val * factor
                    if deposit(v, pi, dv):
                        next_frontier.add(v)
                        new_sal += dv
        frontier = next_frontier
        if new_sal < voi_eps * max(sal_so_far, 1e-9):  # VOI: hop added ~nothing distinctive
            break
        sal_so_far += new_sal

    # score: salience = strength · (1 + ln confluence) · distinctiveness
    nodes: dict[str, dict] = {}
    for node, pv in act.items():
        st = sum(pv.values()) if sum_probes else max(pv.values())
        conf = len(pv)
        dist = 1.0
        if use_dist:
            dist = 1.0 / (1.0 + math.log1p(g.degree(node))) ** dist_gamma
        sal = st * (1.0 + math.log(conf)) * dist
        nodes[node] = {"strength": round(st, 4), "confluence": conf,
                       "distinctiveness": round(dist, 4), "salience": round(sal, 6)}
    return {"nodes": nodes, "q_emb": q_emb, "probes": probes,
            "top_sim": top_sim, "seeded": seeded}


# ── materialise bright nodes → verbatim fragments → assembly ────────────────────
def resonance_recall(conn, user_id: str, query: str, *,
                     calibration: dict | None = None, signals: bool = False,
                     run_id: str | None = None) -> dict:
    """Returns {nodes, fragments, frame, probes}: `nodes` = the scored activation
    field, `fragments` = the assembled verbatim spans (most-informative first).

    `signals` (R8): log fetched/dropped fragments for consolidation C13 (PRD §Retrieve
    — every retrieval leaves signals). Off for speculative/eval reads, ON in prod."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    field = activate(conn, user_id, query, calibration=calibration)
    nodes, q_emb = field["nodes"], field["q_emb"]
    if not nodes or field["top_sim"] < float(calibration.get("res_triage_min_rel", TRIAGE_MIN_REL)):
        if signals:
            retrieve.record_retrieval_signal(conn, user_id, query, fetched=[],
                                             seed=[], truncated=False, run_id=run_id)
        return {"nodes": nodes, "fragments": [], "frame": [], "probes": field["probes"]}

    ranked = sorted(nodes.items(), key=lambda kv: -kv[1]["salience"])
    top = ranked[: int(calibration.get("res_materialize_nodes", MATERIALIZE_NODES))]
    n_frags = int(calibration.get("res_frags_per_node", FRAGS_PER_NODE))

    # Distilled breadth frame — the brightest CONCEPT nodes, enumerated (canonical +
    # top member claims). Selection is by confluence-salience, so navigation, not knn,
    # picks what to frame. Carries the breadth verbatim spans can't (broad queries).
    frame = []
    fc = int(calibration.get("res_frame_concepts", FRAME_CONCEPTS))
    fcp = int(calibration.get("res_frame_claims_per", FRAME_CLAIMS_PER))
    for node, sc in ranked:
        if len(frame) >= fc:
            break
        if not node.startswith("cpt_"):
            continue
        c = store.get_concept(conn, user_id, node)
        if not c:
            continue
        frame.append({"id": node, "label": c["label"],
                      "canonical": c["canonical"] or "", "salience": sc["salience"],
                      "claims": _concept_members(conn, user_id, node, fcp)})

    # bright node → its source episodes → verbatim fragments; tag each fragment with
    # the MAX salience of any bright node that reached it (its budget weight).
    include_peaks = bool(calibration.get("res_include_peaks", INCLUDE_PEAKS))
    peaks_per = int(calibration.get("res_peaks_per_node", PEAKS_PER_NODE))
    deep_top_n = int(calibration.get("res_deep_top_n", DEEP_TOP_N))
    deep_frags = int(calibration.get("res_deep_frags", DEEP_FRAGS))
    breadth_frags = int(calibration.get("res_breadth_frags_per_node", BREADTH_FRAGS_PER_NODE))
    cand: dict[str, dict] = {}
    for rank_i, (node, sc) in enumerate(top):
        if node.startswith("cpt_"):
            eps = store.concept_episode_ids(conn, user_id, node)
        else:
            eps = store.claim_source_episodes(conn, user_id, node)
        if not eps:
            continue
        frs = store.fragments_for_episodes(conn, user_id, eps, q_emb)
        frs.sort(key=lambda f: -f["similarity"])
        # DEPTH nodes (the brightest deep_top_n) get a deep read; BREADTH nodes (the
        # rest) contribute few fragments each → coverage across many notes. The slice a
        # fragment lands in (`_depth`) decides which reserved budget it draws from.
        is_depth = rank_i < deep_top_n
        cap = deep_frags if is_depth else breadth_frags
        picks = frs[:cap]
        # plus the node's DISTINCTIVE fragments (novel-peak / centre) the cosine slice
        # dropped — the answer-relevant-but-query-dissimilar nuance (Gap-2 fix).
        if include_peaks:
            picks = picks + [f for f in frs[cap:]
                             if f.get("is_novel_peak") or f.get("is_centre")][:peaks_per]
        for f in picks:
            prev = cand.get(f["frag_id"])
            if prev is None or sc["salience"] > prev["_sal"]:
                cand[f["frag_id"]] = {**f, "_sal": sc["salience"], "_depth": is_depth}
    # Coverage layer: union the top distinct EPISODES by raw query cosine into the
    # breadth pool, so query-relevant notes that aren't graph-bright nodes still get a
    # slot (the reachable Mode-3 fact-notes at fragment-rank ~20). Breadth-tagged.
    cov_notes = int(calibration.get("res_coverage_notes", COVERAGE_NOTES))
    if cov_notes:
        seen_ep = {c["episode_id"] for c in cand.values()}
        best_per_ep: dict[str, dict] = {}
        for c in store.fragment_candidates(conn, user_id, q_emb, k=cov_notes * 6):
            ep = c["episode_id"]
            if ep in seen_ep:
                continue
            if ep not in best_per_ep or c["similarity"] > best_per_ep[ep]["similarity"]:
                best_per_ep[ep] = c
        for c in sorted(best_per_ep.values(), key=lambda x: -x["similarity"])[:cov_notes]:
            if c["frag_id"] not in cand:
                cand[c["frag_id"]] = {**c, "_sal": float(c["similarity"]), "_depth": False}

    if not cand:
        if signals:
            retrieve.record_retrieval_signal(conn, user_id, query, fetched=[],
                                             seed=[], truncated=False, run_id=run_id)
        return {"nodes": nodes, "fragments": [], "frame": frame, "probes": field["probes"]}

    candidates = list(cand.values())
    smax = max(c["_sal"] for c in candidates) or 1.0
    weights = [c["_sal"] / smax for c in candidates]  # parent-salience, normalised
    query_row = {"id": "__query__", "text": query, "embedding": q_emb}
    res = assembly.assemble(candidates, seed=[query_row], weights=weights,
                            calibration=calibration)
    out = []
    for pos, idx in enumerate(res["chosen"]):
        c = candidates[idx]
        out.append({**c, "rank": pos, "gain": res["gains"][pos],
                    "salience": round(c["_sal"], 4)})
    if signals:
        # R8: chosen = fetched, the rest of the candidate pool = dropped (C13 maps
        # frag → episode → claims; demotes salient-but-never-fetched).
        retrieve.record_retrieval_signal(conn, user_id, query, fetched=out,
                                         seed=candidates,
                                         truncated=not res.get("stopped", False),
                                         run_id=run_id)
    return {"nodes": nodes, "fragments": out, "frame": frame,
            "probes": field["probes"]}


def _concept_members(conn, user_id: str, concept_id: str, n: int) -> list[dict]:
    """Top-n member claims of a concept by strength, with first provenance title."""
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
        out.append({"text": r["text"], "title": src["title"] if src else None})
    return out


def resonance_context(conn, user_id: str, topic: str, max_chars: int = 6000, *,
                      calibration: dict | None = None, signals: bool = False,
                      run_id: str | None = None) -> str:
    """Answerer entrypoint — verbatim spans from the brightest graph regions, grouped
    by source episode, sized to the char budget. `signals` (R8) default OFF for eval/
    speculative reads; the prod MCP flow passes signals=True to close the C13 loop."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    res = resonance_recall(conn, user_id, topic, calibration=calibration,
                           signals=signals, run_id=run_id)
    frags, frame = res["fragments"], res.get("frame", [])
    if not frags and not frame:
        return f"_Slate has nothing stored about “{topic}” yet._"

    lines = [f"## Slate context: {topic}\n"]

    # ── BACKGROUND frame (distilled breadth from the brightest concepts; capped) ──
    if frame:
        cap = int(max_chars * float(calibration.get("res_frame_budget_cap", FRAME_BUDGET_CAP)))
        frame_lines, used, stop = ["### Background"], 0, False
        for c in frame:
            block = [f"- **{c['label']}** — {c['canonical']}" if c["canonical"]
                     else f"- **{c['label']}**"]
            for m in c["claims"]:
                prov = f" _({m['title']})_" if m.get("title") else ""
                block.append(f"  - {m['text']}{prov}")
            for row in block:
                if used + len(row) > cap and len(frame_lines) > 1:
                    stop = True
                    break
                frame_lines.append(row)
                used += len(row) + 1
            if stop:
                break
        lines.extend(frame_lines)
        lines.append("")

    # ── SPECIFICS: depth/breadth budget partition (Path 1) ──
    # The frame already consumed some budget; the rest splits between a DEPTH slice
    # (deep-read of the brightest note) and a BREADTH slice (coverage across notes).
    # Reserve breadth FIRST, then give depth everything left — so a deep query (few
    # real breadth notes) hands its leftover to depth, while a broad query keeps its
    # breadth coverage. No deep-vs-broad classification needed.
    def _emit(frag_list, budget):
        """Render frags grouped by source episode within `budget` chars. Returns
        (lines, chars_used). Episode order follows assembly rank (most-informative)."""
        groups: dict[str, list[dict]] = {}
        order: list[str] = []
        for f in frag_list:
            ep = f["episode_id"]
            if ep not in groups:
                groups[ep] = []
                order.append(ep)
            groups[ep].append(f)
        rendered, used = [], 0
        for ep in order:
            head = groups[ep][0]
            title = head.get("title") or "untitled"
            when = (head.get("ts") or "")[:10]
            block = [f"### {title} _({when})_"]
            for f in groups[ep]:
                flag = " ⚠️ contested" if f.get("direction") == "contradict" else ""
                block.append(f"- {f['text']}{flag}")
            block.append("")
            blen = sum(len(x) + 1 for x in block)
            if used + blen > budget and rendered:
                break
            rendered += block
            used += blen
        return rendered, used

    frame_chars = sum(len(l) + 1 for l in lines)
    specifics_budget = max(0, max_chars - frame_chars)
    depth_share = float(calibration.get("res_depth_share", DEPTH_SHARE))
    breadth_budget = int(specifics_budget * (1.0 - depth_share))

    depth_f = [f for f in frags if f.get("_depth")]
    breadth_f = [f for f in frags if not f.get("_depth")]
    breadth_lines, breadth_used = _emit(breadth_f, breadth_budget)
    depth_lines, _ = _emit(depth_f, specifics_budget - breadth_used)  # depth gets the rest

    if depth_lines or breadth_lines:
        lines.append("### Specifics")
        lines += depth_lines + breadth_lines  # focused depth first, then coverage

    out, total = [], 0
    for line in lines:
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — narrow the query for more)_")
            break
        out.append(line)
    return "\n".join(out)


__all__ = ["activate", "resonance_recall", "resonance_context", "DEFAULT_CALIBRATION"]
