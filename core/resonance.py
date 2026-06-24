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

    for pi, pe in enumerate(p_embs):
        for h in store.knn_claims(conn, user_id, pe, k=int(C("res_seed_claims", SEED_CLAIMS))):
            if h["similarity"] > float(C("res_seed_floor", SEED_FLOOR)):
                deposit(h["claim_id"], pi, h["similarity"])
        for h in store.knn_concepts(conn, user_id, pe, k=int(C("res_seed_concepts", SEED_CONCEPTS))):
            if h["similarity"] > float(C("res_seed_floor", SEED_FLOOR)):
                deposit(h["id"], pi, h["similarity"])

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
    return {"nodes": nodes, "q_emb": q_emb, "probes": probes, "top_sim": top_sim}


# ── materialise bright nodes → verbatim fragments → assembly ────────────────────
def resonance_recall(conn, user_id: str, query: str, *,
                     calibration: dict | None = None) -> dict:
    """Returns {nodes, fragments, probes}: `nodes` = the scored activation field,
    `fragments` = the assembled verbatim spans (most-informative first)."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    field = activate(conn, user_id, query, calibration=calibration)
    nodes, q_emb = field["nodes"], field["q_emb"]
    if not nodes or field["top_sim"] < float(calibration.get("res_triage_min_rel", TRIAGE_MIN_REL)):
        return {"nodes": nodes, "fragments": [], "probes": field["probes"]}

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
    cand: dict[str, dict] = {}
    for node, sc in top:
        if node.startswith("cpt_"):
            eps = store.concept_episode_ids(conn, user_id, node)
        else:
            eps = store.claim_source_episodes(conn, user_id, node)
        if not eps:
            continue
        frs = store.fragments_for_episodes(conn, user_id, eps, q_emb)
        frs.sort(key=lambda f: -f["similarity"])
        for f in frs[:n_frags]:
            prev = cand.get(f["frag_id"])
            if prev is None or sc["salience"] > prev["_sal"]:
                cand[f["frag_id"]] = {**f, "_sal": sc["salience"]}
    if not cand:
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
                      calibration: dict | None = None) -> str:
    """Answerer entrypoint — verbatim spans from the brightest graph regions, grouped
    by source episode, sized to the char budget. Signals OFF (clean A/B; the loop-
    closing R8 write is a separate concern from the retrieval comparison)."""
    calibration = calibration or calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **DEFAULT_CALIBRATION}, user_id)
    res = resonance_recall(conn, user_id, topic, calibration=calibration)
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

    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for f in frags:
        ep = f["episode_id"]
        if ep not in groups:
            groups[ep] = []
            order.append(ep)
        groups[ep].append(f)

    if frags:
        lines.append("### Specifics")
    for ep in order:
        head = groups[ep][0]
        title = head.get("title") or "untitled"
        when = (head.get("ts") or "")[:10]
        lines.append(f"### {title} _({when})_")
        for f in groups[ep]:
            flag = " ⚠️ contested" if f.get("direction") == "contradict" else ""
            lines.append(f"- {f['text']}{flag}")
        lines.append("")

    out, total = [], 0
    for line in lines:
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — narrow the query for more)_")
            break
        out.append(line)
    return "\n".join(out)


__all__ = ["activate", "resonance_recall", "resonance_context", "DEFAULT_CALIBRATION"]
