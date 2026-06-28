"""Combined Coverage@B + Reach@B for the resonance path, over all three gold sets.

Coverage@B (eval.coverage.coverage_eval): query passes iff every key_fact is covered
  (cosine of fact vs assembled context >= tau). Strict sr_at_b + hard-tail.

Reach@B (here): SELECTION granularity, graded. For each gold key_fact, find its nearest
  claim node (knn, claim-level — independent of concept structure, so it's a stable
  A/B mapping). The fact is REACHED if that claim is in the navigation's selected node
  set: the top MATERIALIZE_NODES by salience (claims directly + member claims of any
  selected concept). Reach@B = mean over queries of (fraction of facts reached).
  This is what consolidation moves: merging/splitting concepts changes which nodes
  light up and get selected, independent of verbatim text bias.

Usage: python3 scratchpad/eval_both.py <db> [user]
"""
import sys
import statistics

import numpy as np

from core import store, calibration as calib, retrieve, resonance
from eval.coverage import coverage_eval, _load_gold, _embed, HERE

U_DEFAULT = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
TAU = 0.54
GOLD = {"narrow": "gold.jsonl", "broad": "gold_broad.jsonl", "paragraph": "gold_paragraph.jsonl"}


def _selected_claims(conn, user_id, query, calibration, top_n):
    """The claim set navigation surfaces: top-n nodes by salience → claims + members."""
    field = resonance.activate(conn, user_id, query, calibration=calibration)
    ranked = sorted(field["nodes"].items(), key=lambda kv: -kv[1]["salience"])[:top_n]
    claims = set()
    for node, _ in ranked:
        if node.startswith("clm_"):
            claims.add(node)
        elif node.startswith("cpt_"):
            claims.update(store.concept_member_ids(conn, user_id, node))
    return claims


def reach_eval(gold, conn, user_id, calibration):
    top_n = int(calibration.get("res_materialize_nodes", resonance.MATERIALIZE_NODES))
    per = []
    for g in gold:
        selected = _selected_claims(conn, user_id, g["query"], calibration, top_n)
        facts = g["key_facts"]
        FV = _embed(facts)
        reached = []
        for i, f in enumerate(facts):
            hits = store.knn_claims(conn, user_id, FV[i], k=1)
            cid = hits[0]["claim_id"] if hits else None
            reached.append(bool(cid and cid in selected))
        frac = statistics.fmean(reached) if reached else 0.0
        per.append({"id": g.get("id", g["query"][:30]), "hard": bool(g.get("hard")),
                    "reach": frac, "reached": reached})
    graded = statistics.fmean([p["reach"] for p in per]) if per else 0.0
    hard = [p["reach"] for p in per if p["hard"]]
    return {"reach_at_b": round(graded, 4),
            "reach_tail": round(statistics.fmean(hard), 4) if hard else None,
            "per_query": per}


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slate_work.db"
    user = sys.argv[2] if len(sys.argv) > 2 else U_DEFAULT
    conn = store.connect(db)
    calibration = calib.merged(
        conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, user)

    print(f"db={db} user={user} tau={TAU}\n")
    print(f"{'gold':10} {'Coverage@B':>20} {'Reach@B (graded)':>22}")
    rows = {}
    for gname, gfile in GOLD.items():
        gold = _load_gold(HERE / gfile)
        cov = coverage_eval(gold, conn, user, resonance.resonance_context, TAU)
        rch = reach_eval(gold, conn, user, calibration)
        rows[gname] = (cov, rch)
        ct = f"{cov['sr_at_b']:.1%}" + (f"/{cov['sr_tail']:.0%}" if cov["sr_tail"] is not None else "")
        rt = f"{rch['reach_at_b']:.1%}" + (f"/{rch['reach_tail']:.0%}" if rch["reach_tail"] is not None else "")
        print(f"{gname:10} {ct:>20} {rt:>22}")
    return rows


if __name__ == "__main__":
    main()
