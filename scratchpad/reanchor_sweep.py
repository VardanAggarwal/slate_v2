"""Verify the WIRED consolidate._reanchor_concepts + sweep its knobs (λ, J).

Each setting: reset vec_concepts to medoid, run the production re-anchor, eval.
"""
import sys
import numpy as np
from core import store, consolidate, calibration as calib, retrieve, resonance
from eval.coverage import coverage_eval, _load_gold, HERE
from eval_both import reach_eval, GOLD, TAU

DB = "/tmp/slate_work.db"
U = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
conn = store.connect(DB)
cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, U)
golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}
cids = [r["id"] for r in conn.execute("SELECT id FROM concepts WHERE user_id=?", (U,))]


def reset_medoid():
    with conn:
        for cid in cids:
            store.recompute_concept_embedding(conn, U, cid)  # medoid (per-event default)


def run_eval(tag):
    cs = store.all_concepts(conn, U)
    A = np.vstack([store.concept_embedding(conn, U, c["id"]) for c in cs])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    pc = (A @ A.T)[np.triu_indices(A.shape[0], 1)].mean()
    cells = []
    for gname, gold in golds.items():
        cov = coverage_eval(gold, conn, U, resonance.resonance_context, TAU)
        rch = reach_eval(gold, conn, U, cal)
        cells.append(f"{gname[:4]} C={cov['sr_at_b']:.0%} R={rch['reach_at_b']:.0%}")
    print(f"{tag:22} pc={pc:.3f}  " + "  ".join(cells))


grid = [(lam, j) for lam in (0.5, 1.0, 2.0) for j in (3, 5, 8)]
print("== sweep (λ, J) via production _reanchor_concepts ==")
for lam, j in grid:
    reset_medoid()
    with conn:
        consolidate._reanchor_concepts(conn, U, lam=lam, j=j)
    run_eval(f"λ={lam} J={j}")
