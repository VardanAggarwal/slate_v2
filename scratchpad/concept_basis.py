"""Test concept-representative selection strategies, eval Coverage@B + Reach@B each.

The concept's stored vector is its REPRESENTATIVE. We only change WHICH point
represents it (always a real member → on-manifold, encoder-frame, reversible):

  centroid     normalized mean of members (original; collapses toward global centre)
  medoid       member with max summed self-cosine ≈ LOWEST residual vs own children.
               Squishy only — central to self, blind to neighbours.
  pushy        member with HIGHEST residual vs neighbouring concepts' members.
               Pushy only — most distinct from neighbours, blind to self-centrality.
  contrastive  argmax [ residual_vs_neighbours − residual_vs_own ].
               Squishy AND pushy: central to self, distinct from neighbours (a margin).

Neighbours are fixed by the MEDOID geometry (the J nearest concepts' member vectors)
so the neighbour set doesn't shift with the strategy under test.

Usage: python3 scratchpad/concept_basis.py [db] [user]
"""
import sys
import numpy as np

from core import store, predict, calibration as calib, retrieve, resonance
from eval.coverage import coverage_eval, _load_gold, HERE
from eval_both import reach_eval, GOLD, TAU

J_NEIGHBORS = 5      # nearest concepts whose members form the push-against pool
NBR_POOL_CAP = 60    # cap neighbour pool size (speed)


def _medoid(V):
    return V[0] if V.shape[0] == 1 else V[int(np.argmax((V @ V.T).sum(1)))]


def _select(strategy, own, nbr):
    """Return the chosen representative vector for one concept.
    own  = (m,d) member vectors;  nbr = (p,d) neighbouring-concept member vectors."""
    if own.shape[0] == 1:
        return own[0]
    if strategy == "centroid":
        m = own.mean(0); n = np.linalg.norm(m); return m / n if n > 0 else m
    if strategy == "medoid":
        return _medoid(own)
    # per-candidate residuals
    r_own = np.array([predict.residual_against(own[i], np.delete(own, i, axis=0))
                      for i in range(own.shape[0])])           # low = central (squishy)
    if nbr.shape[0] == 0:
        return _medoid(own) if strategy == "pushy" else own[int(np.argmin(r_own))]
    r_nbr = np.array([predict.residual_against(own[i], nbr)
                      for i in range(own.shape[0])])           # high = distinct (pushy)
    if strategy == "pushy":
        return own[int(np.argmax(r_nbr))]
    if strategy == "contrastive":
        return own[int(np.argmax(r_nbr - r_own))]
    raise ValueError(strategy)


def recompute(conn, user_id, strategy):
    cs = store.all_concepts(conn, user_id)
    mem, med = {}, {}
    for c in cs:
        rows = [r for r in (store.claim_embedding(conn, user_id, m)
                            for m in store.concept_member_ids(conn, user_id, c["id"]))
                if r is not None]
        if rows:
            V = np.vstack(rows); mem[c["id"]] = V; med[c["id"]] = _medoid(V)
    ids = list(med)
    M = np.vstack([med[i] for i in ids])
    M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    S = M @ M.T
    with conn:
        for k, cid in enumerate(ids):
            own = mem[cid]
            order = np.argsort(-S[k])
            nbr_ids = [ids[j] for j in order if ids[j] != cid][:J_NEIGHBORS]
            nbr = np.vstack([mem[j] for j in nbr_ids]) if nbr_ids else np.zeros((0, own.shape[1]))
            if nbr.shape[0] > NBR_POOL_CAP:
                nbr = nbr[:NBR_POOL_CAP]
            v = _select(strategy, own, nbr)
            n = np.linalg.norm(v); v = v / n if n > 0 else v
            conn.execute("DELETE FROM vec_concepts WHERE concept_id=?", (cid,))
            conn.execute("INSERT INTO vec_concepts (user_id, concept_id, embedding) VALUES (?,?,?)",
                         (user_id, cid, store.serialize_float32([float(x) for x in v])))


def main():
    db = sys.argv[1] if len(sys.argv) > 1 else "/tmp/slate_work.db"
    user = sys.argv[2] if len(sys.argv) > 2 else "usr_01KTXAYR20J4R6F7PT3DP10W3W"
    conn = store.connect(db)
    cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, user)
    golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}

    for strat in ["centroid", "medoid", "pushy", "contrastive"]:
        recompute(conn, user, strat)
        # separation diagnostic
        cs = store.all_concepts(conn, user)
        A = np.vstack([store.concept_embedding(conn, user, c["id"]) for c in cs
                       if store.concept_embedding(conn, user, c["id"]) is not None])
        A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
        off = (A @ A.T)[np.triu_indices(A.shape[0], 1)]
        print(f"\n=== {strat.upper()}  (mean pair-cos {off.mean():.3f}) ===")
        print(f"{'gold':10} {'Coverage@B':>16} {'Reach@B':>16}")
        for gname, gold in golds.items():
            cov = coverage_eval(gold, conn, user, resonance.resonance_context, TAU)
            rch = reach_eval(gold, conn, user, cal)
            ct = f"{cov['sr_at_b']:.1%}" + (f"/{cov['sr_tail']:.0%}" if cov["sr_tail"] is not None else "")
            rt = f"{rch['reach_at_b']:.1%}" + (f"/{rch['reach_tail']:.0%}" if rch["reach_tail"] is not None else "")
            print(f"{gname:10} {ct:>16} {rt:>16}")


if __name__ == "__main__":
    main()
