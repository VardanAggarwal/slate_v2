"""Stages 1–3: let the contrastive margin drive MEMBERSHIP (not just representation),
purely geometrically (no LLM), and measure the retrieval effect.

We reuse the spine: predict.measure() scores each claim's residual z against its
NEAREST cluster's own cohesion (density-aware, compute_baselines μ/σ) and reports the
dominant nearby cluster. That dominant-cluster vote IS the squish signal, denoised over
STAT_K neighbours. Reassign each claim to it (label propagation), re-anchor concepts
contrastively, repeat (the co-evolving EM loop). Mint = a claim NOVEL against every
cluster (high z, no anchor) → its own seed. Merge = concept pairs with low cross-residual.

Boundary: this tests whether contrastive GEOMETRY improves membership for retrieval.
The LLM still owns meaning (labels, contradiction-vs-paraphrase) in production; that
layer doesn't change retrieval reach, so it's deferred.

Eval after each pass: Coverage@B + Reach@B vs the current (LLM-built) membership.
"""
import sys
import numpy as np

from core import store, consolidate, predict, calibration as calib, retrieve, resonance
from eval.coverage import coverage_eval, _load_gold, HERE
from eval_both import reach_eval, GOLD, TAU

DB = "/tmp/slate_work.db"
U = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
MINT_Z = 2.5          # claim this far above its nearest cluster's spread → mint a seed
MIN_CONCEPT = 2       # drop concepts that fall below this after reassignment

conn = store.connect(DB)
cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, U)
golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}


def _load_claims():
    rows = conn.execute("SELECT id FROM claims WHERE user_id=?", (U,)).fetchall()
    out = []
    for r in rows:
        emb = store.claim_embedding(conn, U, r["id"])
        if emb is None:
            continue
        cm = conn.execute("SELECT concept_id FROM concept_members WHERE claim_id=? AND user_id=? LIMIT 1",
                          (r["id"], U)).fetchone()
        cl = conn.execute("SELECT text FROM claims WHERE id=? AND user_id=?", (r["id"], U)).fetchone()
        out.append({"id": r["id"], "text": cl["text"], "embedding": np.asarray(emb, float),
                    "cluster": cm["concept_id"] if cm else None})
    return out


def _write_membership(claims):
    """Re-point concept_members from claims' (possibly new) cluster labels; drop tiny
    concepts; recompute medoid then contrastive re-anchor. New seeds get fresh ids."""
    from collections import Counter
    counts = Counter(c["cluster"] for c in claims if c["cluster"])
    with conn:
        conn.execute("DELETE FROM concept_members WHERE user_id=?", (U,))
        for c in claims:
            cl = c["cluster"]
            if cl and counts[cl] >= MIN_CONCEPT:
                conn.execute("INSERT OR REPLACE INTO concept_members (user_id, concept_id, claim_id) VALUES (?,?,?)",
                             (U, cl, c["id"]))
        # ensure every live concept row exists for orphan seeds
        live = {cl for cl, n in counts.items() if n >= MIN_CONCEPT}
        for cl in live:
            ex = conn.execute("SELECT 1 FROM concepts WHERE id=? AND user_id=?", (cl, U)).fetchone()
            if not ex:
                conn.execute("INSERT INTO concepts (id, user_id, label, canonical, state, strength, created_at, last_activity) "
                             "VALUES (?,?,?,?, 'active', 1.0, '2026-01-01', '2026-01-01')",
                             (cl, U, "(geom seed)", ""))
        cids = [r["id"] for r in conn.execute("SELECT id FROM concepts WHERE user_id=?", (U,))]
        for cid in cids:
            store.recompute_concept_embedding(conn, U, cid)  # medoid
    with conn:
        consolidate._reanchor_concepts(conn, U)              # contrastive


def reassign(claims):
    """One label-propagation pass: each claim → dominant nearby cluster (squish, density-
    aware via measure's z vs nearest-cluster cohesion); NOVEL claims mint a seed."""
    corpus = [{"id": c["id"], "text": c["text"], "embedding": c["embedding"], "cluster": c["cluster"]}
              for c in claims]
    base = predict.compute_baselines(corpus)
    ms = predict.measure(corpus, corpus, baselines=base, exclude_self=True)
    moved = 0
    for c, m in zip(claims, ms):
        if m["cold_start"]:
            continue
        if m["z"] >= MINT_Z and (m["cluster"] is None):  # novel & no clear home → seed
            new = "cpt_geom_" + c["id"][-12:]
        else:
            new = m["cluster"] or c["cluster"]
        if new and new != c["cluster"]:
            moved += 1
            c["cluster"] = new
    return moved


def split_pass(claims):
    """Push/repulsion at the membership level: any concept whose members fall into two
    separated blobs (consolidate._spread_is_bimodal) is two ideas wearing one concept —
    split along the poles into two concepts. Counters squish-agglomeration; the broad lever."""
    from collections import defaultdict
    groups = defaultdict(list)
    for c in claims:
        if c["cluster"]:
            groups[c["cluster"]].append(c)
    n_split = 0
    for cl, members in groups.items():
        if len(members) < 4:
            continue
        V = np.vstack([m["embedding"] for m in members])
        if not consolidate._spread_is_bimodal(V):
            continue
        G = V @ V.T
        n = V.shape[0]
        i, j = divmod(int(np.argmin(G)), n)
        side = G[i] >= G[j]                       # near pole i → keep; else → new concept
        if side.sum() < 2 or (~side).sum() < 2:
            continue
        new_id = "cpt_split_" + cl[-10:]
        for k, m in enumerate(members):
            if not side[k]:
                m["cluster"] = new_id
        n_split += 1
    return n_split


def evaluate(tag):
    cs = store.all_concepts(conn, U)
    n = len(cs)
    sizes = [len(store.concept_member_ids(conn, U, c["id"])) for c in cs]
    A = np.vstack([store.concept_embedding(conn, U, c["id"]) for c in cs
                   if store.concept_embedding(conn, U, c["id"]) is not None])
    A = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-9, None)
    pc = (A @ A.T)[np.triu_indices(A.shape[0], 1)].mean()
    cells = []
    for gname, gold in golds.items():
        cov = coverage_eval(gold, conn, U, resonance.resonance_context, TAU)
        rch = reach_eval(gold, conn, U, cal)
        cells.append(f"{gname[:4]} C={cov['sr_at_b']:.0%} R={rch['reach_at_b']:.0%}")
    print(f"{tag:18} concepts={n} maxsz={max(sizes)} pc={pc:.3f}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "split"   # split | em
    claims = _load_claims()
    _write_membership(claims)
    evaluate("baseline(LLM)")

    if mode == "split":
        # (A) Task #5: split-only on the baseline LLM membership, iterated until dry.
        for it in range(1, 6):
            ns = split_pass(claims)
            _write_membership(claims)
            evaluate(f"split pass {it} (n={ns})")
            if ns == 0:
                break
    else:
        # (B) co-evolving EM with the push counterforce: reassign(squish) + split(push).
        for it in range(1, 5):
            mv = reassign(claims)
            ns = split_pass(claims)
            _write_membership(claims)
            evaluate(f"em pass {it} (mv={mv} sp={ns})")
            if mv == 0 and ns == 0:
                break
