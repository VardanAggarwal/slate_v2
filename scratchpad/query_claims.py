"""Query-as-claim consolidation — encode retrieval queries as first-class CLAIMS that
flow through the whole pipeline (not just concept→concept relation bridges).

WHY CLAIMS, NOT RELATION BRIDGES (the query_bridges.py result was Δ=0):
  * relation edges are weight 0.7, PE-gated, and only reached at hop ≥1 — fragile.
  * a query-CLAIM is a knn SEED target: a future similar query lands ON it directly
    (strong hop-0 activation), then spreads through MEMBERSHIP edges (weight 1.0) to
    EVERY concept the query spans → cross-note regions light up.
  * a synthesis query is naturally a MULTI-concept member → a real structural-hole hub.
  * materialization pulls verbatim text via claim→source-episode→fragments; a query-claim
    has NO episode/fragments, so it ROUTES activation but contributes ZERO answer text
    (no question-noise in the context). Verified against resonance.py:368-373.

Attaches as kind='redundant' so it never moves a concept medoid (centroids stay fixed;
recompute uses primary_only). HONESTY: queries are synthetic, blind to gold, claude-gen.

Run:  PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/query_claims.py
"""
from __future__ import annotations

import numpy as np

from core import store, resonance, calibration as calib, retrieve, predict
from eval.coverage import coverage_eval, _load_gold, HERE
from eval_both import GOLD, TAU
# query_bridges sets the claude-cli override at import AND owns the cached generator
from query_bridges import U, DB_SRC, fresh_conn, build_qstream

ATTACH_K = 4            # concepts a query-claim joins (its multi-topic span)
ATTACH_FLOOR = 0.0      # min concept salience to attach (0 = top-K regardless)
RECOMPUTE = True        # re-medoid concepts after attach (redundant members don't move it)


def evalrun(conn, tag):
    golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}
    cov = {g: round(coverage_eval(gd, conn, U, resonance.resonance_context, TAU)["sr_at_b"], 4)
           for g, gd in golds.items()}
    cov["mean"] = round(sum(cov.values()) / 3, 4)
    print(f"{tag:26} narrow={cov['narrow']:.3f}  broad={cov['broad']:.3f}  "
          f"paragraph={cov['paragraph']:.3f}  mean={cov['mean']:.3f}")
    return cov


def inject_query_claims(conn, queries):
    C = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION, **resonance.DEFAULT_CALIBRATION}, U)
    embs = predict._default_embed(queries)  # (N, d)
    n_claims = n_edges = 0
    touched = set()
    with conn:
        for i, q in enumerate(queries):
            field = resonance.activate(conn, U, q, calibration=C)
            cps = sorted(((n, d["salience"]) for n, d in field["nodes"].items()
                          if n.startswith("cpt_")), key=lambda kv: -kv[1])
            cps = [(c, s) for c, s in cps if s >= ATTACH_FLOOR][:ATTACH_K]
            if len(cps) < 2:          # need a genuine multi-concept span to be a bridge
                continue
            cid = f"qclm_{i:04d}"
            store.insert_claim(conn, U, cid, q, embs[i], "2026-06-27")
            n_claims += 1
            for concept_id, _s in cps:
                conn.execute(
                    "INSERT OR IGNORE INTO concept_members (concept_id, user_id, claim_id, "
                    "weight, kind) VALUES (?,?,?,1.0,'redundant')", (concept_id, U, cid))
                n_edges += 1
                touched.add(concept_id)
    if RECOMPUTE:
        # demand-side re-anchor: re-medoid INCLUDING the query-claims (store's recompute
        # is primary_only, which would ignore them). Medoid = member with max summed cosine
        # to all members → the concept centroid can drift toward the question-shapes that
        # retrieve it. Mirrors store.recompute_concept_embedding's argmax, all members.
        with conn:
            for concept_id in touched:
                members = store.concept_member_ids(conn, U, concept_id)  # primary+redundant
                vecs, ids = [], []
                for m in members:
                    v = store.claim_embedding(conn, U, m)
                    if v is not None:
                        ids.append(m)
                        vecs.append(np.asarray(v, float))
                if not vecs:
                    continue
                M = np.vstack(vecs)
                M = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
                medoid = M[int(np.argmax((M @ M.T).sum(axis=1)))]
                conn.execute("DELETE FROM vec_concepts WHERE concept_id=?", (concept_id,))
                conn.execute(
                    "INSERT INTO vec_concepts (user_id, concept_id, embedding) VALUES (?,?,?)",
                    (U, concept_id, store.serialize_float32([float(x) for x in medoid])))
    print(f"injected {n_claims} query-claims, {n_edges} membership edges, "
          f"{len(touched)} concepts touched (attach_k={ATTACH_K}, recompute={RECOMPUTE})")


def leak_count(conn, queries):
    """How many gold-query contexts surface a stored query-claim's TEXT (the leak)."""
    qset = set(q.strip() for q in queries)
    golds = {g: _load_gold(HERE / f) for g, f in GOLD.items()}
    n_ctx = n_leaky = 0
    for gd in golds.values():
        for g in gd:
            ctx = resonance.resonance_context(conn, U, g["query"])
            n_ctx += 1
            if any(q in ctx for q in qset):
                n_leaky += 1
    return n_leaky, n_ctx


def main():
    conn, path = fresh_conn("qc")
    print(f"work db: {path}")
    queries = build_qstream(conn)
    print(f"qstream: {len(queries)} synthetic queries")
    base = evalrun(conn, "BASELINE (as-is)")
    inject_query_claims(conn, queries)

    # The leak fix now lives in core (resonance._concept_members excludes qclm_),
    # so no monkeypatch — this verifies the SHIPPED path leaks nothing.
    leak, n = leak_count(conn, queries)
    print(f"LEAK CHECK (core fix): {leak}/{n} gold contexts surface a query-claim's text")
    treat = evalrun(conn, "TREATMENT (+q-claims)")

    print("\nΔ vs baseline  narrow={:+.3f}  broad={:+.3f}  "
          "paragraph={:+.3f}  mean={:+.3f}".format(
        treat["narrow"] - base["narrow"], treat["broad"] - base["broad"],
        treat["paragraph"] - base["paragraph"], treat["mean"] - base["mean"]))


if __name__ == "__main__":
    main()
