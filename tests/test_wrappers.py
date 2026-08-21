"""Synthetic-corpus tests for the three predictor wrappers (execution plan §0):
scan (segment/decompose), assembly (stop/allocate/dedupe), guard (merge/forget).

Embeddings are built directly as near-orthogonal topic directions + small noise,
so structure (planted boundaries, planted duplicates, planted nuance) is known
exactly and correctness is checkable without an LLM or the live DB. The residual
primitive reconstructs a point from its SPAN_K nearest neighbours, so within a
topic residual is low and across topics it spikes — the signal every wrapper reads.
"""
import numpy as np
import pytest

from core import scan, assembly, guard

DIM = 48
SEED = 7


def _topic_dirs(n_topics, rng):
    """n_topics near-orthogonal unit directions."""
    M = rng.standard_normal((n_topics, DIM))
    Q, _ = np.linalg.qr(M.T)        # orthonormal columns
    return Q.T[:n_topics]


def _frag(direction, rng, noise=0.04):
    v = direction + noise * rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


def _correlated_dirs(cos, rng):
    """Two unit directions with a chosen cosine — for OVERLAPPING topics whose
    cross-topic similarity stays well above SIM_FLOOR, so a boundary can only be
    found via the relative-jump path, not the absolute floor."""
    Q = _topic_dirs(2, rng)
    a = Q[0]
    b = cos * Q[0] + np.sqrt(1 - cos * cos) * Q[1]
    return a, b / np.linalg.norm(b)


def _blend(d1, d2, a, rng, noise=0.04):
    """A fragment a*d1 + (1-a)*d2 — partially reconstructable from a d1 set."""
    v = a * d1 + (1 - a) * d2 + noise * rng.standard_normal(DIM)
    return v / np.linalg.norm(v)


# ── scan: planted topic shifts ─────────────────────────────────────────────────
def test_scan_cuts_near_planted_boundaries():
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    # 5 frags topic A, 5 topic B, 5 topic C → boundaries planted at 5 and 10.
    E = np.vstack([_frag(dirs[t], rng) for t in (0, 0, 0, 0, 0,
                                                 1, 1, 1, 1, 1,
                                                 2, 2, 2, 2, 2)])
    cuts = scan.boundaries(E)
    planted = {5, 10}
    # every planted boundary has a detected cut within ±1
    for p in planted:
        assert any(abs(c - p) <= 1 for c in cuts), f"missed boundary {p}: {cuts}"
    # and no spurious over-segmentation
    assert len(cuts) <= len(planted) + 1, f"over-segmented: {cuts}"


def test_scan_single_topic_no_cuts():
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    E = np.vstack([_frag(d, rng) for _ in range(8)])
    assert scan.boundaries(E) == []


def test_segment_partitions_completely():
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    E = np.vstack([_frag(dirs[t], rng) for t in (0, 0, 0, 1, 1, 1)])
    segs = scan.segment(E)
    assert sorted(i for s in segs for i in s) == list(range(6))  # covers all, no dup
    assert len(segs) == 2


# ── assembly: dedupe + stop ─────────────────────────────────────────────────────
def test_assemble_skips_duplicates_and_stops():
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    # 3 distinct info items + 4 near-duplicates of the first.
    distinct = [_frag(dirs[0], rng), _frag(dirs[1], rng), _frag(dirs[2], rng)]
    dups = [_frag(dirs[0], rng, noise=0.02) for _ in range(4)]
    cand = np.vstack(distinct + dups)
    out = assembly.assemble(cand, k=6, calibration={"gain_floor": 0.35})
    # picks roughly the 3 distinct themes, not all 7 — duplicates add no residual
    assert len(out["chosen"]) <= 4
    assert out["stopped"], "should stop on marginal gain, not exhaust k"
    # the three distinct themes (indices 0,1,2) are each represented
    chosen_dirs = {int(np.argmax(np.abs(dirs @ cand[i]))) for i in out["chosen"]}
    assert chosen_dirs == {0, 1, 2}


def test_assemble_shares_sum_to_one():
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    cand = np.vstack([_frag(dirs[t], rng) for t in (0, 1, 2)])
    out = assembly.assemble(cand)
    assert out["chosen"]                          # picked something
    assert abs(sum(out["shares"]) - 1.0) < 1e-2   # normalised (4-dp rounded shares)


def test_assemble_seed_context_suppresses_covered():
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    # seed already covers topic 0; a topic-0 candidate should not be the first pick.
    seed = np.vstack([_frag(dirs[0], rng) for _ in range(3)])
    cand = np.vstack([_frag(dirs[0], rng), _frag(dirs[1], rng)])
    out = assembly.assemble(cand, seed=seed)
    assert out["chosen"][0] == 1                  # the novel topic, not the covered


# ── guard: safe forget / safe merge (thin iterators over measure(), read z) ──────
def _rows(vecs, cluster="c"):
    """Wrap raw vectors as memory-row dicts for the measure()-based guard API."""
    return [{"id": i, "text": str(i), "embedding": v, "cluster": cluster}
            for i, v in enumerate(vecs)]


def test_guard_forget_ranks_and_gates_by_z():
    """forget() measures each member leave-one-out vs its cluster. A near-duplicate
    member is more rebuildable (lower z) than a cluster outlier — and a z_forget
    line between them drops the redundant, protects the outlier."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    members = [_frag(dirs[0], rng) for _ in range(8)]      # cohesive topic-0 body
    members.append(_frag(dirs[0], rng, noise=0.01))        # idx 8: near-duplicate
    members.append(_frag(dirs[1], rng))                    # idx 9: outlier nuance
    res = guard.forget(_rows(members))
    z_red, z_uniq = res[8]["z"], res[9]["z"]
    assert z_red < z_uniq, (z_red, z_uniq)                 # ranking (instrument)
    cut = (z_red + z_uniq) / 2                             # fitted gate between them
    g = guard.forget(_rows(members), calibration={"z_forget": cut})
    assert g[8]["safe_to_drop"] and not g[9]["safe_to_drop"]


def test_guard_merge_folds_covered_keeps_nuance():
    """merge() measures each loser vs the survivor group. A loser the survivors
    cover folds (low z); one carrying new nuance is kept (high z)."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    survivors = _rows([_frag(dirs[0], rng) for _ in range(8)])
    losers = _rows([_frag(dirs[0], rng),          # idx 0: covered by survivors
                    _frag(dirs[1], rng)])         # idx 1: nuance survivors lack
    res = guard.merge(losers, survivors)
    assert res[0]["z"] < res[1]["z"]
    cut = (res[0]["z"] + res[1]["z"]) / 2
    g = guard.merge(losers, survivors, calibration={"z_forget": cut})
    assert g[0]["safe_to_drop"] and not g[1]["safe_to_drop"]


def test_guard_verdicts_are_keyed_by_member_id():
    """Both guard heads are consumed by ID, not by position: consolidate's prune
    and merge-guard build their drop lists as [v["id"] for v in verdicts]. The
    existing wrapper tests index res[i] positionally, which is exactly how a
    verdict with no "id" survived here and crashed the nightly run instead."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    members = _rows([_frag(dirs[0], rng) for _ in range(4)])
    for i, m in enumerate(members):
        m["id"] = f"clm_{i}"
    got = [v["id"] for v in guard.forget(members)]
    assert got == [m["id"] for m in members]

    losers, survivors = members[:2], members[2:]
    assert [v["id"] for v in guard.merge(losers, survivors)] == ["clm_0", "clm_1"]

    # cold start takes the other early return inside measure() — same contract.
    thin = [dict(members[0], id="clm_solo")]
    assert guard.forget(thin)[0]["id"] == "clm_solo"


def test_guard_cold_start_never_drops():
    """Too few peers to estimate spread ⇒ cold_start ⇒ never forget blindly."""
    rng = np.random.default_rng(SEED)
    d = _topic_dirs(1, rng)[0]
    res = guard.merge(_rows([_frag(d, rng)]), _rows([_frag(d, rng)]))  # |Y|=1 < warmup
    assert res[0]["cold_start"] and not res[0]["safe_to_drop"]


def test_guard_empty_remaining_is_irreplaceable():
    rng = np.random.default_rng(SEED)
    v = _frag(_topic_dirs(1, rng)[0], rng)
    assert guard.reconstruction_residual(v, np.empty((0, DIM))) == 1.0


# ── harder cases: exercise the THRESHOLDS, not just flattering geometry ─────────
def test_scan_overlapping_topics_spread_relative():
    """Two topics that SHARE structure (cross-topic cosine ≈ 0.45). The boundary
    sentence's nearest_sim to its prefix drops from ~1.0 (within A) to ~0.45 (A→B)
    — a spread-relative outlier even though 0.45 is not a low similarity in
    absolute terms. Proves the cut is relative to the note's own spread, not an
    absolute floor."""
    rng = np.random.default_rng(SEED)
    a, b = _correlated_dirs(0.45, rng)
    E = np.vstack([_frag(a, rng) for _ in range(5)] +
                  [_frag(b, rng) for _ in range(5)])  # boundary planted at 5
    cuts = scan.boundaries(E)
    assert any(abs(c - 5) <= 1 for c in cuts), f"missed boundary: {cuts}"
    assert len(cuts) <= 2, f"over-segmented overlapping topics: {cuts}"


def test_scan_variable_resolution_fine_where_surprising():
    """W3: granularity tracks surprise. A high-surprise run (each sentence a fresh
    topic) stays FINE — mostly singletons; a low-surprise run (one topic repeated)
    FOLDS into a single coarse fragment. fragments() partitions the note fully."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(5, rng)
    # 0..3: four different topics back-to-back (every sentence surprises its prefix)
    # 4..8: topic-4 repeated five times (each predictable from the last)
    surprising = [_frag(dirs[t], rng) for t in (0, 1, 2, 3)]
    folded = [_frag(dirs[4], rng) for _ in range(5)]
    E = np.vstack(surprising + folded)
    frags = scan.fragments(E)
    assert sorted(i for f in frags for i in f) == list(range(9))  # full partition
    hi = [f for f in frags if f[0] < 4]      # fragments opening in the surprising run
    lo = [f for f in frags if f[0] >= 4]     # fragments opening in the folded run
    # surprising run is kept finer than the folded run collapses it
    assert len(hi) > len(lo)
    assert max(len(f) for f in lo) >= 3, f"low-surprise run did not fold: {lo}"


def test_scan_fragments_finer_than_segments():
    """Variable resolution is a refinement of segmentation: every hard boundary is
    still a fragment edge, but fragments may sub-split a segment where surprise
    rises without a full topic break — so it is never coarser than segment()."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    E = np.vstack([_frag(dirs[t], rng) for t in (0, 0, 0, 1, 1, 2, 2, 2)])
    segs = scan.segment(E)
    frags = scan.fragments(E)
    assert len(frags) >= len(segs)
    # every segment boundary (a segment's first index) is also a fragment boundary
    seg_starts = {s[0] for s in segs}
    frag_starts = {f[0] for f in frags}
    assert seg_starts <= frag_starts


def test_scan_gradual_drift_does_not_oversegment():
    """A chain that rotates smoothly from topic A to orthogonal topic B in tiny
    steps. nearest_sim is the MAX over the whole prefix, so each sentence still
    matches a recent neighbour even as the note drifts — no single sentence is an
    outlier, so scan must not chop it up."""
    rng = np.random.default_rng(SEED)
    Q = _topic_dirs(2, rng)
    thetas = np.linspace(0, np.pi / 2, 12)
    E = np.vstack([np.cos(t) * Q[0] + np.sin(t) * Q[1] for t in thetas])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    assert len(scan.boundaries(E)) <= 1, "gradual drift falsely segmented"


def test_assemble_near_threshold_duplicate_excluded_by_floor():
    """An item that PARTIALLY overlaps the seed sits near the gain line. With the
    floor set between it and a fully-novel item, the partial is dropped and STOP
    fires; with a low floor both are kept, novel first."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    seed = np.vstack([_frag(dirs[0], rng) for _ in range(3)])     # covers topic 0
    partial = _blend(dirs[0], dirs[1], 0.85, rng)                 # mostly covered
    novel = _frag(dirs[2], rng)                                   # fully new
    cand = np.vstack([partial, novel])
    low = assembly.assemble(cand, seed=seed, calibration={"gain_floor": 0.0})
    assert low["chosen"][0] == 1                                  # novel picked first
    g_novel, g_partial = low["gains"][0], low["gains"][1]
    assert g_partial < g_novel                                   # partial is less new
    mid = (g_partial + g_novel) / 2
    strict = assembly.assemble(cand, seed=seed, calibration={"gain_floor": mid})
    assert strict["chosen"] == [1] and strict["stopped"]         # partial excluded


def test_assemble_value_floor_excludes_tangential_novel():
    """R2/R7 relevance-aware STOP. A span ORTHOGONAL to the query has high residual
    (fully novel vs the assembly) but ~0 relevance — the over-injection failure mode
    in retrieval, where Y grows from a tiny seed so raw residual never saturates.
    The default gain_floor STOP pads it in; the value_floor STOP (residual×relevance)
    drops it."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(3, rng)
    q = _frag(dirs[0], rng)                          # the query
    relevant = _blend(dirs[0], dirs[1], 0.7, rng)    # on-topic, partly novel → high weight
    tangential = _frag(dirs[2], rng)                 # orthogonal → high residual, ~0 weight
    cand = np.vstack([relevant, tangential])
    w = [float(q @ relevant), float(q @ tangential)]  # relevance = cosine to the query

    # raw-residual STOP (default): tangential's high residual pads it in
    loose = assembly.assemble(cand, seed=q[None], weights=w,
                              calibration={"gain_floor": 0.0})
    assert set(loose["chosen"]) == {0, 1}            # both kept — over-injection
    assert len(loose["values"]) == len(loose["chosen"])

    # value-based STOP: tangential has high residual but ~0 value → excluded, STOP fires
    strict = assembly.assemble(cand, seed=q[None], weights=w,
                               calibration={"gain_floor": 0.0, "value_floor": 0.15})
    assert strict["chosen"] == [0] and strict["stopped"]


def test_guard_borderline_between_redundant_and_unique():
    """A blended loser is partially reconstructable — measure()'s z must place it
    BETWEEN a pure restatement and a unique fact (monotone), so any fitted
    z_forget splits the set consistently."""
    rng = np.random.default_rng(SEED)
    dirs = _topic_dirs(2, rng)
    survivors = _rows([_frag(dirs[0], rng) for _ in range(8)])
    losers = _rows([_frag(dirs[0], rng),                 # restatement
                    _blend(dirs[0], dirs[1], 0.6, rng),  # blend
                    _frag(dirs[1], rng)])                # unique
    z = [m["z"] for m in guard.merge(losers, survivors)]
    assert z[0] < z[1] < z[2], z
    cut = (z[0] + z[1]) / 2                               # split below the blend
    g = guard.merge(losers, survivors, calibration={"z_forget": cut})
    assert g[0]["safe_to_drop"] and not g[1]["safe_to_drop"] and not g[2]["safe_to_drop"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
