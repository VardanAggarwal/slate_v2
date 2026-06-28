"""Prediction error: the one signal the spine hangs from (PRD v2 §"prediction error").

This module is split into two layers that must stay separate:

  1. THE PREDICTOR — `measure()`. A pure sensor. Given a fragment and the memory
     M nearest it, it reports HOW NEW the fragment is relative to M, judged
     against M's own spread, and WHERE it attaches. That is all. It applies no
     policy, holds no thresholds, makes no store/forget decision. Its output
     depends on the embedding (the lens) and on M — never on a bet about the
     future. Swapping the embedding model, even re-embedding the whole DB,
     leaves this layer's contract unchanged.

  2. THE DECISION LAYER — `decide(measurement, calibration)`. Applies the bet.
     The compression boundary (how much of a residual to keep) depends on the
     future use Q, which is unknowable at write time, so it cannot be measured —
     only chosen. That choice is `calibration`. It is owned and fitted at
     CONSOLIDATION (which can re-derive from the immutable raw record, so the bet
     is reversible, and which sees accumulated retrieval signals — a real proxy
     for Q). Consolidation pushes the current calibration (and baselines) out to
     write and retrieve, where the decision is taken. `decide()` takes plain
     dicts and no predictor internals, so it can live at the stage layer; it is
     NOT part of the predictor.

Two contexts, never conflated:
  - M, the MEASUREMENT context (memory) — known; the residual is measured against it.
  - Q, the USE context (future queries) — unknown; it decides whether a given
    compression was right. Calibration is the bridge: a prior over Q and budget B.

Direction (contradict / refine / version) is neither measurement nor this
decision — a residual near an anchor is geometrically identical whether it
contradicts or refines. It is a question of meaning, deferred to
`resolve_direction()` (an LLM), run only over the AMBIGUOUS verdicts.

Embedding concerns (query/passage asymmetry, negation, numbers) are failures of
the lens and are fixed BELOW this module, in the embedding layer — never here.
"""
from __future__ import annotations

import numpy as np

# ── measurement hyperparameters (the instrument's precision, not the bet) ─────
SPAN_K = 6          # vectors that reconstruct a point
STAT_K = 12         # wider sample for estimating a region's spread
SHRINK_N0 = 8.0     # pseudo-count: blend a local estimate toward the prior
GRADUATION_N0 = 4.0  # C11: pseudo-count a REGION must accrue before its own
                     # cohesion is trusted over the prior-over-clusters (cold-start
                     # graduation). n >> N0 → trust local; n << N0 → trust prior.
SD_FLOOR = 0.03
RIDGE_LAMBDA = 0.05  # closer neighbours trusted more (weighted reconstruction)
GRAM_MAX_N = 4000   # cache the full n×n Gram below this (~128MB at n=4000); above, per-row

# ── the calibration profile: POLICY, owned by consolidation, not the predictor ─
# JSON-able and scopeable (global + per-cluster overrides). A benchmark fits it
# offline against SR@B; the decision layer consumes it.
#
# z_echo is the critical knob — it IS the reinforce/escalate line. It is a
# SPREAD-RELATIVE floor (a z-score), never an absolute similarity cutoff: reinforce
# (store nothing) only when a fragment reconstructs at least as tightly as the
# region reconstructs ITSELF — i.e. it is effectively the same claim. Everything
# attached-but-looser is escalated to AMBIGUOUS, because a polarity-blind encoder
# makes a contradiction geometrically identical to a paraphrase (PRD: "a low
# residual is necessary but not sufficient"), so direction must be left to the
# resolver, not silently absorbed as a reinforcement. prox_margin sets the
# anchor-present line that splits AMBIGUOUS (sits on an anchor) from NOVEL (open
# territory); it tunes resolver volume. z_echo is strongly negative because a
# near-identical restatement reconstructs far better than the region's own mean;
# the exact value is fitted per corpus/region at consolidation.
Z_ECHO = -3.5
PROX_MARGIN = 1.0
DEFAULT_CALIBRATION = {"z_echo": Z_ECHO, "prox_margin": PROX_MARGIN,
                       "per_cluster": {}}

_PREDICTED, _AMBIGUOUS, _NOVEL = "PREDICTED", "AMBIGUOUS", "NOVEL"
# Smallest corpus Y that still admits a measurement. The instrument SCALES to
# corpus size: the top-k operators (_topk_idx) already cap k at the pool size, so
# SPAN_K/STAT_K degrade to "use all available" on a small Y — which is what lets
# measure() serve a within-note corpus (scan: Y = the note's own sentences) as
# well as the full memory (Write: Y = every claim). Below this floor there are too
# few peers to estimate any spread, so we report cold_start. Two peers is the
# minimum that yields a (crude) σ; for the large-memory Write path this floor is
# never the binding constraint, so that behaviour is unchanged.
WARMUP_MIN_CORPUS = 2


# ── injected defaults (production wiring) ─────────────────────────────────────
def _default_embed(texts: list[str]) -> np.ndarray:
    from core.encode import get_embedder
    arr = get_embedder().encode(texts, normalize_embeddings=True,
                                show_progress_bar=False)
    return np.atleast_2d(np.asarray(arr, dtype=float))


def _default_stance(premise: str, hypothesis: str) -> str:
    from core.encode import classify_stance
    return classify_stance(premise, hypothesis)


def _split(text: str) -> list[str]:
    from core.encode import split_sentences
    out = split_sentences(text)
    return out or [text.strip()]


# ── geometry ──────────────────────────────────────────────────────────────────
def _topk_idx(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores. argpartition is O(n) vs argsort's
    O(n log n); we only ever need the top SPAN_K / STAT_K, never a full order.
    The returned set is identical to argsort(-scores)[:k] (order within the set
    is irrelevant — every consumer sums/projects over it, order-independent).
    Falls back to a full ordered argsort when k covers the whole pool."""
    n = scores.shape[0]
    if k >= n:
        return np.argsort(-scores)
    return np.argpartition(-scores, k)[:k]


def _project_residual(e: np.ndarray, span: np.ndarray,
                      weights: np.ndarray | None = None) -> tuple[float, float]:
    """Predictability = norm of e's reconstruction from the span; residual = the
    part left over. With weights, neighbours are trusted in proportion to
    similarity via ridge regularisation (subspace projection alone is invariant
    to basis scaling, so weighting must enter as ridge)."""
    if span.shape[0] == 0:
        return 0.0, 1.0
    if weights is None:
        coeffs, *_ = np.linalg.lstsq(span.T, e, rcond=None)
        proj = span.T @ coeffs
    else:
        w = np.clip(np.asarray(weights, dtype=float), 1e-3, None)
        G = span @ span.T
        a = np.linalg.solve(G + RIDGE_LAMBDA * np.diag(1.0 / (w * w)), span @ e)
        proj = span.T @ a
    p = min(float(np.linalg.norm(proj)), 1.0)
    return p, float(np.sqrt(max(0.0, 1.0 - p * p)))


def _residual_against(v: np.ndarray, pool: np.ndarray) -> float:
    if pool.shape[0] == 0:
        return 1.0
    sims = pool @ v
    idx = _topk_idx(sims, SPAN_K)
    return _project_residual(v, pool[idx], np.clip(sims[idx], 1e-3, None))[1]


# The one primitive every wrapper hangs from: residual of X against the span of
# its nearest neighbours in Y. Write/Retrieve/Consolidate are all this op with X
# and Y swapped (PRD: "one estimator"). Public alias — scan/assembly/guard build
# on it; the predictor owns magnitude, the LLM owns direction.
residual_against = _residual_against


def residuals_against(X: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Batch `residual_against`: residual norm of EACH row of `X` against its
    SPAN_K nearest neighbours in `pool`. Returns only the magnitude — no
    baselines/z/peer geometry — which is the one quantity `assembly.assemble`
    reads from a measurement. Numerically identical to the `residual` field
    `measure()` returns (measure's two-stage STAT_K→SPAN_K top-k reduces to this
    single SPAN_K projection), so it is a drop-in that skips the per-step baseline
    recompute measure() would do for fields assembly never uses. Empty pool →
    all-ones (every probe maximally novel)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if pool.shape[0] == 0:
        return np.ones(X.shape[0])
    return np.array([_residual_against(X[i], pool) for i in range(X.shape[0])])


def residual_direction(v: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """The part of `v` its nearest SPAN_K neighbours in `pool` CANNOT reconstruct,
    as a VECTOR — the *direction* of the surprise, not just its magnitude (the
    companion to `residual_against`). Same weighted span-projection as
    `_project_residual`; returns `v − proj` (unnormalised). Empty pool → `v`.

    Used by Retrieve.R3 (borrow): the query's residual against its OWN topic is the
    uncovered-nuance direction to match against off-topic memory."""
    v = np.asarray(v, dtype=float)
    if pool.shape[0] == 0:
        return v
    sims = pool @ v
    idx = _topk_idx(sims, SPAN_K)
    span = pool[idx]
    w = np.clip(sims[idx], 1e-3, None)
    G = span @ span.T
    a = np.linalg.solve(G + RIDGE_LAMBDA * np.diag(1.0 / (w * w)), span @ v)
    return v - span.T @ a


def _loo_residual(i: int, C: np.ndarray, sims_row: np.ndarray | None = None) -> float:
    """A member's residual against its STAT_K nearest *other* vectors — the SAME
    operator a probe gets in `_measure_one` (global nearest), minus self. The
    baseline and the probe MUST share this operator or their z-scores are not
    comparable (a within-cluster-only baseline measures a different distribution
    than the probe does, and the z it yields is meaningless). PRD: "scored as a
    z-score against the residuals of Y's own members ... leave-one-out."

    `sims_row` is the precomputed Gram row C @ C[i]; passed in by
    `compute_baselines` so the n×n products are formed once, not per member."""
    sims = (C @ C[i]) if sims_row is None else sims_row.copy()
    sims[i] = -np.inf
    return _residual_against(C[i], C[_topk_idx(sims, STAT_K)])


def corpus_prior(C: np.ndarray, sample: int = 64, seed: int = 0,
                 G: np.ndarray | None = None) -> tuple[float, float]:
    """Pooled member-residual stats — fallback prior when there aren't enough
    clusters to form a prior-over-clusters. Reuses the Gram matrix G when given."""
    n = C.shape[0]
    if n <= SPAN_K:
        return 0.7, 0.2
    rng = np.random.default_rng(seed)
    res = np.array([_loo_residual(int(i), C, G[i] if G is not None else None)
                    for i in rng.choice(n, size=min(sample, n), replace=False)])
    return float(res.mean()), float(max(res.std(), SD_FLOOR))


def compute_baselines(corpus: list[dict], C: np.ndarray | None = None) -> dict:
    """Per-cluster cohesion (μ, σ of member residuals) plus a prior-OVER-CLUSTERS.
    OBJECTIVE — derived from M, no bet about Q. Recompute when clusters change,
    i.e. at consolidation. This is measurement context, so it is a `measure()`
    input — not calibration.

    Members are scored with `_loo_residual` — the SAME global-nearest operator a
    probe gets in `_measure_one`, not a within-cluster-only one — so the spread a
    fragment's z is measured against is the spread it will actually be compared
    to. PRD: "This makes the residual comparable across regions." """
    if C is None:
        C = np.vstack([np.asarray(c["embedding"], dtype=float) for c in corpus])
    # Form the n×n Gram matrix ONCE (one BLAS gemm) and hand each member its row,
    # rather than recomputing C @ C[i] per member (was O(n²·d) of redundant
    # matvecs). Skip above GRAM_MAX_N, where the n² floats stop fitting the host
    # comfortably — there _loo_residual falls back to per-row products.
    G = C @ C.T if C.shape[0] <= GRAM_MAX_N else None
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(corpus):
        cl = c.get("cluster")
        if cl is not None:
            groups.setdefault(cl, []).append(i)

    # First pass: each region's RAW cohesion (μ, σ) + its member count. The prior
    # is the population expectation over these raw estimates, so it must be formed
    # BEFORE the cold-start shrink (else it would chase its own shrunk clusters).
    raw: dict[str, tuple[float, float, int]] = {}
    for cl, idxs in groups.items():
        if len(idxs) < 2:
            continue
        res = np.array([_loo_residual(i, C, G[i] if G is not None else None)
                        for i in idxs])
        raw[cl] = (float(res.mean()), float(max(res.std(), SD_FLOOR)), len(idxs))

    if len(raw) >= 2:
        mus = np.array([m for m, _, _ in raw.values()])
        sds = np.array([s for _, s, _ in raw.values()])
        prior = (float(mus.mean()), float(max(sds.mean(), SD_FLOOR)))
    else:
        prior = corpus_prior(C, G=G)

    # C11 cold-start graduation: a region's own spread is only trustworthy once it
    # has accrued members — a 2-member region's σ is estimated from 2 points, so
    # betting routing on it manufactures noise. Shrink each region toward the
    # prior-over-clusters by its size (weight n/(n+N0)): a fresh region reports ≈
    # the prior, a matured one reports ≈ its local cohesion. Continuous, so a z
    # measured against a region does not jump as the region crosses a count.
    prior_mu, prior_sd = prior
    clusters: dict[str, tuple[float, float]] = {}
    for cl, (mu_raw, sd_raw, n) in raw.items():
        w = n / (n + GRADUATION_N0)
        mu = w * mu_raw + (1.0 - w) * prior_mu
        sd = max(SD_FLOOR, w * sd_raw + (1.0 - w) * prior_sd)
        clusters[cl] = (mu, sd)
    return {"clusters": clusters, "prior": prior}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — THE PREDICTOR: pure measurement. No calibration, no policy.
# ══════════════════════════════════════════════════════════════════════════════
def _measure_one(frag: str, e: np.ndarray, corpus: list[dict], C: np.ndarray,
                 baselines: dict, sims: np.ndarray | None = None) -> dict:
    """How new is `frag` relative to M, and where does it attach. Pure.

    `sims` is the precomputed similarity column C @ e; `measure` forms all
    fragment columns in one gemm and passes them in, avoiding a matvec per
    fragment."""
    if sims is None:
        sims = C @ e
    nearest = int(np.argmax(sims))
    nearest_sim = float(sims[nearest])
    anchor = corpus[nearest]

    nbr_idx = _topk_idx(sims, STAT_K)
    nbrs = C[nbr_idx]
    nbr_sims = sims[nbr_idx]
    nbr_clusters = [corpus[i].get("cluster") for i in nbr_idx]
    r_e = _residual_against(e, nbrs)

    prior_mu, prior_sd = baselines["prior"]
    contrib = [(s, baselines["clusters"][cl])
               for s, cl in zip(nbr_sims, nbr_clusters)
               if cl in baselines["clusters"]]
    if contrib:
        w = np.clip([s for s, _ in contrib], 0, None)
        if w.sum() <= 0:
            w = np.ones(len(contrib))
        mus = np.array([m for _, (m, _) in contrib])
        sds = np.array([sd for _, (_, sd) in contrib])
        mu_local, sd_local = float((w * mus).sum() / w.sum()), float((w * sds).sum() / w.sum())
        n_eff = len(contrib)
    else:
        local = np.array([_residual_against(nbrs[j], np.delete(nbrs, j, axis=0))
                          for j in range(nbrs.shape[0])])
        mu_local = local.mean()
        sd_local = local.std() if len(local) > 1 else prior_sd
        n_eff = len(local)

    mu = (n_eff * mu_local + SHRINK_N0 * prior_mu) / (n_eff + SHRINK_N0)
    sd = max(SD_FLOOR, (n_eff * sd_local + SHRINK_N0 * prior_sd) / (n_eff + SHRINK_N0))
    z = (r_e - mu) / sd

    # Attachment INGREDIENTS only — the prox_margin threshold is policy, applied
    # in decide(). The predictor reports the peer-similarity distribution; it
    # does not judge "attached".
    if nbrs.shape[0] >= 2:
        G = nbrs @ nbrs.T
        np.fill_diagonal(G, -1.0)
        peer_nn = G.max(axis=1)
        peer_mean, peer_std = float(peer_nn.mean()), float(peer_nn.std())
    else:
        peer_mean, peer_std = 0.0, 0.0

    dom = max((c for c in nbr_clusters if c is not None),
              key=[c for c in nbr_clusters].count, default=None)

    return {"text": frag, "residual": round(r_e, 4), "z": round(float(z), 3),
            "nearest_sim": round(nearest_sim, 4),
            "anchor_id": anchor["id"], "anchor_text": anchor["text"],
            "peer_mean": round(peer_mean, 4), "peer_std": round(peer_std, 4),
            "cluster": dom, "cold_start": False}


def measure(x: str | list[str] | list[dict], corpus: list[dict], *,
            embed=None, split=None, baselines: dict | None = None,
            exclude_self: bool = False) -> list[dict]:
    """THE PREDICTOR. Return one pure measurement per fragment of X against Y.

    X (`x`) — the probes. Either:
      - a str           → split into sentence fragments (the Write entrypoint), or
      - a list[str]     → pre-split fragments, used verbatim, or
      - a list[dict]    → items {"text","embedding"(opt)} whose embeddings are
                          reused as-is (no re-embed) — lets a wrapper measure
                          memory rows it already holds.
    Y (`corpus`) — the measurement context: items {"id","text","embedding"(opt),
      "cluster"(opt)}. This is the ONE knob the spine turns: Write sends X=note,
      Y=memory; Scan sends X=Y=the note's own sentences; Retrieve sends
      X=candidates, Y=query+assembly; Consolidate sends X=Y=a cluster's members.

    exclude_self — when X ⊆ Y (scan, guard), drop each probe's own row from its
      neighbourhood so a fragment is never reconstructed from itself (leave-one-
      out). Matched by exact-identity (sim == 1.0) on the normalised vectors.

    baselines: precomputed per-cluster cohesion (compute_baselines); built on the
               fly if omitted. Objective measurement context — not calibration.
    No routing, no thresholds, no store/forget decision — those are decide()'s.
    """
    embed = embed or _default_embed
    split = split or _split

    if isinstance(x, str):
        frags = split(x)
        texts = frags
        xe = None
    elif x and isinstance(x[0], dict):
        frags = [f.get("text") for f in x]
        texts = frags
        xe = np.vstack([np.asarray(f["embedding"], dtype=float) for f in x]) \
            if all(f.get("embedding") is not None for f in x) else None
    else:
        frags = list(x)
        texts = frags
        xe = None
    if not frags:
        return []
    fe = xe if xe is not None else embed([t for t in texts])

    if len(corpus) < WARMUP_MIN_CORPUS:
        return [{"text": f, "residual": 1.0, "z": 0.0, "nearest_sim": None,
                 "anchor_id": None, "anchor_text": None, "peer_mean": 0.0,
                 "peer_std": 0.0, "cluster": None, "cold_start": True} for f in frags]

    missing = [c for c in corpus if c.get("embedding") is None]
    if missing:
        for c, v in zip(missing, embed([c["text"] for c in missing])):
            c["embedding"] = v
    C = np.vstack([np.asarray(c["embedding"], dtype=float) for c in corpus])
    baselines = baselines or compute_baselines(corpus, C)
    fe = np.asarray(fe, dtype=float)
    S = C @ fe.T  # (n, F): all fragment similarity columns in one gemm
    if exclude_self:
        # X ⊆ Y: blank each probe's own corpus row (the exact self-match) so it is
        # measured leave-one-out, the same operator compute_baselines uses.
        S = S.copy()
        for j in range(S.shape[1]):
            hit = np.where(S[:, j] >= 1.0 - 1e-6)[0]
            if hit.size:
                S[hit[0], j] = -np.inf
    return [_measure_one(f, fe[j], corpus, C, baselines, S[:, j])
            for j, f in enumerate(frags)]


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — THE DECISION: applies calibration. Owned by consolidation/stage.
# Takes plain measurements + a calibration profile; holds no predictor internals.
# ══════════════════════════════════════════════════════════════════════════════
def calib_value(profile: dict | None, key: str, default, scope: str | None = None):
    """Resolve ONE calibration value: per-scope (e.g. per-cluster) override →
    global → default. The generic form of `_thresholds`, shared by the three
    wrappers (scan/assembly/guard) so policy lookup is identical everywhere and
    every keep/cut/stop/forget bet lives in ONE scopeable, JSON-able profile —
    fitted at consolidation and pushed down (PRD: "one calibration profile")."""
    if not profile:
        return default
    scoped = profile.get("per_cluster", {}).get(scope) if scope else None
    if scoped and key in scoped:
        return scoped[key]
    return profile.get(key, default)


def _thresholds(calib: dict, cluster: str | None) -> tuple[float, float]:
    return (calib_value(calib, "z_echo", Z_ECHO, cluster),
            calib_value(calib, "prox_margin", PROX_MARGIN, cluster))


def decide(measurements: list[dict], calibration: dict | None = None) -> dict:
    """Apply the calibration bet to pure measurements → routes + decisions.

    Returns the verdicts and the aggregates a stage acts on:
      reinforced — anchor ids to strengthen (predicted, nothing stored)
      new        — fragment texts to encode (genuinely novel, no anchor)
      to_resolve — (fragment, anchor) residuals whose DIRECTION the LLM resolves
      ranking    — fragment indices, most-surprising first (ordinal)
    Each stored fragment carries `weight` — its residual share within THIS batch
    (relative, drift-free); for proportional budget/resolution, never absolute.
    """
    calib = calibration or DEFAULT_CALIBRATION
    out = []
    for m in measurements:
        if m["cold_start"]:
            out.append({**m, "route": _NOVEL, "store": True, "reinforce": False,
                        "resolve": False, "weight": 0.0})
            continue
        z_echo, prox_margin = _thresholds(calib, m["cluster"])
        attached = (m["peer_mean"] == 0.0 and m["peer_std"] == 0.0) or \
                   m["nearest_sim"] >= m["peer_mean"] - prox_margin * m["peer_std"]
        # Reinforce ONLY a near-identical restatement (reconstructs as tightly as
        # the region reconstructs itself). Anything attached-but-looser escalates
        # to the resolver — geometry can't tell a contradiction from a paraphrase,
        # so it must not be silently swallowed as a reinforcement. Anchor-presence,
        # not a magnitude ceiling, is what splits AMBIGUOUS (on an anchor) from
        # NOVEL (open territory) — PRD: "Novel ... with no anchor present."
        if m["z"] <= z_echo:
            v = {"route": _PREDICTED, "store": False, "reinforce": True, "resolve": False}
        elif attached:
            v = {"route": _AMBIGUOUS, "store": True, "reinforce": False, "resolve": True}
        else:
            v = {"route": _NOVEL, "store": True, "reinforce": False, "resolve": False,
                 "anchor_id": None, "anchor_text": None}
        out.append({**m, **v, "weight": 0.0})

    total = sum(v["residual"] for v in out if v["store"]) or 1.0
    for v in out:
        v["weight"] = round(v["residual"] / total, 4) if v["store"] else 0.0

    reinforced = [v["anchor_id"] for v in out if v["reinforce"]]
    new = [v["text"] for v in out if v["route"] == _NOVEL]
    to_resolve = [{"fragment": v["text"], "anchor_id": v["anchor_id"],
                   "anchor_text": v["anchor_text"], "weight": v["weight"]}
                  for v in out if v["resolve"]]
    ranking = sorted(range(len(out)), key=lambda i: -out[i]["z"])
    return {"fragments": out, "reinforced": reinforced, "new": new,
            "to_resolve": to_resolve, "ranking": ranking}


# ── expensive tier: direction resolution (the LLM's job) ──────────────────────
def resolve_direction(fragment: str, anchor_text: str, classify=None) -> dict:
    """Decide the DIRECTION of an AMBIGUOUS residual against the anchor it sits
    on. contradiction → −1 (held strongly); refine / reinforce → +1."""
    classify = classify or _default_stance
    s = classify(anchor_text, fragment)
    if s == "contradiction":
        return {"direction": "contradict", "sign": -1.0}
    if s == "entailment":
        return {"direction": "reinforce", "sign": +1.0}
    return {"direction": "refine", "sign": +1.0}


# ── convenience: compose the two layers (real stages call them separately, with
#    consolidation's pushed-down baselines + calibration) ───────────────────────
def prediction_error(text: str, corpus: list[dict], *, embed=None, split=None,
                     baselines: dict | None = None, calibration: dict | None = None) -> dict:
    return decide(measure(text, corpus, embed=embed, split=split, baselines=baselines),
                  calibration)
