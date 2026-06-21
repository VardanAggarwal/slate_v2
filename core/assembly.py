"""Iterative residual-vs-assembly — wrapper #2 over the predictor (plan §0).

A THIN ITERATOR over `predict.measure()`, like scan and guard. The spine is
`measure(X, Y)`; assembly is the choice of X, Y and reading the **residual** field
— but unlike the others it iterates: Y GROWS one item per step, so it is a loop of
measure() calls, not a single one.

  X = the remaining candidates.
  Y = the seed (e.g. the query) + everything chosen so far — the growing assembly.
  read = `residual` (a candidate's unique contribution beyond what Y already holds).

Greedily add the candidate of highest residual × relevance; STOP when the best
marginal residual falls below `gain_floor` — the assembly has saturated and more
items are padding. Serves Retrieve.Stop/Allocate (pull a thread only while it
deepens the answer; split budget by residual share) and Consolidate.Dedupe (a
near-duplicate has residual≈0 against what's chosen, so it is never picked).

Closed-loop: the pick decision changes Y, which changes the next measurement —
so it cannot collapse to one open-loop measure()/decide() pair (that is why it
loops). The STOP/budget policy comes from the scopeable calibration profile.
"""
from __future__ import annotations

import numpy as np

from core.predict import calib_value, measure

# ── POLICY lives in a calibration PROFILE, not in constants ───────────────────
# Both encode the compression/budget BET the PRD assigns to calibration owned and
# FITTED at consolidation, then pushed down (like predict.decide()'s z_echo).
# These are the scopeable profile's DEFAULTS, not tuned values.
GAIN_FLOOR = 0.35   # below this marginal residual, an item adds no new info
MAX_ITEMS = 6       # cap regardless of gain (a budget; per-call `k` overrides it)
DEFAULT_CALIBRATION = {"gain_floor": GAIN_FLOOR, "max_items": MAX_ITEMS,
                       "per_cluster": {}}


def _rows(x) -> list[dict]:
    """Normalise candidates/seed to measure()'s list[dict] form. Accepts an
    (n, d) ndarray (wrapped, embeddings reused) or an already-dict list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        x = np.atleast_2d(x)
        return [{"id": i, "text": str(i), "embedding": x[i]} for i in range(len(x))]
    return list(x)


def assemble(cand, *, seed=None, weights=None, k: int | None = None,
             calibration: dict | None = None, scope: str | None = None,
             embed=None) -> dict:
    """Greedily assemble the most-informative, least-redundant subset via measure().

    cand:    (m, d) ndarray or list[dict] candidates.
    seed:    initial assembly context (e.g. the query) — ndarray or list[dict].
    weights: per-candidate relevance multiplier (info-per-relevance); the pick
             maximises residual*weight but the STOP test is on raw residual alone,
             so an irrelevant-but-novel item can't be padded in.
    k:       per-call budget B; overrides the profile's max_items.
    Returns {chosen, gains, shares, stopped} — `gains` aligns with `chosen`;
    `shares` is each chosen item's residual share (budget split); `stopped` is
    True if it halted on gain_floor, False if it filled k.

    Each step calls measure(remaining, assembly) and reads `residual`. When the
    assembly is below measure()'s warmup (≤1 item) every candidate reads as
    maximally novel (residual 1.0), so the opening pick(s) fall to relevance —
    correct: nothing is yet assembled to be redundant against."""
    gain_floor = calib_value(calibration, "gain_floor", GAIN_FLOOR, scope)
    k = k if k is not None else calib_value(calibration, "max_items", MAX_ITEMS, scope)
    cand = _rows(cand)
    m = len(cand)
    w = np.ones(m) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    assembly = _rows(seed)            # the growing Y

    chosen: list[int] = []
    gains: list[float] = []
    remaining = list(range(m))
    stopped = False
    while remaining and len(chosen) < k:
        ms = measure([cand[i] for i in remaining], assembly, embed=embed)
        g = {i: ms[pos]["residual"] for pos, i in enumerate(remaining)}
        best = max(remaining, key=lambda i: g[i] * w[i])
        if g[best] < gain_floor:
            stopped = True
            break
        chosen.append(best)
        gains.append(round(g[best], 4))
        assembly.append(cand[best])
        remaining.remove(best)

    total = sum(gains) or 1.0
    shares = [round(x / total, 4) for x in gains]
    return {"chosen": chosen, "gains": gains, "shares": shares, "stopped": stopped}


__all__ = ["assemble", "GAIN_FLOOR", "MAX_ITEMS", "DEFAULT_CALIBRATION"]
