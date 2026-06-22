"""Iterative residual-vs-assembly — wrapper #2 over the predictor (plan §0).

A THIN ITERATOR over the predictor, like scan and guard. The spine is the residual
of X against Y; assembly is the choice of X, Y and reading the **residual** — but
unlike the others it iterates: Y GROWS one item per step, so it is a loop, not a
single measurement.

  X = the remaining candidates.
  Y = the seed (e.g. the query) + everything chosen so far — the growing assembly.
  read = `residual` (a candidate's unique contribution beyond what Y already holds).

It reads ONLY the residual (never z/baselines/clusters/peers), so it calls
`predict.residuals_against` — the batch residual primitive — instead of the full
`measure()`. That skips the per-STEP baseline recompute (a Gram + LOO over Y) and
the per-candidate peer/cluster geometry measure() would compute for fields assembly
never touches; the residual numerics are identical.

Greedily add the candidate of highest residual × relevance; STOP when the best
marginal residual falls below `gain_floor` — the assembly has saturated and more
items are padding. Serves Retrieve.Stop/Allocate (pull a thread only while it
deepens the answer; split budget by residual share) and Consolidate.Dedupe (a
near-duplicate has residual≈0 against what's chosen, so it is never picked).

The default `gain_floor` STOP reads RAW residual — fine when Y is a mature pool.
But in Retrieve Y grows from a tiny query seed, so residual stays high even for a
tangential pick and the floor never fires (over-injection — see
`docs/retrieve-workflow-eval.md`). The optional `value_floor` knob (R2/R7) instead
gates the STOP on marginal **value** = residual × relevance, the same quantity the
PICK maximises, so a novel-but-irrelevant span can't pad. Off by default (preserves
the raw-residual semantics every existing caller relies on); fitted against SR@B and
pushed down by C12 — never pre-tuned.

Closed-loop: the pick decision changes Y, which changes the next measurement —
so it cannot collapse to one open-loop measure()/decide() pair (that is why it
loops). The STOP/budget policy comes from the scopeable calibration profile.
"""
from __future__ import annotations

import numpy as np

from core.predict import (WARMUP_MIN_CORPUS, _default_embed, calib_value,
                           residuals_against)

# ── POLICY lives in a calibration PROFILE, not in constants ───────────────────
# Both encode the compression/budget BET the PRD assigns to calibration owned and
# FITTED at consolidation, then pushed down (like predict.decide()'s z_echo).
# These are the scopeable profile's DEFAULTS, not tuned values.
GAIN_FLOOR = 0.35   # below this marginal residual, an item adds no new info
MAX_ITEMS = 6       # cap regardless of gain (a budget; per-call `k` overrides it)
VALUE_FLOOR = None  # off by default; when set, STOP on residual×relevance (R2/R7)
DEFAULT_CALIBRATION = {"gain_floor": GAIN_FLOOR, "max_items": MAX_ITEMS,
                       "value_floor": VALUE_FLOOR, "per_cluster": {}}


def _rows(x) -> list[dict]:
    """Normalise candidates/seed to measure()'s list[dict] form. Accepts an
    (n, d) ndarray (wrapped, embeddings reused) or an already-dict list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        x = np.atleast_2d(x)
        return [{"id": i, "text": str(i), "embedding": x[i]} for i in range(len(x))]
    return list(x)


def _embeddings(rows: list[dict], embed) -> np.ndarray | None:
    """Stack row embeddings into an (n, d) matrix, embedding any text-only rows via
    `embed` (mirrors measure()'s missing-embedding fill). None for an empty list."""
    if not rows:
        return None
    missing = [r for r in rows if r.get("embedding") is None]
    if missing:
        for r, v in zip(missing, embed([r["text"] for r in missing])):
            r["embedding"] = v
    return np.vstack([np.asarray(r["embedding"], dtype=float) for r in rows])


def assemble(cand, *, seed=None, weights=None, k: int | None = None,
             calibration: dict | None = None, scope: str | None = None,
             embed=None) -> dict:
    """Greedily assemble the most-informative, least-redundant subset via measure().

    cand:    (m, d) ndarray or list[dict] candidates.
    seed:    initial assembly context (e.g. the query) — ndarray or list[dict].
    weights: per-candidate relevance multiplier (info-per-relevance); the pick
             always maximises residual*weight. By default the STOP test is on raw
             residual (gain_floor); set `value_floor` in the profile to STOP on
             residual*weight instead, so an irrelevant-but-novel item can't pad.
    k:       per-call budget B; overrides the profile's max_items.
    Returns {chosen, gains, values, shares, stopped} — `gains` (raw residual) and
    `values` (residual×relevance) align with `chosen`; `shares` is each chosen
    item's residual share (budget split); `stopped` is True if it halted on a
    floor, False if it filled k.

    Each step computes every remaining candidate's residual against the assembly
    via `predict.residuals_against` (identical numerics to measure()'s `residual`
    field, without its baseline/z/peer work). When the assembly is below the warmup
    (<2 items) every candidate reads as maximally novel (residual 1.0), so the
    opening pick(s) fall to relevance — nothing is yet assembled to redundate."""
    gain_floor = calib_value(calibration, "gain_floor", GAIN_FLOOR, scope)
    value_floor = calib_value(calibration, "value_floor", VALUE_FLOOR, scope)
    k = k if k is not None else calib_value(calibration, "max_items", MAX_ITEMS, scope)
    cand = _rows(cand)
    m = len(cand)
    w = np.ones(m) if weights is None else np.clip(np.asarray(weights, float), 0, None)
    embed = embed or _default_embed
    Xe = _embeddings(cand, embed)             # (m, d) candidate embeddings, once
    A = _embeddings(_rows(seed), embed)        # the growing Y (seed first)
    if A is None:
        A = np.empty((0, Xe.shape[1]) if Xe is not None else (0, 0), dtype=float)

    chosen: list[int] = []
    gains: list[float] = []
    values: list[float] = []
    remaining = list(range(m))
    stopped = False
    while remaining and len(chosen) < k:
        ridx = np.fromiter(remaining, dtype=int, count=len(remaining))
        if A.shape[0] < WARMUP_MIN_CORPUS:
            # Y too small to reconstruct against — every candidate reads maximally
            # novel (residual 1.0), mirroring measure()'s warmup, so the opening
            # pick(s) fall to relevance: nothing is yet assembled to redundate.
            g = np.ones(len(remaining))
        else:
            g = residuals_against(Xe[ridx], A)
        gw = g * w[ridx]
        # max over remaining in order (argmax → first max), matching the previous
        # `max(remaining, key=...)` tie-break exactly.
        pos = int(np.argmax(gw))
        best = remaining[pos]
        gbest = float(g[pos])
        # STOP on raw residual (saturation) OR — when calibrated — on marginal value
        # (residual×relevance), which catches the high-residual-but-tangential pad.
        if gbest < gain_floor or (value_floor is not None and gw[pos] < value_floor):
            stopped = True
            break
        chosen.append(best)
        gains.append(round(gbest, 4))
        values.append(round(gbest * float(w[best]), 4))
        A = np.vstack([A, Xe[best]])
        remaining.remove(best)

    total = sum(gains) or 1.0
    shares = [round(x / total, 4) for x in gains]
    return {"chosen": chosen, "gains": gains, "values": values, "shares": shares,
            "stopped": stopped}


__all__ = ["assemble", "GAIN_FLOOR", "MAX_ITEMS", "VALUE_FLOOR", "DEFAULT_CALIBRATION"]
