"""Reconstruction guard — wrapper #3 over the predictor (execution plan §0).

A THIN ITERATOR over `predict.measure()`, like scan — not re-derived geometry.
The spine is `measure(X, Y)`; guard is the choice of X, Y and reading the **z**
field (residual scored against the region's OWN spread — the spread-relative rule
the whole engine turns on: "how tightly the region already coheres sets the bar").

Guard answers: before folding/pruning a claim, can what REMAINS rebuild it? Low
residual ⇒ redundant ⇒ safe to drop; high ⇒ irreplaceable ⇒ protect, however old.
Two entrypoints, same primitive, X and Y swapped:

  forget(members)            X = Y = a cluster's members, leave-one-out
                             (exclude_self). Each member vs the rest of its
                             cluster; z is vs the cluster's own cohesion. Serves
                             Consolidate.Forget — prune the rebuildable, keep the
                             atypical.
  merge(losers, survivors)   X = the absorbed group, Y = the surviving group.
                             Each loser vs the survivors; z is vs the survivors'
                             cohesion. Serves Consolidate.Merge — fold only the
                             members the survivors already cover, keep the rest as
                             their own claim instead of averaging them away.

The keep/forget bet is z ≤ z_forget, read from the scopeable calibration profile
(fitted at consolidation, pushed down — like predict.decide()'s z_echo). A
cold_start measurement (too few peers to estimate spread) is never dropped.
"""
from __future__ import annotations

import numpy as np

from core.predict import calib_value, measure, residual_against

# ── POLICY: calibration-profile defaults (scopeable; fitted at consolidation) ──
Z_FORGET = -0.5         # z this far below the region mean ⇒ unusually rebuildable
RESIDUAL_FLOOR = 0.35   # absolute fallback when no spread/z is available
DEFAULT_CALIBRATION = {"z_forget": Z_FORGET, "residual_floor": RESIDUAL_FLOOR,
                       "per_cluster": {}}


def _verdict(m: dict, calibration: dict | None, scope: str | None) -> dict:
    """Apply the keep/forget bet to one measurement. Prefer the spread-relative z
    gate; fall back to the absolute residual floor only when there is no z (too
    few peers). A cold_start probe is never dropped — you cannot safely forget
    what you could not measure."""
    if m["cold_start"]:
        drop = False
    elif m.get("z") is not None:
        drop = m["z"] <= calib_value(calibration, "z_forget", Z_FORGET, scope)
    else:
        drop = m["residual"] <= calib_value(calibration, "residual_floor",
                                            RESIDUAL_FLOOR, scope)
    return {**m, "safe_to_drop": drop}


def forget(members: list[dict], *, baselines: dict | None = None,
           calibration: dict | None = None, scope: str | None = None,
           embed=None) -> list[dict]:
    """Per-member safe-forget verdicts for one cluster. `members` are memory rows
    {"id","text","embedding"(opt),"cluster"(opt)}. Each is measured leave-one-out
    against the rest; returns each measurement plus `safe_to_drop`."""
    ms = measure(members, members, exclude_self=True, baselines=baselines,
                 embed=embed)
    return [_verdict(m, calibration, scope) for m in ms]


def merge(losers: list[dict], survivors: list[dict], *, baselines: dict | None = None,
          calibration: dict | None = None, scope: str | None = None,
          embed=None) -> list[dict]:
    """Per-loser fold verdicts for a MERGE: which absorbed members the survivors
    can rebuild (`safe_to_drop=True`, fold) vs which carry nuance the survivors
    lack (keep as their own claim). One result dict per row of `losers`."""
    ms = measure(losers, survivors, baselines=baselines, embed=embed)
    return [_verdict(m, calibration, scope) for m in ms]


# ── low-level convenience: residual against a raw vector pool (direct primitive
#    use, for callers holding numpy vectors rather than memory-row dicts) ───────
def reconstruction_residual(victim: np.ndarray, remaining: np.ndarray) -> float:
    """How much of `victim` the `remaining` set CANNOT rebuild (0 = fully
    reconstructable/redundant; 1 = orthogonal/irreplaceable). Empty → 1.0."""
    remaining = np.asarray(remaining, dtype=float)
    if remaining.shape[0] == 0:
        return 1.0
    return residual_against(np.asarray(victim, dtype=float), remaining)


__all__ = ["forget", "merge", "reconstruction_residual",
           "Z_FORGET", "RESIDUAL_FLOOR", "DEFAULT_CALIBRATION"]
