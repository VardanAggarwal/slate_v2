"""C12 — calibration persistence + push-down (plan §3 C12; PRD §"Two layers").

Consolidation OWNS the calibration — the compression/budget BET over (Q, B) — and
pushes it outward: down to Write, up to Retrieve. The predictor (`measure`) is a
pure sensor and never sees it; only the decision-layer thresholds move. Swapping
the embedding model re-fits this profile and leaves the sensor's contract intact.

This module is the persistence + load path. The fit loop that PRODUCES a profile
is `eval/fit_stop.py` (SR@B-gated — the only credit); here we store the result and
merge it over each stage's in-code `DEFAULT_CALIBRATION` at call time, so a fitted
value (e.g. the relevance-aware `value_floor`) actually reaches retrieve/write.

The profile lives in the DB (`calibration_profiles`, per user) — scoped to the
corpus it was fitted on, and never confused with another user's.
"""
from __future__ import annotations

from core import store


def merged(conn, defaults: dict, user_id: str) -> dict:
    """Stage `defaults` overlaid with the user's persisted profile (persisted
    wins). Returns a fresh dict; `defaults` is not mutated. When nothing has been
    fitted yet this is just `defaults` — so the default path is unchanged."""
    out = dict(defaults)
    out.update(store.get_calibration(conn, user_id))
    return out


def push(conn, user_id: str, **keys) -> dict:
    """Update individual profile keys (the fit's push-down). None values are
    ignored so `push(value_floor=None)` is a no-op, not a reset. Returns the new
    profile."""
    prof = store.get_calibration(conn, user_id)
    prof.update({k: v for k, v in keys.items() if v is not None})
    store.set_calibration(conn, user_id, prof)
    return prof


__all__ = ["merged", "push"]
