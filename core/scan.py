"""Sequential scan — wrapper #1 over the predictor (execution plan §0).

This is a THIN ITERATOR over `predict.measure()`, not a re-implementation of any
geometry. The spine is `measure(X, Y)`; scan is just the choice of X and Y and
which field of the measurement to read:

  X = each sentence of the note, in order.
  Y = the note's OWN sentences seen so far (the causal prefix) — "send the note's
      sentences as both the text and the corpus." A boundary is where a sentence
      stops matching what came before it, so Y must be the prefix, not the whole
      note (a topic with ≥2 sentences would otherwise self-match and hide its own
      start). Each step is one `measure([sentence], prefix)` call.
  read = `nearest_sim` — the cosine-to-nearest field measure() already returns.
      Reconstruction *residual* is a weak boundary signal at sentence scale
      (loosely-coherent topics reconstruct poorly regardless), but nearest_sim is
      exactly "did this fragment match its context"; it is the right read of the
      same measurement, no separate operator.

The cut is SPREAD-RELATIVE, the same principle as predict's z_echo: a sentence is
a boundary when its nearest_sim drops anomalously LOW against the note's OWN
nearest_sim spread (robust median/MAD z). A tightly-held note (high, even
nearest_sim) makes a single dip stand out; a loosely-held note tolerates wider
swings before cutting. How tightly the note coheres sets its own boundary bar —
nothing is hard-coded per corpus.

Pure and lens-only: ordered sentences in, boundary indices out. The only LLM/DB
touch is the embedder measure() already uses.
"""
from __future__ import annotations

import numpy as np

from core.predict import calib_value, measure

# ── POLICY lives in a calibration PROFILE, not in constants ───────────────────
# DROP_Z is the boundary bet (PRD: calibration owned + FITTED at consolidation,
# pushed down — like decide()'s z_echo). A scopeable-profile DEFAULT, not tuned.
# A sentence opens a new segment when its nearest_sim z-score (vs the note's own
# spread) falls at or below DROP_Z — i.e. it is unusually unrelated to its prefix.
DROP_Z = -1.5
# FOLD_Z is the variable-RESOLUTION bet (W3): WITHIN a segment, how finely to
# encode. A sentence opens its own FINE fragment when its nearest_sim z dips at or
# below FOLD_Z — surprising enough to deserve standalone attention; quieter (more
# predictable) sentences FOLD into the running fragment. |FOLD_Z| < |DROP_Z|, so it
# is a gentler cut than a topic boundary: a boundary always splits, and between
# boundaries the surprising sentences still split while the predictable ones merge.
# Same novelty signal as the boundary cut, a second threshold on it. A profile
# DEFAULT, fitted at consolidation and pushed down — not a tuned constant.
FOLD_Z = -0.75
# Encoder-noise floor on the nearest_sim spread: a perfectly tight note has
# MAD→0, where any micro-jitter would explode into a huge z and manufacture a
# false cut. You cannot cohere tighter than the embedder's own noise, so the
# spread used for the z-score is floored here. INSTRUMENT precision, not a bet.
SIGMA_FLOOR = 0.06
DEFAULT_CALIBRATION = {"drop_z": DROP_Z, "fold_z": FOLD_Z, "per_cluster": {}}


def _nearest_sims(E: np.ndarray) -> list[float | None]:
    """nearest_sim of each sentence against its causal prefix, via measure().
    Sentence 0 (no prefix) and any sentence whose prefix is below measure()'s
    warmup floor return None (segment openers — no boundary test applies)."""
    items = [{"id": i, "text": str(i), "embedding": E[i]} for i in range(len(E))]
    out: list[float | None] = [None]
    for i in range(1, len(items)):
        # X = this sentence, Y = the prefix; embeddings reused (no re-embed).
        m = measure([items[i]], items[:i])[0]
        out.append(None if m["cold_start"] else m["nearest_sim"])
    return out


def residual_curve(E: np.ndarray) -> list[float]:
    """The causal nearest_sim per sentence (None→0.0). Named for symmetry with
    the other wrappers; this is a similarity curve, not a residual one."""
    return [0.0 if s is None else round(s, 4) for s in _nearest_sims(np.asarray(E, dtype=float))]


def _spread(sims: list[float | None]) -> tuple[float, float] | None:
    """Robust centre/scale (median, MAD→σ) of the causal nearest_sim curve — the
    note's OWN spread, against which a dip is judged an outlier. None when there
    are too few measured sentences to estimate it. σ floored at SIGMA_FLOOR."""
    vals = np.array([s for s in sims if s is not None])
    if vals.size < 2:
        return None
    med = float(np.median(vals))
    sigma = max(SIGMA_FLOOR, 1.4826 * float(np.median(np.abs(vals - med))))
    return med, sigma


def boundaries(E: np.ndarray, *, calibration: dict | None = None,
               scope: str | None = None) -> list[int]:
    """Indices where a new segment STARTS (cut points), excluding 0. A cut is a
    spread-relative LOW outlier in the causal nearest_sim curve."""
    E = np.asarray(E, dtype=float)
    drop_z = calib_value(calibration, "drop_z", DROP_Z, scope)
    sims = _nearest_sims(E)
    spread = _spread(sims)
    if spread is None:
        return []
    med, sigma = spread
    return [i for i, s in enumerate(sims)
            if s is not None and (s - med) / sigma <= drop_z]


def segment(E: np.ndarray, *, calibration: dict | None = None,
            scope: str | None = None) -> list[list[int]]:
    """Group sentence indices into contiguous segments split at `boundaries`."""
    E = np.asarray(E, dtype=float)
    cuts = boundaries(E, calibration=calibration, scope=scope)
    out, start = [], 0
    for b in cuts + [len(E)]:
        out.append(list(range(start, b)))
        start = b
    return out


def fragments(E: np.ndarray, *, calibration: dict | None = None,
              scope: str | None = None) -> list[list[int]]:
    """Variable-resolution fragments (W3): not just WHERE to cut but HOW FINELY —
    fine where surprise is high, folded where low (PRD: "Variable-resolution
    encoding"). Same causal nearest_sim signal as `boundaries`, read at a gentler
    FOLD_Z threshold: a topic boundary (z ≤ drop_z) always splits; BETWEEN
    boundaries a sentence still opens its own fragment when it is surprising
    (z ≤ fold_z), while predictable sentences FOLD into the running fragment.

    So a high-surprise region stays fine (many small fragments) and a low-surprise
    region collapses into few coarse ones — the granularity tracks the novelty,
    not a uniform sentence split. Returns contiguous index groups partitioning
    range(len(E)). With < 2 measurable sentences, the whole note is one fragment."""
    E = np.asarray(E, dtype=float)
    drop_z = calib_value(calibration, "drop_z", DROP_Z, scope)
    fold_z = calib_value(calibration, "fold_z", FOLD_Z, scope)
    cut_z = max(drop_z, fold_z)        # gentler of the two splits within a segment
    sims = _nearest_sims(E)
    spread = _spread(sims)
    if spread is None:
        return [list(range(len(E)))]
    med, sigma = spread
    frags, cur = [], [0]
    for i in range(1, len(E)):
        s = sims[i]
        if s is not None and (s - med) / sigma <= cut_z:
            frags.append(cur)          # surprising → start a new fine fragment
            cur = [i]
        else:
            cur.append(i)              # predictable (or unmeasured) → fold in
    frags.append(cur)
    return frags


__all__ = ["residual_curve", "boundaries", "segment", "fragments",
           "DROP_Z", "FOLD_Z", "DEFAULT_CALIBRATION"]
