# Consolidate / contrastive concept layer — handoff (2026-06-26)

Continuation doc. Goal of the session: fix the consolidate symptoms (too many concepts,
all concepts close, no bridges) using prediction error as the spine.

## TL;DR

- **Concept representation** is the win. Contrastive medoid de-collapses the concept
  layer and lifts retrieval. **Shipped + committed** (`7d32d53`).
- **Concept membership** via geometry is a dead end. Autonomous mint/merge/split
  (squish reassignment, split, EM) all *regress* retrieval. Membership stays
  LLM-owned. **Refuted, scratchpad-only.**
- "Broad ceiling is gated on splitting mega-hubs" — **refuted**. Splitting lowers broad.

## What shipped (committed in `7d32d53`, working tree clean)

1. **Concept vector: centroid → medoid** — `core/store.py:996` `recompute_concept_embedding`.
   The mean dragged every concept toward the global centre (that was "all concepts
   close": pair-cos 0.236, nearest-neighbour 0.63). Medoid is a real member, on-manifold
   → halves it (0.126 / 0.49). Per-event default; deterministic.
2. **Contrastive re-anchor** — `core/consolidate.py` `_reanchor_concepts` (Step 6b,
   end-of-run batch after create/merge/split settle; call site right before
   `_fit_baselines`). Representative = member maximizing `r_nbr − λ·r_own`
   (central to self AND distinct from J nearest concepts' members). Defaults **λ=1.0,
   J=3** (`REANCHOR_LAMBDA`, `REANCHOR_J`).

Validated deltas vs centroid baseline (eval corpus, τ=0.54):

| gold | Coverage@B centroid → medoid → contrastive |
|---|---|
| narrow | 71.4 → 78.6 → 78.6 |
| broad | 33.3 → 33.3 → 33.3 |
| paragraph | 50.0 → 62.5 → **75.0** |

Knobs: λ=0 (pure push) tanks broad (→16.7); λ=2 over-squishes (para→62). Coverage flat
across λ∈[0.5,1.0] = robust. J=3 edges J=5/8 on Reach. Raw residuals are correct for
SELECTION (per-concept argmax vs fixed pools); density z-scoring is only needed if the
margin ever feeds a THRESHOLD (merge/split decisions).

## What was refuted (geometry-only, no-LLM; `scratchpad/contrastive_membership.py`)

- **Squish reassignment** (claim → dominant-nearest cluster): runaway agglomeration,
  rich-get-richer. maxsz 36→148, narrow 79→64.
- **Split-only** (Task #5, `_spread_is_bimodal` poles): broad 33→**17**, narrow 79→71.
  The maxsz-36 concept is *cohesive* (not bimodal) — not a split target. Splitting
  fragments the breadth frame (top-3 concepts carry less) → broad drops.
- **EM (reassign+split)**: split can't arrest squish; maxsz→215, narrow→64. The broad=50
  at pass 4 is a degenerate-collapse artifact, not a real win.

**Lesson:** the contrastive margin is a *representation* operator (which member stands
for a fixed concept), **not a membership** operator. The LLM-built membership is a good
local optimum; meaning-judgment ("same idea vs two ideas") does work geometry can't
(PRD: residual necessary-but-not-sufficient; membership IS the irreversible commit).

## Eval setup (to reproduce / continue)

- Corpus: `/tmp/slate_eval.db`, user `usr_01KTXAYR20J4R6F7PT3DP10W3W`, 2040 claims / 326
  concepts. (Your personal slate is `usr_01KTXFSADRTHGX2EF0T04P1W1E`, 195 concepts — no
  gold there.) Always work on a copy: `cp /tmp/slate_eval.db /tmp/slate_work.db`.
- Env: `.venv/bin/python`, run with `PYTHONPATH=.:scratchpad`.
- Metrics:
  - **Coverage@B** = `eval/coverage.py` `coverage_eval` `sr_at_b` (query passes iff all
    key_facts covered, cosine ≥ τ=0.54). Trustworthy for same-path A/B.
  - **Reach@B** = graded node-selection (is gold fact's nearest claim in the top-N
    surfaced nodes), built in `scratchpad/eval_both.py`. My own proxy; noisier.
- Run combined eval: `PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/eval_both.py /tmp/slate_work.db`
- Gold: `eval/gold.jsonl` (narrow, 20), `eval/gold_broad.jsonl` (11), `eval/gold_paragraph.jsonl` (14).
- Scratchpad scripts: `eval_both.py` (Coverage+Reach), `concept_basis.py` (representative
  strategy sweep), `reanchor_sweep.py` (λ/J sweep), `contrastive_membership.py`
  (refuted membership experiments).

## Open next steps (prioritized)

1. **Run the live recompute on `engine.db`.** Both users still on centroids — shipped
   code only affects future consolidation runs. One-time: for each concept call
   `store.recompute_concept_embedding` then `consolidate._reanchor_concepts(conn, user)`.
   Reversible via event-log rebuild.
2. **Stage 1 — geometry PROPOSES, LLM disposes (untested; needs LLM budget).** The only
   viable membership use of the margin: replace the knn-3 candidate window in
   `_concept_pass_chunk` (`consolidate.py:720`) with contrastive candidates, LLM still
   decides CREATE/ATTACH/MERGE/SPLIT. Targets the fragmentation tail (44 concepts ≤2
   members). Needs a real (small) LLM consolidation run to eval Coverage@B. ~budget note:
   memory says ~$1.9 left; there were past LLM billing incidents — check `core/llm.py`.
3. **Broad (33%) is NOT a concept-geometry problem.** Neither representation nor
   membership moved it across every experiment. It's the R0 vocabulary-disjoint seed
   problem (see `docs/retrieve-resonance-design.md` "Open risks"). Lever is query-side
   (HyDE / query-aware embedding) or usage edges from retrieval signals — not concepts.
4. **Density-thresholded margin** — if Stage 1 works and you want geometry to gate
   merge/split, z-score `r_nbr`/`r_own` against the C12 `compute_baselines` μ/σ (same
   density Write uses) so the bar scales with cluster tightness.

## Memory pointers
- `[[slate-consolidate-contrastive]]` — these findings.
- `[[slate-resonance-retrieval]]` — retrieval path; its "split mega-hubs" claim is now
  corrected.
