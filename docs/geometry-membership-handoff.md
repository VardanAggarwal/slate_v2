# Handoff — geometry vs LLM membership (close-out 2026-06-26)

Pick-up doc for the "can geometry replace LLM membership?" thread. Evidence + full tables
live in **`docs/geometry-vs-llm-membership-findings.md`**; this is the short state + next move.

## Where it landed (TL;DR)
Geometry can **win individual axes but not dominate** the LLM. Best fully-geometric pipeline
= **density anchors + reconstruction-z membership + redundancy → 71/50/62** (narrow/broad/para)
vs LLM **79/33/75**: beats broad, loses narrow + paragraph. The narrow/para ceiling (~71/62)
is the **embedding's meaning limit** — negation/number/scope the encoder flattens — not an
algorithm problem. So this thread is closed: **the lever is now the signal (features/usage),
not the membership algorithm.**

## Decided (stop re-testing these)
- **Reconstruction-z > proximity.** "Can the region rebuild it?" (PRD primitive) gives narrow
  57→71 vs k-means/VQ. Keep this as the membership primitive.
- **Anchors must be density-based, not k-center.** Outlier-seeking anchors tank broad to 0.
- **Redundancy / multi-membership works** (broad 17→50) *with density anchors*. Real lever.
- **Distortion-minimizing geometry can't beat the LLM** (rate–distortion proof: LLM is
  RD-suboptimal yet retrieval-superior). k-means/VQ/squish/EM are dead ends.
- **Bridges are inert** (traversed but ranked below the top-12 cutoff). NOTE 2026-06-27:
  the suppressor is the **distinctiveness term** (`1/(1+ln degree)^γ`), NOT the PE-gate —
  see `docs/consolidation-strategy-team-findings.md`. Tuning retrieval to value bridges
  (boost↑/gate-off/wider materialize) makes broad strictly worse.
- **Geometric retrieval hops regress** narrow; **no intrinsic geometric KPI** (silhouette, DB,
  masked-recon, rate–distortion) tracks Coverage@B — all ≈0 or anti-aligned.
- **Broad is a 6-query metric** — expand `gold_broad` before optimizing it further; every
  "broad +N" so far is 1–2 queries.

## Open / next (prioritized)
1. **Richer features (highest leverage).** Re-run the *current* best pipeline on a stronger /
   query-aware / asymmetric / LLM-derived embedding (one-time featurizer — still geometry).
   Measure how much of the 71→79 / 62→75 ceiling better features close. This is the only lever
   that can lift narrow + paragraph.
2. **Usage geometry (Hebbian).** Co-retrieval / co-activation edges — the one signal all
   session that tracked SR@B; PRD treats usage as a separate input to consolidation. Bootstraps
   from traffic; injects meaning the embedding misses.
3. **Expand `gold_broad`** (currently 6 scored) so broad results stop being coin-flips. Prereq
   for any broad work, incl. the query-side HyDE/decomposition path (additive, can't hurt narrow).
4. **(Optional) Productionize the geometric pipeline** as the *proposer* the PRD intends —
   reconstruction-z multi-attach + density anchors, with the irreversible same-vs-contradiction
   commit still on the resolver. It won't replace the LLM but it's a strong candidate generator.

## Current best geometric recipe (for whoever rebuilds it)
1. **Anchors** = K density medoids (spherical k-means → nearest claim to each centroid). NOT
   farthest-first.
2. **Membership** = reconstruction-z: per claim, residual against each candidate region's
   members, z-scored to that region's leave-one-out spread. Primary = min residual.
3. **Redundancy** = also attach to every other candidate region with z ≤ θ (θ≈1.0). Selective.
4. **Centers** = medoid (+ contrastive re-anchor) on **primary members only** — redundant rows
   must not corrupt the concept vector.
Implemented in `scratchpad/recon_membership.py` + `recon_density.py`.

## Env / reproduce
- Corpus `/tmp/slate_eval.db`, user `usr_01KTXAYR20J4R6F7PT3DP10W3W` (2040 claims / 326 concepts).
  Always `cp /tmp/slate_eval.db /tmp/slate_work.db` first.
- `PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/<script>`; Coverage@B τ=0.54.
- LLM bar = **79/33/75**. Golds: narrow 20, broad **6 scored**, paragraph 14.
- 11 harness scripts listed at the bottom of the findings doc.

## Pointers
- Evidence + all tables: `docs/geometry-vs-llm-membership-findings.md`
- Prior context: `docs/consolidate-contrastive-handoff.md`, `docs/retrieve-resonance-design.md`
- PRD: `docs/Slate PRD v2.md` (§How — membership = reconstruction residual + resolver commit)
- Memory: `[[slate-geometry-vs-llm-membership]]`, `[[slate-resonance-retrieval]]`,
  `[[slate-consolidate-contrastive]]`
