# Broad-lift via user-traversal signal — findings

**Date:** 2026-07-08. **Implements:** `docs/broad-lift-traversal-plan.md`.
**Verdict: no lane ships.** Even an ORACLE demand signal (walks built from the gold
facts themselves) cannot move broad Coverage@B through any of the three lanes. The
demand-side traversal signal is now captured (Phase 0 ships, log-only); its
consumption stays default OFF. One real bug was found and fixed on the way
(qclm dead-slot displacement — affects shipped 6d).

## Setup

- Corpus: `/tmp/slate_eval.db` rebuilt from `data/engine.db` (166 notes, 2 040
  claims, 326 concepts) + `write.refine_pending` → 969 fragments. User
  `usr_01KTXAYR20J4R6F7PT3DP10W3W`.
- Base per config copy: cleared stale prod relations, medoid recompute (6a),
  contrastive re-anchor (6b), channel redundancy (6c), plus the engagement log.
- Engagement log (eval-only, plan §0.2): one ideal walk per `gold_broad` probe
  (key_fact → nearest concept, in fact order) + the one real trace
  (Compromise → Identity-Erosion → Society-vs-Individuality → Identity-Preservation).
  19 synthetic + 1 real = 20 events. This is the ORACLE upper bound: anything a
  lane can't lift given this signal, it won't lift with real usage.
- Metric: deterministic `eval/coverage.py` (τ=0.54, B=2000 tok) over
  `gold.jsonl` (16q narrow) / `gold_broad.jsonl` (19q, see below) /
  `gold_paragraph.jsonl` (9q), production `resonance_context`.
- Gold: **+6 new broad probes** (tension / lineage / cross-scale / assembly /
  meta-recurrence / bridge), key_facts near-verbatim from the source notes.
  All 6 FAIL at baseline — they probe exactly the under-tested shapes.

## Ablation matrix (extended gold, 19 broad probes)

| Config | narrow | broad | paragraph | gate |
|---|---|---|---|---|
| baseline (6a+6b+6c) | .750 | **.210** | .889 | ref |
| +A Lane2, reanchor OFF | .750 | .210 | .889 | kill (inert) |
| +A Lane2, reanchor ON | .750 | .158 | .667 | **kill (regressive)** |
| +B Lane1 walked edges | .750 | .210 | .778 | kill |
| +B + degree-exempt | .750 | .210 | .778 | kill |
| +C arc nodes (template) | .812 | .210 | .889 | kill (broad inert; narrow +1q) |
| +C arc nodes (LLM) | .750 | .158 | .889 | kill |
| +C + arc frame slot | .750 | .158 | 1.000 | kill (paragraph +1q, broad −1q) |
| +D salience | .750 | .210 | .889 | kill (inert, by construction) |
| +E calibration fit | — | — | — | kill (value_floor flat; keep hand floor) |
| combo A(off)+B(ex)+C | .812 | .158 | 1.000 | kill |

Single-query granularity: broad ±.053, paragraph ±.111. Every off-baseline delta
above is ±1 query and the flipped queries straddle τ (e.g. p06 .658→.451,
p12 .536→.570) — boundary noise under perturbation, not mechanism.

## What actually happened, per lane

- **A (walked-path query-claims).** Routing alone (reanchor OFF) is EXACTLY
  neutral — the walked concepts do get bridged, but their activation lands below
  the materialization window, so nothing new is read. Reanchor ON is the only
  thing that moves broad and it moves it DOWN here (−1q) while destroying
  paragraph (−2q): dragging ~50 concept medoids toward question-shapes breaks
  the content geometry paragraph queries need. Per the plan's kill criteria,
  **6d reanchor must stay OFF**.
- **B (cross-episode Hebbian edges).** 53 walked edges, weights 0.5–1.0. Inert
  on broad even though these are genuinely cross-episode, demand-validated
  edges — confirming Lane 1 dead *even with real traversal* (6th consecutive
  "edges inert" result). Degree-exempt sub-variant changes nothing (log1p degree
  shift too small to reorder salience). The −1 paragraph query is τ-boundary
  noise.
- **C (arc nodes).** 18 arcs minted (both template and Opus/sonnet summaries
  tested; LLM cost ~$0.03). Arcs ARE lit by broad queries — but rank 13–65 of
  ~200, below `MATERIALIZE_NODES=12`; the direct-knn incumbents always outrank
  the abstraction. A reserved frame slot (`res_arc_frame`, new flag) makes an
  arc actually spend budget: paragraph → 100%, but broad still unmoved — the
  facts the probes need live in per-note fragments, and an arc's frame text
  (member claims) doesn't clear τ for 4 facts across 4 notes within B.
- **D (engagement salience).** Correctly inert on Coverage@B *by construction*:
  `background` is only read by the legacy `recall.py` display path, never by
  resonance/fragments. Shipped as the C13 promote unlock (engaged node = the
  explicit 'needed' vote), felt-quality only.
- **E (calibration fit).** `value_floor` swept none/.10–.30 against the
  engagement-derived gold: completely flat on all four gold sets — the stop
  never binds on this corpus (gain floor / budget stop first). Keep hand floor.

## The one real find: qclm dead-slot displacement (FIXED)

With query-claims present, a live query similar to a stored one puts the `qclm_`
node at the TOP of the salience ranking (cos≈1, salience 1.59); it holds a
`MATERIALIZE_NODES` slot but has no episodes, silently displacing a real claim
from the window. Measured −2 broad queries. This affects **shipped 6d** whenever
a stored query resembles a live one. Fix: `resonance_recall` now excludes
`qclm_` nodes from the materialization slice (they still route activation) —
`core/resonance.py`. After the fix, A(off) went from −2q broad to exactly 0.

## Post-matrix: the 2B run (B=4000)

Re-running the matrix at DOUBLE budget separates the two suspects (selection caps
vs render budget) and changes one verdict:

| Config @2B | narrow | broad | paragraph |
|---|---|---|---|
| baseline | .750 | .210 | .889 |
| +B degree-exempt | .750 | **.263** | .778 |
| +C frame slot | .812 | .210 | 1.000 |
| combo A(off)+B(ex)+C(frame) | .750 | **.263** | .889 |

- **Selection caps, not budget, are the wall.** The materializer emits ~1 790 tok
  under B=2000 and only ~2 950 under B=4000 — it never fills either budget, and
  baseline broad is identical at both. The binding constraints are
  `MATERIALIZE_NODES`/frag-caps/assembly stop, upstream of the char budget.
- **Walked edges DO bridge real content — given slack.** The +1 broad query
  (b_capitalism) is a genuine recovery, not τ-wobble: fact 4 goes 0.31 → 0.845
  (fact 3: .648 → .815) — the walked edge pulled the labour-absorption note into
  materialization. At B=2000 the same config instead DISPLACES (b_community fact
  4: .804 → .445): with no slack, bridged content steals slots from incumbents.
- **combo passes the ship gate at 2B** (Δbroad +.053, Δnarrow 0, Δparagraph 0)
  and fails it at the pre-registered primary budget. n=+1 query — do not flip
  defaults on this alone; if pursued, confirm combo@2B under the Opus judge
  first. Results: `scratchpad/traversal_lab_results_2B.json`.

## Post-matrix: the selection-cap sweep (res_materialize_nodes × budget)

Sweeping the cap N∈{12,16,20,24} for baseline and combo at both budgets
(`scratchpad/matnodes_sweep.py`, results `scratchpad/matnodes_sweep.json`):

| Config @B=4000 | narrow | broad | paragraph | broad ctx (tok) |
|---|---|---|---|---|
| baseline N=12 | .750 | .210 | .889 | 2 917 |
| baseline N=16 | .750 | .263 | .889 | 3 178 |
| **baseline N=20** | **.812** | **.263** | **.889** | 3 311 |
| baseline N=24 | .812 | .263 | .889 | 3 396 |
| combo N=12–24 | .750 | .263 | .889 | ~3 200 |

At B=2000 every cell is flat (ctx pinned ~1 790 regardless of N — the render/
partition path saturates first; a separate bottleneck).

- **The combo's whole 2B win is subsumed by the cap.** Baseline N=20 recovers the
  IDENTICAL b_capitalism facts (0.31→0.845) the walked edges recovered — the
  bridged note sat at rank 13–20 all along; the rank cutoff was the only thing
  excluding it. Plus a narrow gain (g04, .489→.584) combo never delivers.
- **The old over-injection fear does not materialize.** N=24 admits no noise the
  assembly gain-floor/relevance weights don't filter (paragraph/narrow never
  drop; ctx grows only ~500 tok). The 2020-era harm came from padding with raw
  low-relevance candidates; a wider SALIENCE-ranked window is a different pool.
- **baseline N=20 @2B strictly dominates combo @2B on every column** → no
  traversal lane is needed for the gain the traversal experiment surfaced.

**Recommendation:** the broad lever is `res_materialize_nodes` (fit via the
calibration profile — safe: neutral at B=2000, dominant at B=4000; forgetting
gate clean at both) plus the render-path saturation fix (why ctx never fills B
even at N=24). Confirm under the Opus judge before pushing to prod. No traversal
consumption flag should be flipped ON.

## Post-matrix: full strategy comparison at N=20 (`scratchpad/matrix_n20.json`)

At res_materialize_nodes=20, B=4000 (baseline .812/.263/.889): **every strategy
converges to broad .263** — broad is entirely cap/budget-driven; the lanes only
redistribute narrow/paragraph, mostly by displacement. A_off/D identical to
baseline (inert); B −1 para −1 narrow (its bridge now subsumed, only the
displacement remains); C −1 narrow; A_on still destructive (−1 narrow, −2 para);
combo dragged down by B. **C_frame is the sole strict Pareto improvement:
.812/.263/1.000** (+1 paragraph, nothing else moves) — but it fails the ship
gate as written (Δbroad>0) and at B=2000 costs a broad query, so it's a
"only if standardizing on B=4000, after judge confirm" candidate. At B=2000
nothing beats baseline. Ranking: C_frame > baseline = A_off = D > B_exempt >
C ≈ combo > A_on.

## Interpretation

The broad wall is not reachability — it is the **materialization bottleneck**
(and specifically its SELECTION caps — the 2B run shows the budget itself is
never filled).
Every lane succeeded at its own mechanism (queries bridge, edges exist, arcs
get lit) and none of it matters because the top-12 window + per-node fragment
caps are spent on direct-knn incumbents, and coverage needs ≥4 facts from ≥4
notes inside B. Supply-side structure (5 prior results) AND demand-validated
supply (this result) are both inert; the reanchor — the only broad mover ever
measured (Opus judge, 6d) — fails the forgetting gate under deterministic
coverage. If broad is to move, the lever is the **materializer's budget policy**
(how many notes get read, and what earns a slot), not more graph structure.

## Shipped (all default OFF unless noted)

- Phase 0: `ENGAGEMENT` event + `retrieve.record_engagement` (path normalised to
  concepts) — **ON** (log-only recorder; host decides when to call).
- `eval/engagement_gen.py` — synthetic/real-trace walk generator + engagement→gold
  converter (E).
- Workstream A: 6d reads ENGAGEMENT walks (`ENGAGEMENT_QUERY_CLAIMS=False`).
- Workstream B: step 6f `TRAVERSAL_EDGES` + `store.set_traversal_edges`
  (`TRAVERSAL_EDGES=False`); `res_walked_edge_exempt` flag in resonance.
- Workstream C: step 6g `ARC_SYNTHESIZED` + `store.clear_arc_concepts` /
  `set_concept_vector`, deterministic `arc_concept_id_for`, LLM summary with
  template fallback (`ARC_SYNTHESIS=False`); `res_arc_frame` flag.
- Workstream D: engagement votes in `_relevance_net` (`ENGAGEMENT_SALIENCE=False`).
- Bug fix (**ON**): qclm exclusion from materialization slots.
- Gold: 6 new broad probes appended to `eval/gold_broad.jsonl`.
- Tests: `tests/test_traversal.py` (9) — recorder normalisation, walked-path
  attach, replace-whole/idempotent/rebuild-safe appliers, degree exemption,
  D promote gating.

Lab: `scratchpad/traversal_lab.py` (configs baseline/A_off/A_on/B/B_exempt/C/
C_frame/D/combo; results in `scratchpad/traversal_lab_results.json`).

Judge confirmation was skipped: the plan reserves it for winners, and there were
none. Caveat: 6d's broad gain was judge-only (deterministic coverage floored) —
if a future lane looks near-miss under coverage, re-check under the Opus judge
before killing it.

## 2026-07-09 — judge validation + render-partition fix (instrument + Thread 2)

Two follow-ups from a fresh session. The prior verdict rested entirely on
deterministic `coverage.py`; and Thread 2 (render saturation) had a tested fix
but was never scored on broad. Both now closed.

**Judge validation (does coverage FLOOR broad?).** API credit is exhausted, so
Opus-4.8 was run in-conversation as answerer+judge over the 19 broad probes,
strictly from each config's dumped `resonance_context` (baseline vs A_on/reanchor),
B=2000, symmetric across configs. Raw: `scratchpad/judge_validation.json`,
`scratchpad/broad_contexts.json`.

| config | coverage SR@B | JUDGE SR@B |
|---|---|---|
| baseline | .210 | .158 (3/19) |
| A_on (reanchor) | .158 | .105 (2/19) |

The judge AGREES with coverage: broad is genuinely walled, and reanchor is
neutral-to-regressive (judge −1 via `b_community` displacement — the same note the
2B run flagged). Coverage was NOT masking a reanchor win. Instrument-validity
worry resolved; "no lane ships / reanchor OFF" holds under the real judge.
(Caveat: judge = self on subscription; absolute numbers not comparable to the
historical API judge, but the cross-config ranking is.)

**Reading all 38 contexts exposed the failure mechanism:** broad needs ≥4 facts
from ≥4 distinct notes, but (a) the required notes are often not materialized, and
(b) selection pulls thematically-adjacent-but-WRONG notes on polysemous query
words — `b_meta_recurrence` "ideas recur at scale" → PM-iteration notes;
`b_tension_fault` "system's fault vs my responsibility" → goSTOPS ops notes. The
6 new synthesis probes (tension/lineage/scale/assembly/meta/bridge) score 0–1/4
under the judge too — not a coverage artifact.

**Render-partition fix (Thread 2), shipped behind flag `res_render_packed`
(default OFF → prod byte-identical).** `_emit` gained skip-and-continue; packed
mode drops the fixed `DEPTH_SHARE` split for one rank-ordered pool against the
full specifics budget. Ablation (`scratchpad/render_fix_ablation.json`):

| config | B | narrow | broad | paragraph |
|---|---|---|---|---|
| packed OFF | 2000 | .750 | .210 | .889 |
| packed ON | 2000 | .750 | **.210 (Δ0)** | .889 |
| packed ON | 4000 | .812 (+.062) | **.210 (Δ0)** | .889 |

Packed fills more of B (+85–113 tok @2000) and passes the forgetting gate
(recovers 1 narrow query @4000, no flooding) — but **broad Δ=0 at both budgets.**
The recovered budget renders MORE fragments from notes already in the pool, not a
4th distinct note. Kept, default OFF; does not ship as a broad lever.

**Converged conclusion.** Every lever DOWNSTREAM of the candidate pool is now
eliminated by two instruments: structure (Lane 1/2/arc) inert, reanchor
regressive, render-waste fixed-but-Δ0, N-cap inner-and-Δ0. Broad's wall is the
**candidate pool itself** (seeding + spread): the right notes are often not
candidates, and when they are, selection favours the strongest few / mismatches
on polysemy. Next lever (the only live one): candidate-pool breadth + semantic
precision, UPSTREAM of assembly. Free diagnostic to split it: per broad probe,
distinct notes the gold facts REQUIRE vs distinct notes present in the pool
pre-selection vs notes actually selected → reachability problem (seed/spread) vs
selection problem (assembly).

**Candidate-pool diagnostic** (`scratchpad/pool_diagnostic.json`): **`reachable_frac ≡ selected_frac` for ALL 19 probes** → wall = REACHABILITY, not selection (mean 0.39; 17 reachability-limited, 0 selection-limited, 2 covered). Assembly-side fixes (MMR, per-note slot cap) are DEAD. Caveat: fact→note mapping is weak (73% cosine <0.6) — broad key_facts are cross-note syntheses with no verbatim home — but `reachable≡selected` is robust to that.

**Oracle-seeding upper bound** (`scratchpad/oracle_seeding.json`) — the decisive test. Hand the answerer the best-possible evidence:

| baseline | oracle-V1 (nearest frag / fact) | oracle-V2 (full source note) |
|---|---|---|
| .210 | **.263** | .210 |

Perfect retrieval lifts broad by **at most +1 net query** (V1 drops 3 baseline passes; true ceiling ≈ ~7/19). **4 seeding-fixable vs 14 synthesis/absent:** for 14 probes even the single closest fragment per fact sits at cosine 0.22–0.52 (below τ) — the synthesized key_facts are not verbatim in any stored span.

**Final verdict: broad is SYNTHESIS/GOLD-bound, not retrieval-fixable.** Coverage@B checks verbatim presence of facts that are cross-note *syntheses* with no textual home — it structurally cannot score them, regardless of retrieval. This explains every null (lanes, reanchor, render, N-cap, reachability): the broad wall is substantially a **measurement wall**. No retrieval/seeding scheme can exceed ~.263 on Coverage@B. Only remaining live experiment: Opus-judge over the oracle-V1 contexts to split "coverage-τ floor for synthesis facts" (metric problem, judge-scored eval has headroom) from "content genuinely absent" (→ lever = consolidation-time synthesis nodes carrying synthesized TEXT).

## 2026-07-09 — committed subset (dead lanes reverted out of core)

Lanes all inert → only the genuinely-useful, safe pieces committed. **Kept:** `ENGAGEMENT` event + `retrieve.record_engagement` (demand-signal capture, log-only); the `res_render_packed` render-partition fix (calibration flag, default OFF — fills B, +1 narrow @4000, zero forgetting); the **qclm dead-slot bug fix** (ON — affects shipped 6d). **Reverted from core** (learnings above; code in `scratchpad/dead-lanes-full.patch`, prototypes in `scratchpad/`): Lane-1 traversal edges (6f), arc synthesis (6g), Workstream-A `ENGAGEMENT_QUERY_CLAIMS`, Workstream-D `ENGAGEMENT_SALIENCE`, `res_walked_edge_exempt`. Kept the +6 broad probes as a known-hard *synthesis* reference set (not verbatim-coverable). Test: `tests/test_engagement.py`.
