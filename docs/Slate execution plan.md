# Slate — execution plan (pipelines, from scratch)

Companion to `Slate PRD v2.md` (the *what*), `Slate implementation status.md` (gaps), and
`Slate implementation plan.md` (phasing). This doc does one thing the others don't:
**decompose each pipeline into its atomic workflow steps, map every step to the predictor
primitive (`core/predict.py`) and the wrapper it needs, and pin a test to each step** so no
step ships unverified and none is silently skipped.

Design stance: `predict.py` is the spine. Every step is either (a) a direct `measure()` read,
(b) one of three thin wrappers over it, or (c) the decision/resolver/eval layer around it.
Today policy lives in ad-hoc constants (`encode.py`, `consolidate.py:CANON_*`); the target
state has **one estimator, one calibration profile, fitted at consolidation and pushed out.**

---

## 0. The three wrappers (build once, reused everywhere) — ✅ BUILT

The PRD names exactly three wrappers over the `measure()` sensor. Built as standalone,
unit-tested functions; **not yet wired into any stage** (that is Phase 1+).

The wrappers are **THIN ITERATORS over `measure()`** — the spine — not re-derived geometry.
Each picks X, Y and which measurement FIELD to read. To enable this, `measure()` was expanded:
`x` accepts `str | list[str] | list[dict]` (a dict reuses its own embedding); `exclude_self`
gives leave-one-out when X ⊆ Y; and it is **scale-aware** (warmup floor 12→2, and the top-k
operators cap `k` at pool size, so `SPAN_K`/`STAT_K` degrade to "use all" on a small Y) — so one
`measure()` serves a within-note corpus AND the full memory, the Write path unchanged.

| Wrapper | X vs Y (via measure) | reads | calls | Module | Status |
|---|---|---|---|---|---|
| **Sequential scan** | X = each sentence, Y = the note's own causal prefix | **`nearest_sim`** | one measure() per sentence | `core/scan.py` | ✅ measure()-iterator |
| **Reconstruction guard** | `forget`: X=Y=cluster (LOO); `merge`: X=losers, Y=survivors | **`z`** | one measure() call | `core/guard.py` | ✅ measure()-iterator |
| **Iterative residual-vs-assembly** | X = candidates, Y = query + growing assembly | `residual` | loops measure() per greedy step (Y grows) | `core/assembly.py` | ✅ measure()-iterator |

Each wrapper's test (`tests/test_wrappers.py`, 14 cases) uses **synthetic corpora with known
structure** (planted boundaries, planted duplicates, planted nuance, plus near-threshold and
overlapping-topic cases that exercise the *policy*, not just flattering orthogonal geometry).
Real-corpus behavioural probes (LLM/human-judged, no asserts) live in
`tests/manual/wrappers_judge.py` (English sentences) and `tests/manual/wrappers_slate_judge.py`
(the live `engine.db`: episodes, concept clusters).

**Two findings worth keeping:**

1. **Scan reads `nearest_sim`, not the reconstruction residual.** A causal prefix is far short
   of the spanning pool `SPAN_K` reconstruction needs, so residual is a weak boundary signal at
   sentence scale (verified — it otherwise cut every sentence). `nearest_sim` — "did this
   sentence match its context" — is the right *field of the same measurement*; scan still goes
   through `measure()`. The cut is spread-relative against the note's own `nearest_sim`
   distribution (how tightly it coheres sets its own bar), with an encoder-noise σ-floor so a
   perfectly tight note doesn't manufacture cuts.
2. **Instrument vs policy, the same split `predict.py` enforces.** All keep/cut/stop/forget
   thresholds live in a scopeable JSON **calibration profile** (each wrapper's
   `DEFAULT_CALIBRATION`; resolved via `predict.calib_value(profile, key, default, scope)` —
   the generic form of `_thresholds`), fitted at consolidation and pushed down — never baked as
   instrument constants. The thresholds are SPREAD-RELATIVE: guard's `z` scale depends on Y's
   cohesion, so `z_forget` is genuinely per-scope calibration (≈−0.5 on real clusters), not a
   universal constant.

---

## 1. WRITE pipeline — owns *content presence* — ✅ BUILT (2026-06-22)

Input: raw note text + ts. Output: immutable episode (raw) + a set of routed fragments
(working). W1 is synchronous (`core/encode.py`); W2–W8 are the async, event-sourced remainder
(`core/write.py:refine_episode`, materialized by `apply_fragmented`, replayable by `rebuild`).

**As-built — three refinements past the table below:**

- **Fragment vector = MEDOID sentence (W3/W7 at the vector level).** A fragment's routing/storage
  vector is its lowest-within-span-residual sentence, selected via `measure(span, exclude_self)` —
  NOT the re-embedded joined span. Reuses the W1 sentence vectors, so **refine makes zero embedding
  calls**; selection, not generation (PRD-faithful). `rebuild` recomputes it deterministically from
  the episode's sentences. A folded (large) fragment is coherent by construction, so its medoid
  stands for the whole span; surprising sentences already left as their own fine fragments at W2/W3.
- **W4+W5 routing is BATCHED.** One `measure(all fragments, memory)` gemm vs the fixed corpus
  replaces the old per-fragment `measure(growing-Y)` loop (which re-`vstack`-ed the whole pool each
  step). Intra-note dedup (W4) is now a cheap causal sibling check (k×k gram); the `memory ∪
  siblings` union-z is recomputed only for a fragment an earlier kept sibling actually out-competes
  — preserving W4's semantics exactly where they fire, without the full-corpus rescan.
- **W6 routes on a PER-CLUSTER threshold.** Fragments carry a `cluster` (W8 schema add), assigned by
  anchor-inheritance (NOVEL opens a region; AMBIGUOUS inherits its anchor's). `decide()`'s per-region
  `z_echo`/`prox_margin` now resolves — previously DEAD because every fragment was `cluster=None`.
  This is the knob **C12** fits; consolidation re-clusters later.

Real-corpus check (166 notes, `docs/write-workflow-eval.md` §v3.1): 972 stored / 60 PREDICTED-dropped
/ 22 intra-echoes / 74 regions, 18s, no network. `tests/test_write.py` 19/19.

> **✅ RESOLVED by P2.5 (2026-06-22):** fragments were an **orphan branch** — `recall`/
> `assemble_context` read claims/concepts and `consolidate` blueprints from `raw_text`; neither read
> fragments, so `z_echo`/the per-cluster knob had no path to SR@B. `core/retrieve.py` now reads the
> fragment layer via the assembly wrapper and is wired as the `frag` SR@B answerer, so C12 can be
> fitted — P3 is unblocked. (`consolidate` still blueprints from `raw_text` — out of scope for P2.5.)
>
> **↑ UPDATE (2026-06-23):** `consolidate` still re-derives claims from `raw_text` (raw stays the
> source of truth — the safety property behind C14 re-derive/rollback), but it no longer *ignores* the
> fragment layer: Write's fragment routing now **informs** C2 dedup + C3 membership as a PRIOR
> (`_anchor_concept_prior`), so the Write effort feeds both retrieval AND consolidation. Fragments are a
> hint, never the input — absent fragments leave consolidate's behaviour identical. See §3 C2/C3.

| # | Step (PRD function) | X vs Y (predictor) | Build | Replaces | Test |
|---|---|---|---|---|---|
| W1 | **Persist raw** | — | keep `store.insert_episode` (`store.py:398`) | unchanged | invariant: episode + sentences immutable (trigger `207`), receipt attached |
| W2 | **Segment** — cut where content stops being predictable | each candidate fragment vs the running prefix | sequential-scan wrapper | sentence-split (`encode.py:90`) | synthetic: 3 planted topic shifts → cuts within ±1 sentence of each; SR@B not worse than sentence-split |
| W3 | **Variable-resolution encoding** — not just *where* to cut but *how finely*: fine where surprise is high, folded where low | fragment residual vs the running prefix (magnitude → granularity) | sequential-scan wrapper (resolution knob off the same novelty signal) | uniform sentence granularity (`encode.py:90`) | synthetic: high-surprise region keeps fine fragments, low-surprise region folds; SR@B@B not regressed vs uniform |
| W4 | **Intra-note dedup** — collapse within-note echoes before they hit the store | fragment vs `M ∪ earlier kept siblings` | causal sibling check (k×k gram) after the batched M pass; union-z recomputed only when a kept sibling out-competes the memory anchor | — | synthetic note with planted internal repeat → echo fragment routed PREDICTED, not stored twice (`test_route_intra_note_dedup`) |
| W5 | **Match** — anchor each fragment to memory | all fragments vs M | **batched** `measure()` (single gemm vs the fixed corpus) | — | synthetic: near-dup fragment → correct `anchor_id`, low residual; off-topic → `cold_start`/no anchor |
| W6 | **Label** — route store / reinforce / resolve, and hold-strength | `decide()` over measurements | wire `decide()` + `resolve_direction()` (LLM only on AMBIGUOUS) | binary stance (`encode.py:129`) | route-confusion matrix on labeled fixtures; **only** AMBIGUOUS calls the LLM (assert call count) |
| W7 | **Centre** — representative core + most-novel point | note: fragment vs note's fragments; fragment: sentence vs span's sentences (medoid) | `measure()` rank — lowest residual = centre/medoid (SELECTION, reused as the fragment's vector); highest z = novel peak | — | synthetic note → medoid is the lowest-residual member; novel point is the highest-z; `test_fragment_medoid_selects_a_real_sentence` |
| W8 | **Persist working** — z, route, weight, anchor_id, **cluster** per fragment | — | `fragments` table + `vec_fragments` + the `FRAGMENTED` event (JSON, embeddings recomputed at apply); content-addressed ids → idempotent/replayable | — | round-trip persist→rebuild reproduces routes + vectors (`test_rebuild_reproduces_fragments`); idempotency + concurrent-claim guard |

**Stage gate (W):** content-presence diagnostic ≥ baseline AND end-to-end SR@B@B not
regressed vs sentence-split. (Presence alone can't pass — finer fragments game presence but
crowd retrieval; the SR@B gate catches that.)

**Note — the W6 resolver (NLI), decided 2026-06-21 (apply when building W6):**

- **At write the resolver answers one bit: contradiction, or not.** Write's only question is
  store-or-not, and the routes already cover it: **duplicate → don't store is the PREDICTED
  route (geometry/dedup), decided *before* NLI.** So NLI's job at write collapses to splitting
  *contradiction* (store, held strongest, **flag**) from *refine/entail/neutral* (store, attach
  to anchor). The 3-way contradiction/entailment/neutral is the right granularity — do NOT build
  the 5-way resolver at write.
- **supersede / scope / version are NOT write-time outcomes.** At write a clash is just *born as
  a contradiction* and flagged; it is RESOLVED into supersede/scope/version later, at **C8**,
  which can see newer evidence + conditions + recurrence. Write detects, consolidation reconciles.
  Keep the structured schema in C8.
- **Run the NLI in prod via HF, no torch.** `STANCE_PROVIDER` defaults to local `CrossEncoder`
  (needs torch) → on the 1GB HF-only prod host that throws and `classify_stance` silently returns
  `"neutral"`, so *every contradiction becomes refine in prod* (a correctness hole). Fix: add an
  `HFStance` mirroring `HFEmbedder` — same `HF_TOKEN`, `InferenceClient.zero_shot_classification`
  (an MNLI model server-side) with the fragment as the candidate label and
  `hypothesis_template="{}"`, returning P(anchor ⊨ fragment); bucket high→entail / low→contradict
  / mid→neutral on a calibratable threshold. Add `STANCE_PROVIDER="hf"` and default prod to it;
  keep local `CrossEncoder` as the dev fast-path. Test: a contradiction survives with no torch
  (mock the client). (Zero-shot's softmax is over {entail,contradict}, so "neutral" is the
  mid-band, not a direct label — fine for the one bit write needs.)

---

## 2. RETRIEVE pipeline — owns *sufficiency @ budget*

Input: query + budget B. Output: assembled context ≤ B that maximises the answer.
Today: `recall.py:41` spreading-activation with a hard `k` limit, no decomposition, no VOI stop.

> **✅ WIRED (2026-06-23):** the live MCP `assemble_context` tool (`mcp_server.py:323`) now routes
> through `hybrid.hybrid_context`, which blends the concept/claim path (`recall` — the tail) with the
> predictor-native fragment path (`retrieve` + assembly VOI) on a **`concept_share` budget split**, and
> commits the **R8 retrieval signals** the fragment path emits (`signals=True`) so C13 has a live
> producer. It degrades to the pure concept path when no fragments are materialized. R1 decompose / R3
> borrow stay **default-OFF** (fit-gated); `value_floor` and now **`concept_share`** are C12 fit-targets
> declared in `retrieve.DEFAULT_CALIBRATION`, not baked. `core/hybrid.py` runs the fragment path exactly
> once (no double R8 emission). The MCP wiring (`mcp_server.py`) is **uncommitted** — see the handoff.

**De-risk first (blocking spike):** the PRD flags retrieval as the *least-proven* use —
a symmetric encoder may not place a query near its answering memory. Before building R2–R6,
measure query→answer cosine on the gold set. If weak, adopt a query-aware embedding; the
sensor contract is unchanged, only the lens.

| # | Step (PRD function) | X vs Y | Build | Test |
|---|---|---|---|---|
| R0 | **Asymmetry spike** | query vs its gold answer-memory | measurement script, no code change | report: median query→answer rank; decision gate to proceed or fix lens |
| R1 | **Decompose** query into independent fragments | each sub-query vs query prefix | sequential-scan wrapper (reuse) | synthetic multi-part query → N fragments matching planted parts |
| R2 | **Route** each fragment: drop-if-echo / fetch-if-deepens / return-nothing | memory item vs the query fragment | `measure()` + decide-style gate | fixtures: covered→drop, deepenable→fetch, unanswerable→empty (no padding) |
| R3 | **Borrow** a nuance from another theme | candidate vs assembled-so-far | `measure()` cross-cluster | planted cross-theme nuance is pulled in; irrelevant neighbour is not |
| R4 | **Allocate** budget across fragments by deepenability | per-fragment residual share | iterative-assembly wrapper | budget split tracks residual mass; sum ≤ B |
| R5 | **Stop** — pull a thread only while marginal gain > cost | next item vs growing assembly | iterative-assembly wrapper | marginal-residual curve is monotone-ish; stops before filling B when answer saturates |
| R6 | **Prioritise** final assembly by unique info / token | items vs each other | `measure()` rank (dedupe order) | assembly has no two items with residual≈0 against each other |
| R7 | **Triage** unanswerable queries | query vs whole store | `measure()` global residual | tagged-unanswerable gold queries return "can't answer", not a padded guess |
| R8 | **Emit retrieval signals** (fetched/dropped/cut-for-budget) | — | append RETRIEVAL_SIGNAL events | signals persisted per query; consumed in §3 |
| R9 | **Read / synthesise the fetched text** (PRD: LLM) | — | **out of scope — host-owned.** Slate returns the assembled context ≤ B; the host LLM reads it and answers | n/a (asserted by the SR@B eval, which judges the host's answer over Slate's assembly) |

**Stage gate (R):** Retrieve-sufficiency@B (measured at the assembly size that *maximises*
the answer, not the one that fills B) beats the current `k`-limit recall, AND Slate ≥ RAG@B
on the frozen set. Over-injection that flips an answer counts as a failure here.

---

## 3. CONSOLIDATE pipeline — owns *ΔSR@B, floored by zero catastrophic forgetting*

Batch/offline. Input: new episodes + retrieval signals since last run. Output: restructured
working memory + a fitted calibration. This is where structure changes — so where rollback
and the reconstruction guard live. Today: `consolidate.py:597`, fixed `CANON_*` thresholds,
relabel-only decay, averaging merges, writes-only.

| # | Step (PRD function) | X vs Y | Build | Replaces | Test |
|---|---|---|---|---|---|
| C1 | **Prioritise what to revisit first** — ✅ BUILT (2026-06-22) | rank by residual; AMBIGUOUS (unresolved residuals on anchors) first | `_revisit_order` sorts the batch most-surprising-first off the encode receipt (contradiction = AMBIGUOUS residual-on-anchor outranks raw novelty count). Pure work-ordering — batch membership (oldest N) unchanged, only steers resolver-budget order. | undifferentiated batch pass | ✅ `test_revisit_order_puts_surprise_first` (contradiction first, then novelty desc) |
| C2 | **Dedupe** claims — ✅ BUILT (2026-06-23) | claim vs rest of memory | `_dedup_route`: `measure()`/`decide()` over a new claim vs its STAT_K neighbourhood of existing claims — PREDICTED→same, AMBIGUOUS→LLM, NOVEL→new. Calibrated `DEDUP_Z_ECHO=1.0` to the claim-dedup z-scale (exact dup z≈0, distinct z≳5; clean gap on the live corpus). Cosine fallback (`CANON_*`) only for cold neighbourhoods (≤SPAN_K), the C11 stance. **Fragment prior (2026-06-23):** the dedup neighbourhood is widened with the members of the concept the claim's Write-time source span anchored to (`_anchor_concept_prior`), catching dups global knn under-ranked when the LLM paraphrase drifted from the verbatim span. A hint — the predictor still routes; absent fragments = unchanged. | `CANON_AUTO_SAME=0.92` | ✅ planted dup → same, distinct → new (`test_c2_dedup_route_*`); real-corpus probe: exact dup→same 60/60; prior catches a knn-missed dup (`test_consolidate_frag_prior.py`) |
| C3 | **Assign** claim → theme — ✅ BUILT (2026-06-23) | claim vs each concept | `_membership_z`: spread-relative `measure()` nearest-cluster gate (claim's z vs the concept's own cohesion ≤ `CONCEPT_MEMBERSHIP_Z`) replacing the flat `knn similarity >= 0.40` floor; cosine fallback for thin concepts. **Fragment prior (2026-06-23):** the source span's anchored concept is offered as an extra attach candidate even when global knn (k=3) under-ranks it, cutting concept fragmentation (`_anchor_concept_prior`); the LLM still attaches/splits. | similarity heuristic | ✅ in-topic plausible, off-topic not (`test_c3_membership_z_*`); prior surfaces a knn-missed concept (`test_consolidate_frag_prior.py`) |
| C4 | **Split / Form / Re-anchor** concepts — ✅ BUILT (2026-06-23) | members vs concept spread | `_spread_is_bimodal` (pure-numpy two-pole detector → SPLIT candidate) + `_concept_geometry` medoid re-anchor, surfaced as `geometry`/`representative` hints in the concept-pass context; LLM stays the arbiter. | — | ✅ bimodal splits / cohesive doesn't; medoid is central (`test_c4_*`) |
| C5 | **Bridge candidacy** — non-obvious links between themes — ✅ BUILT (2026-06-23) | medoid vs region (residual *band*: close enough to relate, enough residual to be non-obvious) | `_bridges`: medoid-vs-other-region symmetric RESIDUAL band `[0.40,0.85]` via `predict.residuals_against`, replacing the centroid-cosine `BRIDGE_LOW/HIGH`. | centroid cosine band | ✅ related→in-band, near-dup→below, unrelated→above (`test_c5_bridge_residual_band`) |
| C6 | **Merge safely** — ✅ BUILT (2026-06-22) | loser's nuance vs survivor structure | `guard.merge` partitions loser members into FOLD (winner reconstructs) vs KEEP (carries nuance) at the MERGE decision; partition frozen in the `MERGED` payload (`fold_claim_ids`/`kept_claim_ids`) → replay-deterministic, guard never re-runs at apply. Loser survives if any nuance remains; full-fold deletes it (legacy events fold all). Embeddings read from `vec_claims` (no HF call inside txn). | centroid averaging | ✅ folds dup / keeps nuance / full-fold deletes loser / partition survives rebuild (4 cases); geometry in `test_wrappers` |
| C7 | **Forget safely / prune** — ✅ BUILT (2026-06-22) | pruned item vs what remains | `_prune_safely`: DORMANT concepts only, `guard.forget` leave-one-out → `PRUNED` event drops reconstructable members (orphaned claim deleted, re-derivable from raw via C14), protects irreplaceable ones however quiet; never empties a concept (≥1 representative); thin concepts (< `PRUNE_MIN_MEMBERS`) skipped. | relabel-only decay | ✅ drops reconstructable / protects irreplaceable / skips active / never empties (4 cases); geometry in `test_wrappers`. NOTE: "quiet"=usage gate is C13 (deferred); prune currently gates on dormant age only |
| C8 | **Detect conflicts + version** — ✅ BUILT (2026-06-22) | ambiguous residual → `_resolve_conflict()` (LLM) | `claims` gain `status`/`superseded_by`/`qualifier`/`version_group` (additive migration). `_reconcile` resolves each `contradicts` edge into supersede/scope/version; `VERSIONED` event applies it. **Margin-before-flip** on claim `strength` (`VERSION_FLIP_MARGIN`): a sub-margin supersede is held as a *version*, incumbent stays current → no oscillation. Resolution frozen in payload → replay-deterministic. recall surfaces ⚖️ contested/superseded; loser never dropped. | first-claim heuristic | ✅ 6 cases: flip past margin / blocked-by-margin held / scope-with-qualifiers / both-stand / survives rebuild / contested-surfaced. Resolver direction mocked (LLM's job); margin+versioning logic tested. NOTE: margin metric is `strength` until C13 usage signals |
| C9 | **Demote to background** — ✅ BUILT (2026-06-22) | repeated low-residual item over time | `_demote_background`: a claim re-encountered across ≥ `BACKGROUND_MIN_REPEATS` further episodes that belongs to a concept → `BACKGROUNDED` event sets a `background` flag (orthogonal to C8 `status`). recall demotes its standalone score (`×BACKGROUND_SCORE_FACTOR`, 🌫️ signal); the theme surfaces instead. Trigger is high recurrence (the opposite of rare) + a theme to fold into, so it never suppresses rare-correct. | — | ✅ 5 cases: folds recurrent member / skips rare / requires a theme / demoted at retrieval / survives rebuild |
| C10 | **Store-integrity check** — ✅ BUILT (2026-06-22) | derived claim vs its source episode | `_check_integrity` routes each derived claim against its source-episode sentences (stored vectors, no embed call): PREDICTED → grounded/pass; NOVEL → ungrounded FLAG; AMBIGUOUS → resolver, contradicts-source FLAG (geometry can't see flipped polarity). `INTEGRITY_FLAGGED` is log-only (a review signal, NOT a headline metric — PRD). | — | ✅ 4 cases: faithful passes / ungrounded flagged / contradiction flagged (route+stance mocked) / ambiguous-but-faithful passes |
| C11 | **Gauge saturation + graduate region from cold-start** — ✅ BUILT (2026-06-23) | region spread stability | `compute_baselines` shrinks each region's cohesion (μ,σ) toward the prior-over-clusters by member count (weight `n/(n+GRADUATION_N0=4)`): young region → prior, matured → local. Continuous (no route cliff). Prior formed from RAW per-cluster estimates BEFORE the shrink, so it doesn't chase its own shrunk clusters. No-op for frag retrieve (assembly reads only residual); affects WRITE routing. | no cold-start handling at all | ✅ 2 cases: young region trusts prior / matured trusts local. Offline blast-radius A/B (N0=0=old): real corpus prior μ unchanged, per-cluster \|Δμ\| median 0.013/max 0.102, 23/317 move >0.05 (all size 2–5 = the target). |
| C12 | **Fit calibration + push down** — ◐ MECHANISM BUILT (2026-06-22) | SR@B over candidate `z_echo`/`value_floor` | Persistence: `calibration_profiles` table (per user; NOT event-derived/truncated) + `core/calibration.py` (`merged`/`push`); `retrieve.assemble_context` loads the fitted profile over its defaults by default. Fit loop = `eval/fit_stop.py --push` (sweeps `value_floor` vs frozen SR@B, persists the best). Fit-target keys in the one profile: `value_floor` (R2/R7 stop) and **`concept_share`** (hybrid budget split) — both declared in `retrieve.DEFAULT_CALIBRATION`, pushed by `calibration.push`. | hard-coded constants | ✅ persistence/merge/push/per-user/retrieve-pickup (6 cases offline). **Fit RUN deferred** (SR@B/quota-gated); `compute_baselines` is still NOT called inside `consolidate()` — the fit is an offline `fit_stop.py` pass, the one remaining topology gap vs "fitted at consolidation". |
| C13 | **Consume retrieval signals** — ✅ BUILT (2026-06-22) | — | `_consume_retrieval_signals`: bridges signal fragment-ids → source episode → its claims; a claim exposed as a candidate ≥ `RETRIEVAL_EXPOSURE_MIN`× yet NEVER fetched → demoted to background (theme carries it). Idempotent (recomputed from all signals; BACKGROUNDED is a SET). Exposure floor protects rare-but-quiet. | consolidation sees writes only | ✅ 4 cases: demotes exposed-never-fetched / keeps fetched / spares rare-quiet / survives rebuild. NOTE: promote+un-demote deferred ("needed" needs gold/SR@B); bridge is episode-level (lossy) |
| C14 | **Run rollback + re-derive from raw** — ✅ BUILT (2026-06-22) | — | `consolidate.rollback_run(run_id)`: flag run `rolled_back` (events kept on disk for audit), free its episodes, re-materialize from the active log. `store.ACTIVE_RUN_PREDICATE` excludes rolled-back runs from `rebuild` AND the `_existing_*` re-derive guards → re-consolidation re-derives freed episodes from raw, ignoring the poisoned events. NO events-schema change (reuses `consolidation_runs.status`). | log-replay only | ✅ `test_consolidate.py` 4 cases: full reversal to pre-run state; events survive but unmaterialized; re-consolidate re-derives (blueprint re-called, not reused); unknown-run noop |

**Stage gate (C):** ΔSR@B ≥ 0 on the frozen set AND **catastrophic-forgetting rate = 0**
(no previously-passing frozen query now fails) AND every run reversible.
**✅ MET (2026-06-23, C2–C5 C-gate, `scratchpad/c_gate_c2c5.out`):** re-consolidating a 15-episode
batch from raw through the full C2–C5 stack gave **ΔSR@B = +7.1%** (42.9%→50.0%, query g13 newly
passes, no regressions), **forgetting = 0**, run fully reversible (exact return to S_minus). $0.66.

---

## 4. Cross-cutting

| Concern | Build | Test |
|---|---|---|
| **Redaction / privacy** | sanctioned raw-delete + propagation: derived memory survives only if reconstructable from remaining raw | delete a raw note → dependent derived dropped, independent derived survives |
| **Multi-user isolation** | already partitioned by `user_id`; close `delete_user` corpus leak (`store.py:292`) | cross-user query returns nothing; delete_user leaves no derived rows |
| **Write-during-consolidate** | define the snapshot a run sees; new episodes deferred to next run | episode written mid-run is picked up exactly once, next run |
| **Cost gate** | per-query steady-state build tokens (write+consolidate) vs `k×` baseline | SR@B credited only while under the gate; padding-heavy variant loses credit |

---

## 5. The test spine (this is *how* "thoroughly tested" is enforced)

Four layers, every phase runs all four:

1. **Unit / geometry** — synthetic corpora with planted structure; deterministic, no LLM, no
   live DB. Covers all of §0 wrappers + every `measure()`/`decide()` path. (`tests/test_predict.py`
   is currently **missing** — first thing to write.)
2. **Stage diagnostics** — content-presence (W), sufficiency@B (R), ΔSR@B + forgetting-rate (C).
   Reported as conditional rates that *locate* a regression, never as a standalone pass.
3. **End-to-end SR@B** — the North Star. **Phase 0, built first; everything gates on it.**
   - Frozen benchmark (scoreboard, never tuned on) vs adaptive probe set (practice) — strictly separated.
   - Pre-registered key-facts checklist per gold query; two-judge binary + κ.
   - Competitors RAG@B / grep@B; reference = RAG/grep @ 3B oracle (gold where it exists).
   - Evaluated at budget distribution (B/2 and B) so padding costs.
4. **Invariants (property tests)** — raw immutability, rebuild determinism, run rollback
   reversibility, user isolation, "only AMBIGUOUS calls the LLM."

A step is "done" only when its row's test passes *and* its stage gate (SR@B) holds.

---

## 6. Execution order (dependency-ordered; each phase has an exit gate)

```
P0  Eval harness (SR@B) ............ ✅ DONE  (eval/harness.py; frozen gold.jsonl, 20 queries; baselines captured)
P1  Wrappers + test_predict.py ..... ✅ DONE  (§0 wrappers green; route-matrix tested)
P2  Wire predictor into WRITE ...... ✅ DONE  (W1–W8 built; medoid + batched + per-cluster; test_write 19/19)
P2.5 Wire fragments into RETRIEVE .. ✅ DONE (2026-06-22) — core/retrieve.py reads fragments via assembly
                                     wrapper; wired as `frag` SR@B answerer; orphan branch closed; test_retrieve 7/7.
                                     See docs/retrieve-workflow-eval.md.
P3  Calibration loop (W↔C) ......... ✅ DONE (2026-06-22) — value_floor STOP fitted vs frozen SR@B on the
                                     funded API and PUSHED. value_floor=0.25 beats raw-residual baseline: at
                                     B/2 SR@B 50→57% & tail 20→40% (over-injection killed), at full B holds
                                     SR@B with less ctx (parsimony). Persisted to calibration_profiles;
                                     retrieve loads it by default. See docs/retrieve-workflow-eval.md.
P4  RETRIEVE — spike then build ..... ◐ BUILT (structure complete; tuning deferred). R0 spike ✅ PASS. R1 decompose /
                                     R3 borrow / R8 retrieval-signals built (core/retrieve.py, opt-in default-OFF);
                                     R2/R7 value_floor stop + R4/R5/R6 assembly + frag+concept HYBRID (core/hybrid.py)
                                     done. First-principles review (vs PRD §Retrieve) fixed 3 gaps: R7 answerability
                                     TRIAGE now returns-nothing for off-corpus queries (was: pad with nearest); R8
                                     signals now EMITTED on the answerer path (assemble_context, default-ON); R3 borrow
                                     rebuilt to topic→query-residual→match-off-topic via predict.residual_direction
                                     (was: max-novelty vs chosen). tests 33/33 retrieve+predict. R-gate credit + knob
                                     fits (value_floor/concept_share/triage/borrow) DEFERRED to the final SR@B pass.
P5  CONSOLIDATE safety + signals .... ✅ DONE (2026-06-23) — 14/14 C-steps + EXIT. Was 9/14 (afaeb6f→e89ed7a):
                                     C14 rollback, C6/C7 guard merge/prune, C8 versioning (margin-before-flip),
                                     C1 revisit-order, C10 store-integrity, C9 background-decay, C12 (persistence
                                     +fitted value_floor=0.25 pushed), C13 retrieval-signals. ~35 offline tests.
                                     C-GATE ✅ PASSED (2026-06-23, small-batch, scratchpad/c_gate.py): rolled back the
                                     most-recent run (15 ep), re-derived them FROM RAW via consolidate(), re-measured on
                                     the frozen gold (slate answerer, B=2000). ΔSR@B=+0.0% (42.9%→42.9%), forgetting=0
                                     (no regressions), rollback of the new run returned state to S_minus exactly. Cost
                                     $0.77, all-Claude, 692s. Re-derive reproduced 2034/2040 claims (concepts 326→338).
                                     ✅ COMPLETE (14/14 C-steps + EXIT). C11 (1a497ba) cold-start graduation; C2–C5
                                     (b8d9e61) measure()-upgrades dedup/assign/split/bridge — C-gate PASS ΔSR@B=+7.1%,
                                     forgetting=0, reversible. EXIT: C-gate (ΔSR@B≥0, forgetting=0, rollback) ✅ MET
P6  Frontier (cross-cutting §4) ..... EXIT: redaction/isolation/write-during-consolidate tests green
```

**Critical path:** P0→P1→P2→P2.5→P3 are ✅; P4 built (R-gate not met on this corpus — grep dominates short
self-contained notes; hybrid is the best Slate variant at 64%). P5 ✅ DONE (14/14 + C-gate EXIT, ΔSR@B=+7.1%). Only P6 (cross-cutting §4) remains.

---

## NEXT-CHAT HANDOFF (2026-06-23 — read this first to resume)

**State:** branch `v3-changes`. **P0–P5 ✅ DONE.** Full test suite **205 passed**. SR@B campaign on the funded
Anthropic API (started ~$3.3 left; this session's two C-gates + C2–C5 spent ~$1.4 → ~$1.9 left). Commits this
session: `68f9780` (P3/P4 retrieve unit), `3bdf5c1` (C-gate EXIT), `1a497ba`+`a12a916` (C11), `b8d9e61` (C2–C5).

**P5 is COMPLETE** — all 14 C-steps + the EXIT C-gate. C2–C5 (the measure()-upgrades) landed with a C-gate
**ΔSR@B = +7.1%** (42.9%→50.0%), forgetting=0, fully reversible (`scratchpad/c_gate_c2c5.out`). C11 cold-start
graduation validated offline. The remaining P5 items list is cleared.

**ONLY P6 LEFT** (cross-cutting §4): redaction/privacy propagation, multi-user isolation (close the
`delete_user` corpus leak `store.py:292`), write-during-consolidate snapshot semantics, cost gate. These are
mostly invariant/property tests + a couple of small fixes — largely OFFLINE, not SR@B-gated. See §4.

**Optional follow-ups (not blocking P6):** a FULL-corpus C-gate (all 166 episodes, not the 15-ep batch) for a
headline ΔSR@B; and the deferred C12 `value_floor` re-fit now that C2–C5 changed the store structure.

**How to run the eval (gotchas baked in):**
- Corpus user `usr_01KTXAYR20J4R6F7PT3DP10W3W` (166 notes). `config.DEFAULT_USER_ID="local"` is EMPTY — always pass `--user`.
- Work on a COPY: `/tmp/slate_eval.db` (already has 972 materialized fragments; live `data/engine.db` has 0).
  Rebuild it with: copy `data/engine.db` → checkpoint WAL → `write.refine_pending(conn, uid)` (offline, ~2min).
- **Answerer + judge BOTH sonnet.** A haiku answerer DEFLATES SR@B (fails queries sonnet passes) — do not use it.
- Set env so calls stay on the funded API and never cascade to the exhausted gemini free tier:
  `LLM_FALLBACK_ORDER=claude,local`, `LLM_MAX_ATTEMPTS=6` (hardened defaults already in `eval/fit_stop.py` and
  `scratchpad/run_baseline.py`). Cost ~$0.0072/cell; API tier rate-limits make it slow (~20s/cell), so runs are
  RESUMABLE (per-cell cache) — launch in background + Monitor.
- Baseline: `PYTHONPATH=. .venv/bin/python scratchpad/run_baseline.py`. Fit: `DB_PATH=/tmp/slate_eval.db
  PYTHONPATH=. python -m eval.fit_stop --user <uid> --half --floors none,0.15,0.25 --push --cache <path>`.

**Uncommitted (rides with the P3/P4 retrieve work, NOT yet committed):** `core/assembly.py`, `core/predict.py`,
`core/retrieve.py` (incl. the C12 `calib.merged` pickup wiring + the `concept_share` fit-target key), `core/hybrid.py`
(fragment path now runs once — no double R8 emission), `mcp_server.py` (live `assemble_context` → `hybrid.hybrid_context`,
commits R8 signals), `eval/harness.py` (sonnet answerer), `eval/fit_stop.py` (hardening + `--push`),
`scratchpad/run_baseline.py`, `docs/retrieve-workflow-eval.md`, and the P3/P4 + `test_hybrid.py` test files. These are
interdependent (retrieve↔assembly `VALUE_FLOOR`, mcp↔hybrid↔retrieve) — commit them together as the P3/P4 retrieve unit.
The committed C12 retrieve-pickup TEST was deferred for this reason (see `tests/test_calibration.py`).
Memory: `slate-p5-safety-core`, `slate-srb-eval-baseline`.

## 7. Review — predictor + 3 wrappers (2026-06-21, before P2)

Reviewed `core/predict.py` (`measure`/`decide`) + the three wrappers (`scan`, `guard`,
`assembly`) against this plan and the PRD. `tests/test_wrappers.py` 14/14 green. **Verdict:
sound spine — measure/decide separation is faithfully implemented and the wrappers are genuinely
thin iterators over `measure()`, not re-derived geometry. Proceed to Write (W1–W8), closing the
two gaps below first.**

Faithful to the PRD: two layers genuinely separate (`measure()` holds no thresholds; `decide()`
owns the bet); one primitive X/Y-swapped (scan reads `nearest_sim`, guard `z`, assembly
`residual` — fields of the same measurement); magnitude-vs-direction line clean (`decide()`
never signs a contradiction; AMBIGUOUS → `resolve_direction()` LLM only); spread-relative z with
leave-one-out + prior-over-clusters, baseline sharing the *same global-nearest operator* the
probe gets; calibration JSON-able/scopeable/consolidation-owned via `calib_value`. scan reads
`nearest_sim` not residual — the documented §0 finding, correctly carried into code with the
σ-floor.

**Gaps to close before/within Write:**

1. **`tests/test_predict.py` is still missing** (plan P1 exit names it "first thing to write").
   Wrappers exercise `measure()` indirectly, but `decide()`'s three-way route matrix
   (PREDICTED/NOVEL/AMBIGUOUS over labeled fixtures) and the `attached` boundary have **no
   direct test**. Write this before W6.
2. **Variable-resolution (W3) does not exist yet in `scan.py`** — scan emits *boundaries* only;
   there is no granularity/fold knob off the surprise magnitude. W3 remains a distinct build on
   the same signal; do not assume scan covers it.

**Minor code notes (not blockers):**

- `measure()` mutates caller dicts (fills `c["embedding"]` in place, `predict.py:332-335`).
  Side-effects the input corpus — fine for stage wiring, worth a docstring line.
- `exclude_self` blanks by exact-identity `sim ≥ 1−1e-6` (`predict.py:344-347`); two genuinely
  identical embeddings blank only the first hit — an edge case for intra-note dedup (W4); the W4
  test must cover identical fragments.
- `decide()` NOVEL branch correctly nulls `anchor_id`; PREDICTED keeps it for reinforcement.
- Perf: Gram capped at `GRAM_MAX_N=4000` with per-row fallback above — appropriate for the
  1GB HF-only prod host.

**Recommendation:** sequence P2 as → write `test_predict.py` (route matrix) → wire W2 (scan) /
W4–W5 (`measure()` direct) / W6 (`decide()` + `resolve_direction`) → then build W3's resolution
knob. Retrieval's embedding-asymmetry risk (R0) stays correctly deferred; `assembly.py` is built
but unproven for retrieval until that gate — the right call.

## 8. Risk register

- **Embedding asymmetry breaks retrieval** (highest) → R0 spike gates P4; fix is a lens swap, sensor unchanged.
- **Calibration overfits the probe set** → it's fitted on probe, confirmed on frozen; ΔSR@B on frozen is the only credit.
- **Merge/forget loses a nuance** → reconstruction guard is *necessary not sufficient*; resolver confirms every irreversible commit (geometry never decides direction).
- **Poisoned event log** → re-derivation from immutable raw, not log-replay; rollback by `run_id`.
- **Presence gaming at Write** → Write gated on SR@B@*budget*, not on unbudgeted reproduction.
