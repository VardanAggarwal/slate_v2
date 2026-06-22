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
| C1 | **Prioritise what to revisit first** — spend effort where surprise was highest | rank by residual; AMBIGUOUS (unresolved residuals on anchors) first | `measure()` rank over pending items (work-ordering, not a write) | undifferentiated batch pass | AMBIGUOUS/high-residual items processed before low; resolver budget spent on them first |
| C2 | **Dedupe** claims | claim vs rest of memory | `measure()` + iterative-assembly | `CANON_AUTO_SAME=0.92` | planted dup set collapses to 1; near-but-distinct kept |
| C3 | **Assign** claim → theme | claim vs each concept | `measure()` nearest-cluster | similarity heuristic | assignment matches planted membership |
| C4 | **Split / Form / Re-anchor** concepts | members vs concept centroid | `measure()` spread test | — | bimodal concept splits; drifted anchor re-centres to medoid |
| C5 | **Bridge candidacy** — non-obvious links between themes | medoid vs medoid (residual *band*: close enough to relate, enough residual to be non-obvious) | `measure()` between concept medoids; emit bridge candidates | — | planted related-but-distinct concept pair → bridge proposed; near-dup pair (residual≈0) and unrelated pair (residual≫) → no bridge |
| C6 | **Merge safely** — ✅ BUILT (2026-06-22) | loser's nuance vs survivor structure | `guard.merge` partitions loser members into FOLD (winner reconstructs) vs KEEP (carries nuance) at the MERGE decision; partition frozen in the `MERGED` payload (`fold_claim_ids`/`kept_claim_ids`) → replay-deterministic, guard never re-runs at apply. Loser survives if any nuance remains; full-fold deletes it (legacy events fold all). Embeddings read from `vec_claims` (no HF call inside txn). | centroid averaging | ✅ folds dup / keeps nuance / full-fold deletes loser / partition survives rebuild (4 cases); geometry in `test_wrappers` |
| C7 | **Forget safely / prune** — ✅ BUILT (2026-06-22) | pruned item vs what remains | `_prune_safely`: DORMANT concepts only, `guard.forget` leave-one-out → `PRUNED` event drops reconstructable members (orphaned claim deleted, re-derivable from raw via C14), protects irreplaceable ones however quiet; never empties a concept (≥1 representative); thin concepts (< `PRUNE_MIN_MEMBERS`) skipped. | relabel-only decay | ✅ drops reconstructable / protects irreplaceable / skips active / never empties (4 cases); geometry in `test_wrappers`. NOTE: "quiet"=usage gate is C13 (deferred); prune currently gates on dormant age only |
| C8 | **Detect conflicts + version** — ✅ BUILT (2026-06-22) | ambiguous residual → `_resolve_conflict()` (LLM) | `claims` gain `status`/`superseded_by`/`qualifier`/`version_group` (additive migration). `_reconcile` resolves each `contradicts` edge into supersede/scope/version; `VERSIONED` event applies it. **Margin-before-flip** on claim `strength` (`VERSION_FLIP_MARGIN`): a sub-margin supersede is held as a *version*, incumbent stays current → no oscillation. Resolution frozen in payload → replay-deterministic. recall surfaces ⚖️ contested/superseded; loser never dropped. | first-claim heuristic | ✅ 6 cases: flip past margin / blocked-by-margin held / scope-with-qualifiers / both-stand / survives rebuild / contested-surfaced. Resolver direction mocked (LLM's job); margin+versioning logic tested. NOTE: margin metric is `strength` until C13 usage signals |
| C9 | **Demote to background** | repeated low-residual item over time | usage + repetition signal (logged, not measured) | — | item surprising once then echoed N× → folded into theme |
| C10 | **Store-integrity check** — derived claim faithful to its source | derived claim vs its source episode | `measure()`/route between claim and raw episode | — | derived claim that contradicts its raw source (without changing today's answer) flagged; faithful claim passes |
| C11 | **Gauge saturation + graduate region from cold-start** | region spread stability | `compute_baselines` maturity check | no cold-start handling at all | region with < graduation count → trusts prior; matured region → trusts local |
| C12 | **Fit calibration + push down** | SR@B over candidate `z_echo`/`prox_margin` | calibration persistence + fit loop | hard-coded `Z_ECHO=-3.5` | fitted calibration beats default on frozen SR@B; pushed to Write/Retrieve |
| C13 | **Consume retrieval signals** | — | promote/demote/resolve from R8 events | consolidation sees writes only | salient-but-never-retrieved demoted; dropped-but-needed promoted; rare-correct NOT suppressed |
| C14 | **Run rollback + re-derive from raw** — ✅ BUILT (2026-06-22) | — | `consolidate.rollback_run(run_id)`: flag run `rolled_back` (events kept on disk for audit), free its episodes, re-materialize from the active log. `store.ACTIVE_RUN_PREDICATE` excludes rolled-back runs from `rebuild` AND the `_existing_*` re-derive guards → re-consolidation re-derives freed episodes from raw, ignoring the poisoned events. NO events-schema change (reuses `consolidation_runs.status`). | log-replay only | ✅ `test_consolidate.py` 4 cases: full reversal to pre-run state; events survive but unmaterialized; re-consolidate re-derives (blueprint re-called, not reused); unknown-run noop |

**Stage gate (C):** ΔSR@B ≥ 0 on the frozen set AND **catastrophic-forgetting rate = 0**
(no previously-passing frozen query now fails) AND every run reversible.

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
P3  Calibration loop (W↔C) ......... ◐ IN PROGRESS — relevance-aware STOP built as opt-in `value_floor`
                                     knob (assembly.py, default OFF; geometry-tested, 147/147). Fit driver
                                     eval/fit_stop.py (resumable, sweeps value_floor vs frozen SR@B). LEFT:
                                     run the quota-gated SR@B sweep + push the fitted floor down (C12).
P4  RETRIEVE — spike then build ..... ◐ BUILT (structure complete; tuning deferred). R0 spike ✅ PASS. R1 decompose /
                                     R3 borrow / R8 retrieval-signals built (core/retrieve.py, opt-in default-OFF);
                                     R2/R7 value_floor stop + R4/R5/R6 assembly + frag+concept HYBRID (core/hybrid.py)
                                     done. First-principles review (vs PRD §Retrieve) fixed 3 gaps: R7 answerability
                                     TRIAGE now returns-nothing for off-corpus queries (was: pad with nearest); R8
                                     signals now EMITTED on the answerer path (assemble_context, default-ON); R3 borrow
                                     rebuilt to topic→query-residual→match-off-topic via predict.residual_direction
                                     (was: max-novelty vs chosen). tests 33/33 retrieve+predict. R-gate credit + knob
                                     fits (value_floor/concept_share/triage/borrow) DEFERRED to the final SR@B pass.
P5  CONSOLIDATE safety + signals .... ◐ IN PROGRESS — safety core ✅: C14 rollback + C6/C7 reconstruction-guard
                                     merge/forget + C8 conflict versioning (margin-before-flip). 16 invariant/
                                     wiring tests; geometry in test_wrappers; resolver direction mocked.
                                     LEFT: C1 (revisit-order), C9 (background decay), C10 (store-integrity),
                                     C11 (cold-start graduation), C12 (calibration fit), C13 (retrieval signals).
                                     SR@B + forgetting-rate credit deferred to the quota window (as P3/P4).
                                     EXIT: C-gate (ΔSR@B≥0, forgetting=0, rollback works)
P6  Frontier (cross-cutting §4) ..... EXIT: redaction/isolation/write-during-consolidate tests green
```

**Critical path:** P0→P1→P2→P2.5 are ✅. **P2.5 closed the orphan branch** — `core/retrieve.py` reads
the fragment layer via the assembly wrapper and is wired as the `frag` SR@B answerer, so `z_echo`/the
per-cluster knob finally move something measurable. **P3 (calibration) is now unblocked** and is the next
step; its top lever is the relevance-aware stop (R2/R7) that kills the observed over-injection — fitted
against SR@B, not pre-tuned. P4's R0 asymmetry risk is **retired** (spike passed; symmetric MiniLM kept),
but its R-gate is **not yet met** by fragment-only retrieval (frag loses the tail to claims; grep dominates
this short-note corpus) — the frag+concept hybrid is the path to the gate, and is now **built**
(`core/hybrid.py`, wired as the `hybrid` answerer, offline-tested 6/6); its SR@B credit is deferred with the
quota-gated sweep. P5 is the heaviest schema/safety lift. P6 is independent.

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
