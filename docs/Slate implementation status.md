# Slate — implementation status

Companion to `Slate PRD v2.md` (the *what*) and `Slate execution plan.md` (the step map).
This file is the *how / now*: what the **code actually runs today**, traced from the MCP entry
points through `encode.py`, `write.py`, `consolidate.py`, `recall.py`, `hybrid.py` and
`retrieve.py` — not what the docs describe. File:line refs are ground truth on branch
`v3-changes`.

## The shape of it today

Slate runs as **two derived layers over one immutable raw store, and retrieval reads both.**

- **Raw memory** — `episodes` + `episode_sentences` + `vec_sentences` + the `ENCODED` event.
  Immutable, trigger-enforced. Ground truth for every rebuild.
- **Fragment layer** — built by **Write** (W2–W8) off the sentence vectors via the predictor.
  Replayable from the `FRAGMENTED` event. Read by the hybrid retriever's fragment path.
- **Claims / concepts layer** — built by **Consolidate**, which re-extracts claims with an
  **LLM `blueprint()` over `raw_text`** (not over fragments) and then runs the predictor + guard
  over the resulting claims.
- **Retrieve** — `assemble_context` → `hybrid.hybrid_context` blends the concept/claim path
  (`recall`) with the fragment path (`retrieve` + assembly), split by a calibrated `concept_share`.

The predictor (`core/predict.py`) is the spine and is genuinely reached by Write and Consolidate.
The remaining gaps are **tuning, not topology**.

---

## WRITE — all 8 steps reached at runtime

`save_note` (MCP) runs W1 synchronously, then spawns a daemon thread (`trigger_refine_async`,
`write.py:414`) for W2–W8. The CLI `encode` runs `refine_episode` (`write.py:333`) inline.
Refine makes **zero new embedding calls** — medoids reuse the W1 sentence vectors; fragment
vectors aren't stored, they're reconstructed from a medoid index on read.

| # | Step | Mechanism | Code |
|---|---|---|---|
| W1 | Persist raw (sync) | episode + sentences + vec_sentences, `ENCODED` event with receipt | `encode.py:216` · `store.insert_episode:508` |
| W2 | Segment | causal `nearest_sim` z-score boundaries (DROP_Z) | `scan.fragments:116` ← `plan_fragments:79` |
| W3 | Variable-resolution | FOLD_Z keeps fine fragments where surprise is high | `scan.fragments:141` |
| W4 | Intra-note dedup | if a kept sibling beats memory, re-measure vs `M ∪ siblings` | `write.py:206` · `_nearest_kept_sibling:98` |
| W5 | Batched match | one GEMM: all fragments vs memory → anchor + residual | `route_fragments:154` · `predict.measure:189` |
| W6 | Route & resolve | `decide()` → route; NLI `resolve_direction` fires **only on AMBIGUOUS** | `predict.decide:430` · `resolve_direction:481` |
| W7 | Centre & peak | medoid = min-residual sentence; novel peak = max z (read off measurements) | `_fragment_medoid:121` |
| W8 | Persist working | `FRAGMENTED` event + `insert_fragment` (NOVEL/AMBIGUOUS only); race-guarded, replayable | `mark_fragmented:1049` · `apply_fragmented:285` |

**Status: faithful to spec.** Every W-step is wired and reached. LLM is invoked only on the
AMBIGUOUS route.

---

## CONSOLIDATE — 13 of 14 active in a run

Offline batch (CLI `consolidate` / nightly cron). It does **not** read the fragment layer — each
episode is re-blueprinted by an **LLM over `raw_text`** (`consolidate.py:1156`), and the
predictor + guard then operate on the resulting claims/concepts. `CANON_*` cosine constants
still fire as the cold-start fallback when a neighbourhood is smaller than `SPAN_K`.

Reached in `consolidate()`'s main loop, in order:

| # | Step | Mechanism | Code |
|---|---|---|---|
| C1 | Triage / revisit | sort batch by contradiction count then novelty | `_revisit_order:1133` |
| C2 | Dedup claims | `measure()`/`decide()` vs neighbourhood; CANON_* only when cold | `_dedup_route:397` · `_canonicalize:1163` |
| C3 | Concept membership | spread-relative z gate (≤ `CONCEPT_MEMBERSHIP_Z`), cosine fallback | `_membership_z:610` · `_concept_pass:1171` |
| C4 | Split / re-anchor | bimodal spread flags SPLIT candidates; LLM decides | `_spread_is_bimodal` · `_concept_geometry:626` |
| C5 | Relations & bridges | residual band → bridge candidates; LLM confirms | `_relations:1174` · `_bridges:1185` |
| C6 | Merge safely | `guard.merge` partitions loser into FOLD vs KEEP at the merge | `guard.merge:725` |
| C7 | Forget / prune | dormant concepts only; leave-one-out drops reconstructable members | `_prune_safely:1090` |
| C8 | Reconcile & version | `contradiction_pairs` → supersede/scope/version; margin-before-flip | `_reconcile:1177` · `_resolve_conflict:790` |
| C9 | Demote background | re-predicted ≥ N times → `BACKGROUNDED`, folds into theme | `_demote_background:1016` |
| C10 | Integrity check | claim vs source episode; ungrounded → `INTEGRITY_FLAGGED` (log-only) | `_check_integrity:1179` |
| C11 | Cold-start fallback | `CANON_*` cosine path when a neighbourhood ≤ `SPAN_K` | `_dedup_route:435` |
| C13 | Consume retrieval signals | active — now fed by the live fragment path's R8 (committed in MCP) | `_consume_retrieval_signals:1040` |

**In a normal run (added):**

- **C12 — fit baselines + push down.** `_fit_baselines` runs at the end of `consolidate()`:
  re-clusters every fragment onto its nearest consolidated concept (`RECLUSTERED` event →
  `store.set_fragment_clusters`, so it survives rebuild / reverts on rollback), then
  `predict.compute_baselines` over the re-clustered corpus → `store.set_baselines`. Write loads
  the pushed snapshot (`store.get_baselines` → `route_fragments(baselines=…)`) instead of
  recomputing per write. So the **measurement half** of calibration is now "fitted at
  consolidation and pushed down". The **Q,B bet** (`value_floor`/`concept_share`) still needs the
  LLM-judged SR@B sweep and stays the offline `eval/fit_stop.py --push` pass — by design.

**Not in a normal run:**

- **C14 — rollback / re-derive.** `rollback_run:288` is implemented and correct, but admin-invoked,
  not part of a normal pass.

---

## RETRIEVE — hybrid, both layers live

`assemble_context` now runs `hybrid.hybrid_context` (`mcp_server.py:343`). It reserves a
`concept_share` slice of the budget for the concept/claim path and gives the rest to the fragment
path, then merges. If one side is empty it re-runs the other at full budget — a safe superset of
the old concept-only behaviour. The separate `recall(query, k)` tool still runs pure spreading
activation for quick lookups.

**Concept path** (`hybrid.py:70` → `recall.assemble_context` → `recall.py:42`): the hard tail —
synthesis across notes over the claims/concepts graph (kNN claims/concepts + 2-hop spread,
≤ concept_budget).

**Fragment path** (`retrieve.py:236` → `fragment_recall` + assembly): specificity — verbatim
spans with the predictor's assembly VOI stop. Per-step status:

| Step | Status |
|---|---|
| R1 decompose | **on** (profile flag, default ON) |
| R2 value_floor stop | per calibration |
| R3 borrow cross-theme | **on** (profile flag, default ON) |
| R4 budget alloc | on |
| R5 VOI stop (`GAIN_FLOOR`) | on |
| R6 prioritise | on |
| R7 triage | on |
| R8 emit signals (`signals=True`) | **on** — committed in MCP, feeds C13 |

---

## Code vs the plan

| Area | Plan says | Code does |
|---|---|---|
| Fragments → retrieve | P2.5: "orphan branch closed" | ✅ **Now true.** `assemble_context` → `hybrid` reads the fragment path via `retrieve.assemble_context`. |
| Live budget B | assembly stops at the VOI-maximising size | ✅ **Now live.** Fragment path runs the VOI stop; budget split by `concept_share` (each path still char-bounded). |
| C13 signals | retrieval signals close the loop | ✅ **Now closed.** Fragment path emits R8 (`signals=True`), MCP commits, C13 consumes. |
| Consolidate input | predictor spine reshapes working memory | Claims still re-extracted by an **LLM blueprint over `raw_text`** each run; fragments aren't a consolidation input. |
| R1 / R3 | decompose query · borrow cross-theme nuance | ✅ **Now wired.** Profile flags (`decompose`/`borrow` in `DEFAULT_CALIBRATION`), default ON, threaded through hybrid→assemble→`fragment_recall`. A fitted profile can still flip either off. |
| C12 calibration | "fitted at consolidation and pushed down" | ✅ **Baselines now in the loop.** `_fit_baselines` re-clusters fragments onto concepts + pushes `compute_baselines` down; Write loads it. The `value_floor`/`concept_share` Q,B bet stays the offline `fit_stop.py` sweep (LLM/quota-gated). |
| Per-stage logic | measure/decide + 3 wrappers, magnitude-not-direction | **Faithful.** Write W1–W8 and Consolidate C1–C11 reach the predictor as described. |

**Net:** the topology now matches the intent — retrieval is a hybrid over both the
LLM-blueprinted claims/concepts graph and the predictor-native fragment layer, and the R8→C13
signal loop is closed. R1/R3 are now wired ON and C12 fits + pushes baselines down inside a run;
the only deferred C12 piece is the LLM-judged `value_floor` SR@B sweep (offline by design).
Remaining gaps are cross-cutting §4 work (redaction, multi-user isolation leak at `store.py:292`,
write-during-consolidate snapshot, cost gate).

---

## Predictor wrappers (mechanism)

The three wrappers are **thin iterators over `predict.measure()`** — the spine — not modules that
re-derive residual geometry. Each is a choice of X, Y and which measurement FIELD to read.

| Wrapper | X vs Y (via measure) | reads | loop shape |
|---|---|---|---|
| `core/scan.py` — segment / decompose | X = each sentence, Y = the note's own **causal prefix** | **`nearest_sim`** (residual is a weak boundary signal at sentence scale) | one measure() per sentence |
| `core/guard.py` — merge / forget | `forget`: X=Y=cluster (LOO); `merge`: X=losers, Y=survivors | **`z`** (residual vs the region's own spread) | one measure() call |
| `core/assembly.py` — stop / allocate / dedupe | X = candidates, Y = query + growing assembly | `residual` | loops measure() per greedy step (Y grows) |

`measure()` is scale-aware (`x` accepts `str` | `list[str]` | `list[dict]` carrying its own
embedding; `exclude_self` for leave-one-out; top-k operators cap `k` at pool size), so one
`measure()` serves a within-note corpus and the full memory unchanged. Cut/forget thresholds are
SPREAD-RELATIVE and live in a scopeable JSON calibration profile (each module's
`DEFAULT_CALIBRATION`, resolved by `predict.calib_value`), fitted offline and pushed down — never
baked as constants. Tests: `tests/test_wrappers.py` (synthetic + near-threshold + overlapping-topic)
plus LLM-judged real-corpus probes in `tests/manual/`.

## Storage (context)

SQLite + sqlite-vec (L2-normalized kNN), single `engine.db`, WAL. Embeddings 384-dim
all-MiniLM-L6-v2 — HF Inference API in prod (no torch on the 1GB host), local SentenceTransformer
in dev. FTS5 over episodes. Schema version 2 (multi-user).
