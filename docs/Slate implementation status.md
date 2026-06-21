# Slate — implementation status

Companion to `Slate PRD v2.md`. The PRD is the *what* (design, deferring mechanism). This file is the *how / now*: where the code stands today against that design, and the concrete gaps. Keep mechanism here, not in the PRD.

## Built vs missing

| Capability (PRD concept) | Status |
|---|---|
| Raw memory — immutable, append-only episodes | ✅ built (immutable by trigger) |
| Working memory as a re-derivable cache | ⚠️ exists, but rebuild replays the event log, not the raw record |
| Write-time surprise detection (echo / novelty / contradiction) | ✅ built — but binary, write-only |
| Continuous, stored, reused prediction error | ❌ the core gap (deferred "how") |
| Surprise-driven fragment boundaries | ⚠️ wrapper built (`core/scan.py`, measure()-iterator on `nearest_sim`), not wired — `encode.py` still sentence-splits |
| Spreading activation (associative recall) | ✅ built |
| Query decomposition | ⚠️ wrapper built (`core/scan.py`, shared with segment), not wired into `recall.py` |
| Value-of-information / token-budget stopping | ⚠️ wrapper built (`core/assembly.py`, measure()-iterator: greedy max-marginal-`residual` + STOP), not wired — `recall.py` still a `k` limit |
| Retrieval → consolidation signal loop | ❌ consolidation sees writes only |
| Batch consolidation, idempotent | ✅ built |
| Real forgetting / pruning | ❌ only relabels active→stale→dormant, never removes |
| Reconstruction-residual rule (safe merge + safe forget) | ⚠️ wrapper built (`core/guard.py`, measure()-iterator on `z`: forget=LOO, merge=losers-vs-survivors), not wired — `consolidate.py` MERGE still averages centroids |
| Belief reconciliation & versioning | ❌ no schema for status / version / qualifier |
| Bridge candidacy (non-obvious cross-theme links) | ⚠️ surfaced in MCP (`list_bridges`) but no consolidation step produces them via the residual band |
| Store-integrity check (derived claim vs source episode) | ❌ not built — PRD §40 scopes it inside consolidation |
| Run-level rollback of a bad consolidation | ❌ events append-only; replay reproduces the bug |
| Re-derivation from the raw record (true firewall) | ❌ only event-log replay exists |
| Deletion / redaction path (privacy) | ❌ episodes immutable, no exception |
| Cold-start behaviour | ❌ empty store → everything maximally surprising |
| External eval harness (recall ground truth) | ❌ **critical-path, build first** |

## Concrete gaps found in code (file pointers)

- **Firewall is weaker than the PRD's safety story assumes.** `rebuild` truncates the semantic layer and replays the **event log** (`consolidate.py:198-206`); a bad MERGE/SPLIT is itself an event, so replay reproduces it deterministically. True re-derivation must re-run consolidation from the immutable episodes, bypassing the log. Not built.
- **No run-level rollback.** Events are append-only with no compensating/revert concept. Fix: tag every event with its `run_id` (runs already tracked in `consolidation_runs`); rollback = drop run N's events and rebuild to N−1; escalate to full re-derivation when the log baked in the bad decision.
- **Forgetting only relabels.** Decay transitions active→stale→dormant (`consolidate.py:571-593`); dormant rows live forever. No pruning, no compression.
- **MERGE averages.** Concept centroid is a normalized mean of member vectors (`store.py:631-647`); a merge folds the loser's nuance into the mean. Needs the reconstruction-residual guard before committing.
- **Versioning has no schema home.** `claims` has no status/version column; `relations` has no `superseded_by` type; there is no qualifier/condition field (`store.py:88-133`). This is a schema migration.
- **Contradiction link is fragile.** Stance is classified at write (`encode.py:129`) but the `contradicts` relation is only materialized in consolidation via a "first claim in cluster" heuristic (`consolidate.py:497-503`) — the link to the actual contradicting sentence isn't preserved.
- **Event-log payload versioning.** Adding prediction-error magnitude and retrieval-signal events changes payload shape; old ENCODED/CANONICALIZED events lack the new keys. `apply_event` reads fixed keys — rebuild over a mixed-version log must backfill or null-default.
- **NLI label order is hard-coded.** `encode.py:70` assumes `["contradiction","entailment","neutral"]`; a model swap (config-overridable, `config.py:32`) would silently mislabel every contradiction — the highest-surprise event the whole spine relies on.
- **Multi-user is real but unstated in design.** Every row partitions by `user_id` (`store.py`); `delete_user` leaves the corpus behind (`store.py:292-296`) — no redaction path.

## Predictor wrappers (context, mechanism)
The three §0 wrappers are **thin iterators over `predict.measure()`** — the spine — not modules that re-derive residual geometry. Each is a choice of X, Y and which measurement FIELD to read. Built + validated, **not yet wired into any stage**.

`measure()` was expanded to make this possible:
- `x` accepts `str` | `list[str]` | `list[dict]` (a dict carries its own embedding → reused, no re-embed).
- `exclude_self` — leave-one-out when X ⊆ Y (scan, guard), matched by exact self-similarity.
- **scale-aware** — `WARMUP_MIN_CORPUS` dropped 12→2; the top-k operators already cap `k` at pool size, so `SPAN_K`/`STAT_K` degrade to "use all available" on a small Y. This lets one `measure()` serve a within-note corpus AND the full memory; the large-corpus Write path is unchanged.

| Wrapper | X vs Y (via measure) | reads | loop shape |
|---|---|---|---|
| `core/scan.py` — segment / decompose | X = each sentence, Y = the note's own **causal prefix** | **`nearest_sim`** (cosine field; residual is a weak boundary signal at sentence scale) | one measure() per sentence |
| `core/guard.py` — merge / forget | `forget`: X=Y=cluster members (LOO); `merge`: X=losers, Y=survivors | **`z`** (residual vs the region's own spread) | one measure() call |
| `core/assembly.py` — stop / allocate / dedupe | X = candidates, Y = query + growing assembly | `residual` | loops measure() per greedy step (Y grows) |

Cut/forget thresholds are SPREAD-RELATIVE (the region's own cohesion sets the bar) and live in a scopeable JSON **calibration profile** (each module's `DEFAULT_CALIBRATION`), resolved by `predict.calib_value(profile, key, default, scope)` — fitted at consolidation and pushed down, never baked as constants. Note guard's `z` scale depends on Y's cohesion, so `z_forget` is genuinely per-scope calibration, validated ≈−0.5 on real clusters. Tests: `tests/test_wrappers.py` (14, synthetic + near-threshold + overlapping-topic) and the LLM-judged real-corpus probes in `tests/manual/`.

## Storage (context)
SQLite + sqlite-vec (L2-normalized kNN), single `engine.db`, WAL. Embeddings 384-dim all-MiniLM-L6-v2 — HF Inference API in prod (no torch on the 1GB host), local SentenceTransformer in dev. FTS5 over episodes. Schema version 2 (multi-user).
