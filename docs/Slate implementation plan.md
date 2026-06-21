# Slate — iterative implementation plan

Companion to `Slate PRD v2.md` (the *what*) and `Slate implementation status.md` (where the code stands). This is the *build order*: phased, dependency-ordered, each phase gated on the North Star metric (SR@B).

## Current state (grounding)

- **Predictor primitive is built and write-side validated** (`core/predict.py`: `measure` → `decide` → `resolve_direction`), but **not wired into any stage** — `encode.py`/`consolidate.py`/`recall.py` don't call it. It's a validated island.
- **The three §0 wrappers are thin iterators over `measure()`** (the spine), not from-scratch geometry. `measure()` was expanded to enable this: `x` takes `str | list[str] | list[dict]`, `exclude_self` for leave-one-out when X⊆Y, and scale-aware warmup (12→2) so it serves a within-note corpus as well as full memory. **scan** reads `nearest_sim` per sentence vs its causal prefix; **guard** reads `z` (forget=LOO over a cluster, merge=losers-vs-survivors); **assembly** loops measure() per greedy step, reading `residual` against the growing assembly. All three ✅ rewritten as measure() iterators. Thresholds are spread-relative, in a scopeable calibration profile resolved by `predict.calib_value` (not hardcoded). Validated on synthetic + real corpus (`tests/test_wrappers.py` 14 cases + `tests/manual/` probes); still **not wired into any stage**. See execution plan §0.
- Write still uses **binary** echo/novelty/contradiction; fragments are **sentence-split**, not surprise-cut.
- Retrieval has spreading activation but **no query decomposition, no VOI stopping** (just a `k` limit).
- Consolidation **averages merges, only relabels (never prunes), has no versioning schema, sees writes only**.
- **No SR@B eval harness** — status doc flags this "critical-path, build first."

## Plan

Ordered by dependency. Each phase has an exit gate; nothing ships without moving SR@B (or, for Phase 0, *enabling* it to be measured).

### Phase 0 — Eval harness (build first; everything gates on it)
Without the North Star metric, no later phase is judgeable. Build the *minimum* viable SR@B first, expand later.
- Frozen benchmark set + adaptive probe set, kept strictly separate (PRD §Measurement).
- Per-query **key-facts checklist** (pre-registered, not judge-picked); two-judge binary scoring + κ.
- Competitors **RAG@B / grep@B**; reference = **RAG/grep @ 3B oracle** (gold where it exists).
- Cost gate: per-query steady-state build tokens vs `k×` baseline.
- **Exit:** can produce SR@B (overall + tail slice) + ΔSR@B on the frozen set for any code change. Start with ~30 hand-built gold queries over the live corpus.

### Phase 1 — Wire the predictor into Write
The primitive exists; this is integration, the highest-ROI move.
- Build the **sequential-scan segmenter** (thin wrapper: cut where residual jumps) → replaces sentence-split.
- Replace binary stance logic in `encode.py` with `measure → decide → resolve_direction` (only AMBIGUOUS hits the LLM).
- Schema: persist `z`, `route`, `weight`, `anchor_id` per fragment (status doc flags event-payload versioning — backfill/null-default old events).
- **Exit:** content-presence diagnostic holds; SR@B@B not regressed vs sentence-split.

### Phase 2 — Close the calibration loop (Write ↔ Consolidate)
`decide()` reads a calibration it can't fit itself.
- Wire `compute_baselines` at consolidation; persist calibration (`z_echo`, `prox_margin`, per-cluster).
- Consolidation **fits calibration against SR@B** and pushes it down to write.
- **Exit:** fitted calibration beats the hard-coded `Z_ECHO=-3.5` default on frozen SR@B.

### Phase 3 — Retrieval (the least-proven uses)
Status doc + PRD both warn: query/passage embedding asymmetry may break the residual-at-retrieval assumption. **Measure that first** before building on it.
- Spike: does a query embed near its answering memory? If not, query-aware embedding.
- Then: query **decomposition** (scan wrapper), **residual-against-growing-assembly loop**, **VOI stopping**, budget allocation per fragment.
- Emit retrieval signals (fetched / dropped / cut-for-budget) — needed by Phase 4.
- **Exit:** Retrieve-sufficiency@B beats current `k`-limit recall; Slate ≥ RAG@B on frozen set.

### Phase 4 — Consolidation safety + the signal loop
Where structure changes, so where the risks live.
- **Reconstruction-residual guard** (the third wrapper) → safe merge / safe forget; stop averaging centroids.
- Real **pruning** (not just relabel), **belief versioning schema** (status / version / qualifier + `superseded_by`).
- Feed Phase 3's retrieval signals into consolidation (promote / demote / resolve).
- **Bridge candidacy** (residual band between concept medoids — feeds the existing `list_bridges` MCP) and the **store-integrity check** (derived claim vs its source episode, PRD §40).
- **Run-level rollback** (`run_id` on events; drop run N, rebuild to N−1) + **true re-derivation from raw** (not event-log replay).
- **Exit:** ΔSR@B positive, **catastrophic-forgetting rate = 0** on frozen set.

### Phase 5 — Open frontier (PRD "still to address")
- Cold-start per-region graduation curve; redaction / privacy propagation; write-during-consolidate semantics; multi-user redaction path (`delete_user` currently leaves the corpus behind).

## Critical path

Phase 0 → 1 → 2 are tightly coupled; go in order. Phase 3 carries the biggest unproven risk (embedding asymmetry) — de-risk with the spike before committing. Phase 4 is the heaviest schema/safety lift.
