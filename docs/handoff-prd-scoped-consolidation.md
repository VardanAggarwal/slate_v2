# Handoff — PRD-scoped consolidation (fragment-layer claim genesis)

**Status:** approved for build, 2026-06-23. PoC validated (+50 pts SR@B @ B on a 30-episode sample).
**Branch:** `v3-changes`. **Owner:** —. **Predecessor context:** this doc + the PoC artifacts in
`scratchpad/` (`poc.py`, `armA.db`, `armB.db`, `evalA.json`, `evalB.json`).

---

## Why

`consolidate()` today re-derives claims with a **generative LLM `blueprint()` over `raw_text`**
(`core/consolidate.py:350`, `:1221`) and **never reads the fragment layer** that Write builds. This
violates two PRD lines:

- §21 "Raw content is stored as it is, and **should never have to be processed.** Pre-processed
  structured data and indexes are what get processed."
- §178 dedup "= PREDICTED route. **Replaces the cosine bands**" — i.e. predictor-only, no canon LLM.

And the blueprint does jobs the PRD assigns to the **predictor**, not the resolver: chunking
(`clusters`), membership, centre (`essence`/`kernel`), relations (`spine`). Only `claims` (distil
text) is sanctioned LLM work (§159).

## What the PoC showed (the evidence base)

Same 30-episode corpus / fragments / retrieval / judge; only claim-genesis differs. n=14 gold,
1 sonnet judge. Directional (subset depresses absolute SR@B); the **A-vs-B delta on identical
inputs** is the signal.

| | A — current (LLM blueprint over raw) | B — PRD-scoped (fragments + predictor dedup) |
|---|---|---|
| SR@B @ B=2000 | 35.7% | **85.7%** |
| tail (n=5) | 40.0% | **60.0%** |
| SR@B @ B/2=1000 | 35.7% | **57.1%** |
| mean ctx @ B | 728 tok | 1709 tok |
| build LLM calls | 77 | 16 (+27 canon-band eliminated) |
| build tokens (cost gate) | 228k | 214k |

- **Quality, not just cost, is the win** — inverted the original hypothesis. Blueprint's
  distillation ("strip rhetoric") discards the verbatim specifics the gold key-facts need.
- **Near-free on the cost gate**: tokens barely moved (blueprint is cheap haiku; the token cost is
  the concept pass, sonnet, run by *both* arms).
- **Zero forgetting**: A's passes ⊂ B's passes.
- **Parsimony caveat**: B uses more context. But at B/2 (≈comparable size) B still leads 57 vs 36 —
  the content is genuinely more answer-bearing per token, not just more padding.

---

## The change

Replace blueprint-as-claim-source with **fragment-as-claim-source**, keep the entire downstream
pipeline. The PoC did this by monkeypatching `blueprint()`; the real change makes it a first-class
path.

### Core: a fragment-sourced blueprint

`consolidate()` calls `blueprint(ep["raw_text"])` and feeds the resulting dict to
`_canonicalize_episode` / `_concept_pass` / `_relations`. Those consumers only read
`bp["clusters"][i]["claims"]` + `representative_sentences` (`consolidate.py:516-535`). So produce
the **same dict shape from the fragment layer**:

```python
def blueprint_from_fragments(conn, user_id, episode_id) -> dict:
    # fragments table: text, cluster, sent_start (core/store.py:143).
    # Group fragments by `cluster`; each fragment's verbatim text is a claim AND its
    # representative_sentence (it already IS a predictor-isolated span). No LLM.
    rows = conn.execute(
        "SELECT text, cluster FROM fragments WHERE user_id=? AND episode_id=? "
        "AND medoid_idx IS NOT NULL ORDER BY sent_start", (user_id, episode_id)).fetchall()
    # -> {"title","essence","clusters":[{label,kernel,claims[],representative_sentences[]}],
    #     "assumptions":[],"spine":[],"_method":"frag"}
```

Reference impl: `scratchpad/poc.py::blueprint_frag` (working, validated).

### Wiring in `consolidate()` (`consolidate.py:1215-1234`)

- Replace `bp, bp_cost = blueprint(ep["raw_text"])` with the fragment-sourced builder.
- **Fragment-empty guard**: if Write hasn't refined the episode yet (async), `episode_fragments`
  returns `[]` (`store.py:744`). Decide policy: **skip the episode this run** (it stays
  unconsolidated, next run retries — same pattern as the `LLMError` skip at `:1222`). Do *not*
  silently fall back to LLM-blueprint — that reintroduces the gap. Log the skip.
- Drop the `BLUEPRINTED` event payload or repurpose it to record `{method:"frag", fragment_ids}`
  for replay/audit. Check `_existing_blueprint` retry-reuse (`:370`) still works.

### Dedup must go predictor-only (PRD §178)

The PoC's 1-hour hang was the **canon LLM band** (`consolidate.py:550`) exploding on verbatim
fragments — hundreds of "uncertain" pairs escalated to the LLM. The PRD says dedup is the PREDICTED
route, no LLM. So:

- In `_canonicalize_episode` / `_dedup_route` (`:516`, `:397`), route on
  `predict.measure()`+`decide()` only; **uncertain → NEW** (don't escalate to `PROMPT_CANON`).
- This is what the PoC's `_counting_call` short-circuit simulated (canon_band_skipped=27). Make it
  real: gate the canon band behind a calibration flag defaulting OFF, or remove it.
- **Risk**: predictor-only dedup over-produces claims (B made 308, less-merged). The dedup z-gate
  (`DEDUP_Z_ECHO`, see memory `slate-p5-safety-core`) needs a tuning pass so it actually merges
  near-dupes instead of routing everything NEW. **This is the main quality risk of the change** —
  budget a calibration sweep for it.

### What stays unchanged

Concept pass (C3/C4), relations, reconcile, bridges, decay, prune — all consume claim ids/text and
are agnostic to how claims were minted. Retrieval (`recall.py`, `retrieve.py`, `hybrid.py`)
unchanged: it reads the same `claims`/`concepts`/`concept_members` tables, populated by the same
`CANONICALIZED`/`CONCEPT_*` events.

---

## Build order

1. `blueprint_from_fragments` + unit test (fragment dict shape == blueprint dict shape contract).
2. Predictor-only dedup: gate/remove the canon band; uncertain→NEW.
3. Wire into `consolidate()`; fragment-empty skip; BLUEPRINTED event handling + retry-reuse.
4. **Dedup z-gate calibration sweep** (the over-production risk).
5. Re-run the PoC harness on the **full corpus** (166 eps), not the 30-subset, at B and B/2 →
   confirm the delta holds and ΔSR@B clears the C-gate. Record forgetting rate.
6. Decide blueprint's fate: delete, or keep behind a flag as a no-fragment fallback only.

## Verification gates

- **C-gate** (`scratchpad/c_gate.py`): ΔSR@B ≥ 0, forgetting_rate == 0 on the frozen set.
- Full-corpus SR@B @ B and B/2 ≥ current `slate` baseline (was 50% / 43% pre-change at full corpus).
- Build tokens per query at steady state within the cost gate (PoC: ~flat, expect a slight *drop*).
- Existing suite (215 tests) green; add the dict-shape contract test + a predictor-only-dedup test.

## Open questions / decisions for the implementer

- **Fragment cluster labels** are weak (`write.py` assigns `cluster` heuristically). The concept
  pass leans on them. If concept quality drops, the fix is in clustering, not here — flag don't fix.
- **`essence`/`title`** — fragments don't carry a note-level essence. PoC used `clusters[0].claims[0]`.
  Fine for retrieval (essence isn't on the answer path), but if anything downstream depends on a
  real essence, source it from the centre fragment (`is_centre`, `store.py` fragments table).
- **Spine/relations**: PoC emitted empty `spine`. `_relations` (`:812`) then does less. PRD says
  relations are predictor-detected + LLM-confirmed (§186) — a follow-up, not this change. Note the
  regression so it's intentional.

## Artifacts

- `scratchpad/poc.py` — orchestrator: `select` / `prep` / `run` (arm A|B). `blueprint_frag` is the
  reference impl. Token-tally + canon-band short-circuit + CALL_CAP circuit-breaker live here.
- `scratchpad/{armA,armB}.db` — consolidated DBs (DON'T reuse; rebuild from a fresh
  `VACUUM INTO` snapshot — plain `cp` of the WAL-mode live db corrupts).
- `scratchpad/eval{A,B}.json` — per-query eval (B/2 run; B=2000 is in the `.log` files).
- Gotchas hit during the PoC: WAL copy corruption (use `VACUUM INTO`); `UID` is a read-only shell
  var (use a different name); HF embeddings are slow/unauthenticated locally (scope refine to the
  sample); `LLM_MAX_ATTEMPTS=1` to avoid backoff stalls in experiments.
