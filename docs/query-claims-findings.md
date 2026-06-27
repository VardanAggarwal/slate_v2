# Findings — encoding retrieval queries in memory ("query-claims")

Thread date: 2026-06-27. Status: **SHIPPED into core consolidation, default OFF** (step 6d,
`INJECT_QUERY_CLAIMS=False`). Fully wired + rebuild-safe + leak-safe; not yet *enabled* —
that waits on the validation gate (§Open). See §Shipped for the prod wiring.

## The idea (user)
A single retrieval lights up only the geometry-nearest region, so cross-note ("broad")
synthesis queries miss. If we **encode the queries themselves** into memory as a usage
signal, concepts that keep co-firing under the same questions — even when far apart in
embedding space — can be wired together, so the *next* retrieval spreads into regions a
single geometric shot would miss. ("Light up things that were not lit during retrieval.")

## Honesty constraint
Gold is eval-only. The query stream is **synthetic, generated blind to gold** from concept
labels (cross-topic pairs → "what would a reader ask that needs both?"), cached to
`scratchpad/qbridge_qstream.json`. Generated on the local Claude subscription (`claude -p`).

## Two readings tested

### A. Query → concept relation-bridges (`scratchpad/query_bridges.py`)
Harvest concept pairs that co-fire under the synthetic queries but are geometrically distant
(cos < 0.45 = structural holes), write them as `query_bridge` relations.
**Result: Δ0 on every gold.** Inert — consistent with the documented "geometric bridges
inert" wall. Concept→concept relation edges are weight 0.7, PE-gated, reached only at hop ≥1;
the spreader effectively never traverses them.

### B. Query-as-claim through the full pipeline (`scratchpad/query_claims.py`)
Inject each synthetic query as a first-class **claim** (`qclm_*` id), attached to the top-K
concepts it activates as `kind='redundant'` members. Why claims beat relation-bridges:
- a query-claim is a **knn SEED target** — a future similar query lands on it at hop-0
  (strong direct activation), not a fragile multi-hop;
- its memberships are **weight-1.0 edges** (vs 0.7 relations); a synthesis query is naturally
  a **multi-concept member** → a real structural-hole hub the spreader always crosses;
- it has **no source episode / fragments**, so it routes activation but contributes **zero
  answer text** (verified against the materialization path).

Results (cosine coverage proxy `eval.coverage`, all three golds, n: narrow 16 / broad 13 / para 9):

| variant | narrow | broad | paragraph | mean |
|---|---|---|---|---|
| baseline | 0.688 | 0.231 | 0.556 | 0.491 |
| A: relation-bridges | 0.688 | 0.231 | 0.556 | 0.491 (Δ0) |
| B: query-claims, routing only | 0.688 | 0.231 | **0.667** | 0.528 (+0.037) |
| **B + demand-side recompute** | 0.688 | **0.308** | **0.667** | **0.554** (+0.063) |

- **Routing alone** (query-claims as hubs) lifts **paragraph +0.111**, broad flat.
- **Demand-side re-anchor** — letting query-claims into the concept **medoid** (deliberately
  bypassing `primary_only`), so the centroid drifts toward the *question-shapes* that retrieve
  it — is what moved **broad +0.077**. This is the only thing in the thread that moved broad
  and is exactly the "geometry can't provide this; only queries can" lever. No regression.

## The catch — does it survive a real LLM answerer+judge?
The numbers above are the **cosine proxy**. Under the real harness (LLM answers from the
context, LLM judge checks all key-facts) on **broad**, run with **gemini** (Claude CLI hit its
subscription session limit):

| metric | baseline broad | +q-claims broad |
|---|---|---|
| cosine coverage proxy | 0.231 | 0.308 (+0.077) |
| real LLM answer+judge (gemini) | 0.077 | 0.077 (Δ0) |

**The broad gain did NOT appear under the LLM judge.** BUT this run is **inconclusive**: the
harness docstring requires a **sonnet-class** answerer/judge — a weak reader floors broad and
deflates SR@B, and gemini baseline broad = 1/13 is the floor signature. So gemini can't
distinguish "no real gain" from "judge too weak to see it." **Needs a Claude/sonnet rerun**
(`scratchpad/query_claims_llm.py [gold_file]`) once the subscription resets.
Leaky-vs-suppressed was **identical** under both metrics → the cosine gain was not leaked
question-text gaming the metric.

## Shipped (prod wiring, default OFF)
In prod the signal is **real retrieval queries** (logged as `RETRIEVAL_SIGNAL`), not the
synthetic stream (that was the eval-only gold-blindness device). Wiring:

- **`consolidate.py` step 6d** — `_inject_query_claims` (after channel-redundancy, concepts
  settled): reads the most-recent ≤`QUERY_CLAIM_MAX` distinct logged queries, navigates each
  (`resonance.activate`), and for queries spanning ≥`QUERY_CLAIM_MIN_CONCEPTS` concepts emits
  ONE `QUERY_INJECTED` event carrying the full set (replace-whole → idempotent, like
  `CHANNEL_REDUNDANCY`).
- **`QUERY_INJECTED` applier** — clears prior query-claims, inserts each as a `qclm_` claim
  (embedding recomputed from query text on replay, like `CANONICALIZED`), attaches to its
  top-K concepts as `kind='query'`, and — if `QUERY_CLAIM_REANCHOR` — re-medoids the touched
  concepts including the query-claims (`store.recompute_concept_embedding_demand`). Verified
  rebuild-correct (18 claims / 72 members → identical after `rebuild()`).
- **Flags** (`consolidate.py`): `INJECT_QUERY_CLAIMS=False` (master, step 6d),
  `QUERY_CLAIM_REANCHOR=False` (the broad lever — riskiest), `QUERY_CLAIM_ATTACH_K=4`,
  `QUERY_CLAIM_MIN_CONCEPTS=2`, `QUERY_CLAIM_MAX=200`.

Three isolation guarantees, all verified (leak into a real retrieval context = **False**):
1. **Answer results** — fragment path drops query-claims (no source episode); breadth frame
   `core/resonance.py:_concept_members` excludes `qclm_` (`NOT LIKE 'qclm\_%'`; measured
   20/38 → 0/38 gold contexts before/after).
2. **Concept-minting LLM** — `_consolidate_concepts` reads members `primary_only=True`;
   `kind='query'` is not primary, so query-claims never enter the minting prompt. Same guard
   protects the normal `recompute_concept_embedding` medoid.
3. **Channel-redundancy collision** — query-claims use `kind='query'`, not `'redundant'`, so
   `set_channel_redundancy`'s blanket delete of redundant rows never clobbers them.

Routing still works because the spreader reads `concept_members` via raw SQL
(`resonance.neighbours`), which is kind-agnostic.

## Open / next move (the gate to flip it ON)
The mechanism is shipped; **enabling it** (`INJECT_QUERY_CLAIMS=True`, then
`QUERY_CLAIM_REANCHOR=True`) waits on:
1. **Confirm or kill the broad gain on a sonnet-class judge** (the gemini run is inconclusive —
   gemini floors broad). Rerun `scratchpad/query_claims_llm.py <gold>` on Claude once the
   subscription resets. This is the gate — nothing flips on until it's green.
2. **Forgetting gate** before turning on `QUERY_CLAIM_REANCHOR` — it mutates concept medoids,
   so run the full SR@B before/after `compare()` and require forgetting_rate = 0.
3. **Stability** — broad +0.077 is ~1 query on n=13; show it holds across query-stream seeds +
   attach_k ∈ {3,5,6} before trusting it.
4. **Marker** — `qclm_` id prefix is the current marker; if it proves load-bearing long-term,
   promote to a `claims.origin='query'` column.

## Lab files (scratchpad/)
- `query_bridges.py` — reading A + the cached synthetic query generator (claude-cli forced).
- `query_claims.py` — reading B (inject + cosine eval + leak check against the core fix).
- `query_claims_llm.py [gold_file]` — real LLM answer+judge A/B (baseline/leaky/suppressed).
- `qbridge_qstream.json` — the cached 60-query synthetic stream (blind to gold).
