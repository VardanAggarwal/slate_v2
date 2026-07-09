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

## Does it survive a real LLM answerer+judge? — YES (Opus-4.8 judge)
The cosine numbers above are a proxy. Confirmed against a real answer+judge on **broad**:

| answerer+judge | baseline broad | +q-claims broad | Δ |
|---|---|---|---|
| cosine coverage proxy | 0.231 | 0.308 | +0.077 |
| gemini (weak) | 0.077 | 0.077 | 0 — **floor artifact, inconclusive** |
| **Opus 4.8 (in-session)** | **0.077 (1/13)** | **0.231 (3/13)** | **+0.154, forgetting 0** |

The **gemini** run (Claude CLI was session-limited) floored both conditions at 1/13 — exactly
the weak-judge failure the harness docstring warns about (it needs a sonnet-class reader). So
it was re-judged with the **Opus-4.8 session itself** as answerer+judge (retrieval is local;
contexts emitted by `scratchpad/emit_contexts.py`, then judged in-conversation against the
pre-registered key_facts). Broad lifts **1/13 → 3/13, zero forgetting** (b_religion passes in
both). The two flips rest on verbatim cross-note text present in treatment, absent in baseline:
- **b_labour** ← "Economic Coercion of Gig Work" note (*"gig contract is legally voluntary but
  economically coercive"*) — baseline had no gig content (3/4 → 4/4).
- **b_community** ← "Future Farming Practices" note (*"peer-driven hyperlocal communities for
  inputs, knowledge, and market access"*) — baseline missed the hyperlocal-farmer fact (3/4 → 4/4).

Exactly the mechanism's claim: a query-claim bridged a far note into the query's lit region.
Caveats: judged un-blinded (but flips are literal text diffs, verifiable); broad only (narrow/
paragraph real-judge forgetting check NOT run); n=13, 2-query flips; re-anchor variant + cached
synthetic queries. Leaky-vs-suppressed was identical under every metric → the gain is structural,
not leaked-text gaming.

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

## 2026-07-09 — INJECT enabled; re-anchor KILLED by the traversal findings
`INJECT_QUERY_CLAIMS=True`; `QUERY_CLAIM_REANCHOR=False`.

**Third strike (2026-07-09, shipped-path E2E — `scratchpad/flywheel_e2e.py` /
`flywheel_e2e_reanchor.py`):** on the prepped corpus, 60 REAL logged queries (usage simulated
through `resonance_context(signals=True)`, injected via the production QUERY_INJECTED
applier): reanchor OFF = coverage byte-identical to baseline (.750/.210/.889, zero
forgetting); reanchor ON = **narrow .750→.688, broad/paragraph unmoved**. Each instrument sees
a different axis pay (walked queries: paragraph; judge: broad; signal-log queries: narrow) —
the drift corrupts whatever geometry sits near the touched concepts.

**Re-anchor is superseded: docs/broad-lift-traversal-findings.md (2026-07-08) measured it
REGRESSIVE on the corrected instrument** — 19-probe extended broad gold, per-copy corpus prep
(stale relations cleared + 6a/6b/6c), qclm dead-slot fix in place. Reanchor ON: broad
.210→.158, paragraph .889→.667 (narrow untouched there). The in-session Opus judge agreed
(.158→.105, b_community displacement). The June-27 judge win below and the 2026-07-09
stale-lab re-runs (`scratchpad/reanchor_ab.json` — old synthetic stream, unprepped corpus
copy, baseline .688/.158/.556 vs the doc's .750/.210/.889) predate/miss that prep and do not
survive the re-test. Oracle seeding caps broad at ~.263 regardless — no headroom for a
retrieval-side lever. Verdict: reanchor stays OFF permanently, not as safety margin.

**Inject-only (routing) is coverage-NEUTRAL** (A_off exactly baseline in the traversal matrix;
same in the stale A/B). It stays ON as the demand-capture flywheel bet: in prod the qclm hubs
encode REAL logged queries (a distribution no eval has tested), the mechanism is rebuild-safe,
guarded (3 leak checks + dead-slot fix), and one flag reverts it.

Gate runs from earlier on 2026-07-09 (stale instrument — kept for the record, but superseded
by the traversal findings above):
1. **Forgetting gate on narrow + paragraph — PASSED, zero real regressions.** emit-contexts on
   `gold.jsonl` (16q) + `gold_paragraph.jsonl` (9q), re-anchor ON. Mechanical per-fact cosine
   pre-pass (`scratchpad/forgetting_gate.py`) flagged 7 candidate drops (g07, g12, g15×3,
   p05×2); each was judged in-session (Opus-class) against the actual context pair. All 7 were
   the same benign mechanism: re-anchor reshuffles which Background concept-frames render, but
   the flagged facts survive **verbatim in Specifics** (BNPL note, gig-coercion note, symbiosis
   paragraph). One marginal: g15 "function rather than structure" lost its explicit Background
   bullet but remains derivable from the retained "Black box approach" note.
2. **Stability — PASSED.** `scratchpad/qclaim_stability.py`: attach_k ∈ {3,4,5,6} on the full
   cached 60-query stream + four 30-query composition subsets (first/last half, even/odd) at
   k=4. Broad lifts 0.158 → 0.210 in **every** cell; narrow (0.688) and paragraph (0.556)
   pinned at baseline throughout. Disjoint halves both produce the lift → not a single-query
   artifact. Numbers: `scratchpad/qclaim_stability.json`. Caveat: seed-variation proxied by
   stream-composition subsets (regenerating streams with fresh seeds needs LLM spend); identical
   numbers across subsets say the lift rides on a robust minority of cross-concept queries.
3. **Marker** (open, non-blocking) — `qclm_` id prefix is the current marker; if it proves
   load-bearing long-term, promote to a `claims.origin='query'` column.

## Lab files (scratchpad/)
- `query_bridges.py` — reading A + the cached synthetic query generator (claude-cli forced).
- `query_claims.py` — reading B (inject + cosine eval + leak check against the core fix).
- `query_claims_llm.py [gold_file]` — real LLM answer+judge A/B (baseline/leaky/suppressed).
- `qbridge_qstream.json` — the cached 60-query synthetic stream (blind to gold).
