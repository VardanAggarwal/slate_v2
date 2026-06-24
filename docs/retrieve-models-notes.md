# Retrieve models — salience & limitations

Grounded in the measured SR@B runs (2026-06-24, eval DB `/tmp/slate_eval.db`, 166-note
corpus, sonnet answerer+judge). Two benchmarks: the **frozen-14** (narrow, single-note
factual queries — `eval/gold.jsonl`) and the **broad probe** (6 synthesis-across-many-
notes queries — `eval/gold_broad.jsonl`). They probe opposite regimes; read both columns.

## SR@B @ B=2000 (the scoreboard)

| model | frozen-14 | frozen tail | frozen B/2 | broad |
|---|---|---|---|---|
| grep (FTS) | **86%** | **100%** | **71%** | 17% |
| concept (spreading activation) | 50% | 40% | 43% | 0% |
| fragment (verbatim VOI) | 57% | 40% | 50% | 17% |
| hybrid (concept+fragment split) | 64% | 60% | 43% | **33%** |
| hierarchical (bg frame + nuance) | 57% | 40% | 36%¹ | **33%** |

¹ hier B/2 is its weakest cell — see its limitations. With the bridge-walk ON it
drops further (50%/tail 20%); the walk is OFF by default for this reason.

---

## grep (FTS over raw episodes) — the competitor baseline

**Salience.** Unbeatable when the answer lives *verbatim in one self-contained note*
and the query shares the note's vocabulary. Dominates the frozen-14 (86%, tail 100%)
because that gold is, by construction, single-note factual recall. Zero build cost, no
distillation loss — it hands the LLM the raw note.

**Limitations.** Collapses on synthesis (17% broad). Lexical, so it can't (a) gather
evidence spread across many notes, (b) match a query whose words differ from the note's
("crony capitalism" misses the econ-jargon "growth-welfare" note), or (c) fit >budget
notes — it truncates to the top FTS matches. No notion of theme, recency, or contest.

---

## concept — spreading activation over the claims/concepts graph (`recall.recall`)

**Salience.** The only path that does *cross-note synthesis under a theme*: knn-seeds
concepts/claims, spreads 2 hops over membership + bridge edges, surfaces the distilled
canonical with provenance and the ⚖️/🌉/🌫️ signals (contested, bridged, background).
Strong when the answer is "what's my consolidated view on X" and X is a real concept.

**Limitations.** Lowest scorer on both benchmarks (50% narrow, **0% broad**). Distilled
claims are lossy — they drop the verbatim reasoning chain a depth query needs (it led
with a *qualifier* on g05 and missed). On broad 4-fact queries the distillation can't
carry all four specifics. Mechanism is pre-predictor: fixed cosine + `SPREAD_*` decay +
hard `k`, no VOI stop, no decomposition. Its retrieval emits **no R8 signals**, so
consolidation never learns from it.

---

## fragment — verbatim spans via the predictor VOI assembly (`core/retrieve.py`)

**Salience.** Highest-fidelity Slate path (write-eval 9.1/10 reconstruction): returns
verbatim raw spans, so the causal reasoning survives intact — wins depth queries where
one note holds a connected argument (beat concept on g02/g05/g08). Predictor-native:
max-marginal-residual picks unique-info-per-token and *stops* at saturation. Best single
Slate path on narrow at B/2 (50%) because compact verbatim spans don't truncate.

**Limitations.** No *background* — it jumps straight to leaf fragments, so it can't frame
or synthesize across a theme (loses the cross-note arc on g04/g11). On broad queries the
relevance signal is uniformly weak (~0.3, the R0 limit), so without a relevance gate the
VOI loop pads the budget with novel-but-tangential spans (24 noisy fragments → dilution).
Verbatim spans are long, so at tight budget the tail craters (B/2 tail 20%).

---

## hybrid — concept + fragment, fixed `concept_share=0.4` budget split (`core/hybrid.py`)

**Salience.** Best overall (64% narrow, tail 60%, **33% broad**). The two single paths
are *complementary, not redundant* — they miss different queries — so blending captures
fragment's fidelity wins AND concept's synthesis wins. The richest narrow performer
because its concept leg carries claims grouped under concepts with provenance (the
formatting, more than the graph hops, is what gives it the edge).

**Limitations.** The split is a *fixed knob*, not query-aware — at B/2 it starves both
legs below their working threshold (43%, *below* fragment-alone's 50%). Runs two readers
independently and concatenates; no shared residual, so a fragment can repeat what the
concept leg already said. Inherits the concept leg's no-R8-signals gap.

---

## hierarchical — background concept frame + verbatim nuance (`core/hierarchical.py`, new)

**Salience.** The first-principles design: a query = BACKGROUND (nearest concept(s) +
their member claims, the compressed frame) + NUANCE (verbatim fragments via VOI). The
concept/fragment split is *emergent* from the residual, not a fixed knob. Matches hybrid
on broad (33%, 2× fragment-alone) — the background frame supplies the breadth a
"summarise/list all my X" query needs. The relevance-gated nuance (`rel_keep_frac=0.5`,
`value_floor`) kills the broad-query over-injection that sinks fragment-alone.

**Limitations.** Doesn't beat hybrid yet: ties on broad, ~1 query behind on narrow
(57%), and weakest at B/2 (36%) — the background frame's budget cap + the nuance gates
are mis-tuned at tight budget, leaving too little room for either leg. Its quality is
bounded by **consolidation granularity**: when no concept frames the query (e.g. "my
employers" isn't a concept), the frame is generic PM-skill clusters and adds little. The
**bridge-walk** built to replace `SPREAD_*` is net-negative (dilutes narrow depth
answers, no broad gain) — OFF by default.

---

## Cross-cutting ceilings (cap every Slate variant, not fixable in assembly)

- **R0 encoder asymmetry — a 24% tail, not a wall.** Median answer-note rank is 8; only
  5/21 broad answer-notes sit beyond the top-40 seed, all *vocabulary-mismatch* cases.
  HyDE (hypothetical-doc embedding) recovers ~1/5; the genuinely cross-domain notes stay
  unreachable by both encoder and graph.
- **Consolidation completeness.** The real broad ceiling: some required facts have *no
  concept* and the graph has *no edge* to the fact-note, so neither knn nor a bridge-walk
  can reach them. Lifting this lifts *every* variant — higher leverage than retrieval
  tuning.
- **Budget realism.** The broad gold asks 4 strict facts within B=2000; even the union
  of all systems clears only 2/6. Part of the "ceiling" is simply a hard bar.

## When to use which (routing guidance)

- **Narrow / factual, vocabulary known** → grep is the floor to beat; hybrid is the best
  Slate answer.
- **Broad / aggregation / "summarise my …"** → hierarchical or hybrid (2× grep/fragment).
- **"What's my consolidated view, with contest/versioning"** → concept leg (the ⚖️/🌉
  signals only it surfaces).
- **Depth on one note, fidelity-critical** → fragment (verbatim, no distillation loss).
