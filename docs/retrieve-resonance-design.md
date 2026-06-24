# Retrieve: Resonance (navigation by activation) — design

A new retrieval pathway, to be tested head-to-head against the existing five
(grep / concept / fragment / hybrid / hierarchical). **Retrieval-only**: it reads
the graph consolidation already builds and changes nothing upstream.

## Thesis (from the critical note)

Everything looks close to everything from 50k ft, so cosine-knn over a flat pool
saturates. Reframe retrieval as **navigation of a connected multi-layer cluster**:
a long query is split into probes; each probe *lights up* different parts of the
graph; the parts that light up **most** are interesting either because (a) one probe
matches them strongly, or (b) **many different probes converge there**. Then walk
outward a few hops. The hard rule: **exploration without de-noising returns only
noise** — so every spread step is gated by distinctiveness.

Two mechanisms the current `recall.py` is missing and this path adds:
1. **Confluence** — activation **sums** across probes (today `recall.py:68` takes
   `max`, discarding convergence). A node reached by five probes must outrank one
   reached by one.
2. **De-noising** — a generic hub lights up for *any* query, so confluence alone
   can't tell a real query-hub from a corpus-hub. Three denoisers separate them.

## The graph it navigates (already exists)

- **L0 episodes** (raw notes, verbatim) — `store`, immutable.
- **L1 fragments** (verbatim spans) — what assembly returns; provenance → episode.
- **L2 claims** (distilled) — `clm_`, `claim_support`, `claims_for_episode`.
- **L3 concepts** (clusters w/ medoids) — `cpt_`, `concept_members`.
- **Edges** — `relations(from,to,relation,weight)`: leads_to / supports /
  contradicts / **bridges**; plus membership (claim↔concept) and provenance
  (fragment↔episode↔claim).

Navigation happens on L2+L3 (the indexed layer); the answer is *materialised* from
L1 (verbatim, for fidelity).

## Flow

```
query
  │  R1 decompose (existing): sequential residual scan → probes p1..pn
  ▼
[probes]  each p_i carries its own embedding e_i and a probe-id
  │  multi-source seed: knn(e_i) over claims+concepts, per probe
  ▼
[lit nodes]  act[node] = { probe_i : sim }   ← keep probe identity, do NOT collapse
  │  SPREAD ×H hops:  sum across probes, fan-out normalised, PE-gated conductance
  ▼
[activation field]  Strength, Confluence, Distinctiveness per node
  │  salience = Strength · f(Confluence) · Distinctiveness
  │  VOI frontier stop: halt when a hop adds no distinctive salience
  ▼
[bright regions]  top nodes = "the parts that lit up most"
  │  materialise: bright claim/concept → its verbatim fragments (L1)
  │  assemble (existing max-marginal-residual) weighted by salience, fit B
  ▼
[context]  + R8 signals (lit / dropped / confluence) → consolidation
```

### Stage 0 — Decompose into probes
Reuse R1 (`_seed_pool(decompose=True)`): split the query into independent
sub-queries by the same sequential residual scan Write uses for chunking. Each
probe is an **independent injection site**; we keep which probe lit which node —
that identity *is* the confluence signal.

### Stage 1 — Multi-source seed
For each probe `p_i`, knn over claims **and** concepts. Store activation as a
**per-probe map** `act[node][i] = sim(e_i, node)` for `sim > seed_floor`. No `max`,
no collapse yet — convergence is computed later from the map.

### Stage 2 — Spread (the core change)
Hop over edges. Per source node `u` carrying activation to neighbour `v` via edge
weight `w`:

```
delivered_i = act[u][i] · w · conductance(u,v) / fanout(u)
act[v][i]  += delivered_i          # SUM, per probe — confluence accrues
```

- **SUM, not max** — many probes converging on `v` add up. This is mechanism (1).
- **fanout(u)** = normalise by `u`'s degree (or √degree). A node connected to
  everything cannot broadcast its activation everywhere — denoiser #1 (kills the
  *source*-side flood).
- **conductance(u,v)** = the PE gate. Flow only where `v` is *distinctive given what
  is already lit* — residual of `v` against the active set. ≈0 if `v` merely
  restates already-bright content (adds nothing) → denoiser #2. This is the
  fragment path's max-marginal-residual logic applied **along edges** instead of
  over a flat pool — and it is also what gives true multi-hop (point 4 in review).

### Stage 3 — Score the field
For each node:
- **Strength(node)** = Σ_i act[node][i]  — total activation (close-match *or* sum).
- **Confluence(node)** = #{ i : act[node][i] > 0 } — how many *distinct* probes reached it.
- **Distinctiveness(node)** = corpus-level residual / inverse generality — high for a
  node that is *not* close to everything; low for a background hub. Reuse the
  `baselines` table (C12 member-residual) or an IDF-style inverse-degree prior.
  **Denoiser #3** — the one that separates a real query-hub (distinctive, lit by
  many probes) from a corpus-hub (generic, also lit by many probes but predictable).

```
salience(node) = Strength · (1 + log Confluence) · Distinctiveness
```

The `log Confluence` term is what makes "too many signals reach here" win; the
`Distinctiveness` factor is what stops it from being gamed by generic hubs.

### Stage 4 — Stop (VOI on the frontier)
After each hop, sum the **new** salience landing on distinctive nodes. Stop when it
falls below `ε · salience_so_far`, or at `H_max` hops (≤3). Continue pulling the
thread only while the next hop is expected to add more than it costs — the PRD's
retrieval stop rule, applied to graph traversal rather than a candidate list.

### Stage 5 — Materialise to budget
Navigation chose *where*; now choose *what text*. For each bright node, pull its
backing **verbatim fragments** (claim → source fragment/episode; concept → member
claims → fragments). Run the existing `assembly.assemble` over those fragments,
**weighted by parent-node salience** (replacing the flat cosine relevance), so
max-marginal-residual de-dups and fits `B`. This is how the synthesis reach of the
concept path and the fidelity of the fragment path come from **one walk** instead of
hybrid's blind concatenation.

### Stage 6 — Signals
Emit R8 (`record_retrieval_signal`) with lit / dropped / confluence per node, from
**this** path (closing the loop the concept/hybrid paths leave open).

## What it reuses vs what's new

| Reuse (unchanged) | New (retrieval-side only) |
|---|---|
| graph: concepts/claims/members/relations/bridges | per-probe activation map (sum + confluence) |
| `knn_claims` / `knn_concepts` seeders | fan-out-normalised, PE-gated spread |
| `predict.residuals_against` (conductance, distinctiveness) | VOI frontier stop |
| `baselines` table (distinctiveness prior) | bright-node → fragment materialisation |
| `assembly.assemble` (max-marginal-residual) | salience-weighted assembly input |
| `record_retrieval_signal` (R8) | — |

No consolidation, Write, or schema change. It runs on today's DB.

## How it differs from the existing five

| path | seeding | traversal | salience | fidelity |
|---|---|---|---|---|
| grep | lexical | none | FTS rank | verbatim (1 note) |
| concept | cosine knn | fixed 2-hop, `max`, `SPREAD_*` | activation·strength | distilled (lossy) |
| fragment | cosine knn | none (flat pool) | max-marginal-residual | verbatim |
| hybrid | both | concept leg only | two readers concat | mixed |
| hierarchical | concept frame + frag | OFF bridge-walk | frame + VOI nuance | mixed |
| **resonance** | **per-probe multi-source** | **summed, PE-gated, VOI-stopped hops** | **Strength·Confluence·Distinctiveness** | **verbatim (materialised from bright nodes)** |

It is the concept path's successor: same graph, but **sum not max**, **PE-gated not
fixed-decay**, **decomposed not single-seed**, **stops by VOI**, and **answers in
verbatim** instead of distilled claims.

## Test plan (against existing pathways)

- Same harness: frozen-14 (narrow) + broad probe, `B=2000` and `B/2`, sonnet
  answerer+judge, eval DB. Add a `resonance` model alongside the five.
- **Primary hypothesis (broad):** confluence should beat the 33% hybrid/hier ceiling
  on "summarise/list all my X" — that is exactly the many-probes-converge regime.
- **Guard (narrow):** must not regress fragment's depth wins — the materialisation
  step (verbatim, salience-weighted) is there to protect this.
- **Ablations** (isolate the mechanism): `max` vs `sum`; distinctiveness on/off;
  PE-gate vs fixed-decay; hops 0/1/2/3. The note predicts: sum > max, and
  distinctiveness-off floods (confirms "exploration without de-noising = noise").

## Knobs (calibration-owned, fit later)
`seed_floor`, `H_max`, fan-out exponent, conductance PE-band, distinctiveness
weight, VOI `ε`, confluence base. All operating points on the predictor — none
change the sensor.

## Measured results (2026-06-24, eval DB `/tmp/slate_eval.db`, sonnet answerer+judge)

Prototype: `core/resonance.py`, wired as the 6th harness model. B=2000.

| config (frame ON throughout) | narrow (14) | tail (5) | broad (6) |
|---|---|---|---|
| plain — fan-out spread only | 64.3% | 20% | 33.3% |
| **+ PE-gate + distinctiveness, `max` agg** | **78.6%** | 40% | 33.3% |
| + confluence (`sum` agg) | 71.4% | 40% | 33.3% |
| *(ref) hybrid / hier / grep* | 64 / 57 / 86% | 60 / 40 / 100% | 33 / 33 / 17% |

**Winning config = frame + PE-gated navigation + `max` aggregation: 78.6% narrow /
33.3% broad** — the best Slate path on BOTH axes (beats hybrid 64/33 and hier 57/33).
Still under grep on narrow (86%) and tail (grep 100% vs 40%) — the hard single-note
verbatim-recall queries (g05/g11/g12) remain grep's domain.

What the ablations attribute:
- **The de-noisers are the lever** (+14.3pts narrow, tail 20→40). The critical note's
  "exploration without de-noising returns noise" is *validated*: a PE-gated,
  distinctiveness-weighted graph walk beats a plain fan-out spread.
- **Confluence (sum-across-probes) is REFUTED on this gold** (−7.2pts vs `max`). On
  single-sentence factual queries, summation rewards convergence on tangential nodes.
  Its intended regime — paragraph-length queries where convergence genuinely signals
  the answer — is **not represented in the current gold**, so the mechanism is
  untested in its native habitat, and where it *can* act (clause-split probes) it
  hurts. `res_sum_probes=False` by recommendation until paragraph-query gold exists.
- **The distilled concept-frame is necessary**: verbatim-only materialisation scored
  broad **0%**; adding the frame lifted it to 33% and narrow 64→71/79. So Stage-5
  must emit bright *concepts* as distilled breadth, not just bright claims as spans.
- **Broad is pinned at 33.3% under every retrieval knob** — confirming the notes'
  claim that broad is a **consolidation-completeness** ceiling: b_work/b_ideas stay
  0/4 because the user's employers/specific ventures are not concepts, so navigation
  has nothing on-topic to frame (it pulls distinctive-but-wrong topic concepts). No
  retrieval mechanism lifts this; consolidation must mint those clusters first.

Net: resonance is worth keeping as the best Slate path, but the headline gain is the
**PE-gated navigation + concept-frame**, not confluence.

### Confluence in its native regime — REFUTED (paragraph gold, `eval/gold_paragraph.jsonl`)

Built an 8-query paragraph gold: each query is a 3-sentence paragraph converging on
ONE answer region (key_facts reused verbatim from `gold.jsonl` → controlled
sum-vs-max). Real sentence boundaries give 4 probes/query, confluence up to 4 — the
critical note's exact "long paragraph, each sentence lights up a part" regime.

| aggregation | SR@B | tail |
|---|---|---|
| `max` (confluence OFF, recall.py behaviour) | **87.5%** | **75%** |
| `sum` (confluence ON) | 62.5% | 50% |

**`max` wins even here.** Summing across sentence-probes conflates *genuine*
convergence with *shared-generic-vocabulary* grazing: a paragraph's sentences share
filler ("I keep thinking…", "it seems…", "the system…"), so generic hub nodes get
hit by all 4 probes and their summed activation buries the one sharply-relevant node
a single probe hits hard. The inverse-degree distinctiveness prior dampens but does
not cancel it. `max` is immune — each node scores on its single best probe. Evidence
(p05): `sum` used MORE context (1972 vs 1793 tok) yet covered FEWER facts (1/3 vs
3/3) — the over-injection/dilution signature. **Recommendation: `res_sum_probes=False`
permanently.** True confluence would need probe *independence* (orthogonalised
sentences) and convergence weighted by that independence — a much harder mechanism
than summation, and not worth it on this evidence.

## Open risks
- **R0 still bites the seed.** If no probe seeds near the answer node, navigation
  can't reach it. Confluence + hops *widen* reach vs flat knn, but a fully
  vocabulary-disjoint note stays unreachable until consolidation lays a usage edge.
- **Distinctiveness prior quality.** If the `baselines`/IDF prior is noisy, denoiser
  #3 misfires. Start with inverse-degree (cheap, robust), upgrade to corpus-residual.
- **Fan-out vs confluence tension.** Over-damping fan-out also damps legitimate
  convergence. The ablation on the fan-out exponent settles this.
