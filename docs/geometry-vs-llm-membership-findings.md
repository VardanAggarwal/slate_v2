# Can geometry replace LLM membership? — session findings (2026-06-26)

**Question.** Membership (which claim belongs to which concept) is the one consolidation
step still owned by the LLM. Centers/anchors feel geometrically solid; membership doesn't.
Can a **purely geometric** membership match or beat the LLM partition's retrieval quality?

**The bar (LLM membership, same write path, eval corpus `/tmp/slate_eval.db`, τ=0.54):**

| | narrow | broad | paragraph |
|---|---|---|---|
| **LLM partition** | **79%** | **33%** | **75%** |

Coverage@B = `eval/coverage.py`. Golds: narrow 20, broad **6 scored** (11 lines, loader keeps 6), paragraph 14. Reach@B is a noisier node-selection proxy. **Broad is a 6-query metric — 33→50 is one query; treat broad moves as hints, not results.**

---

## Verdict

**Geometry can match/beat the LLM on individual axes but cannot dominate it.** The best
fully-geometric pipeline (**density anchors + reconstruction-z + redundancy = 71/50/62**)
*beats* the LLM on broad (50 vs 33) but loses narrow (71<79) and paragraph (62<75). It's
**Pareto-incomparable**, not a win. The narrow/paragraph ceiling (~71/62) is the embedding's
**meaning limit** — the LLM reads negation/number/scope the embedding flattens, and no
clustering recovers what the features lack.

Three durable geometric wins to keep: **reconstruction-z membership beats proximity**
(narrow 57→71); **anchor strategy is decisive** (density >> outlier-seeking k-center, which
tanked broad to 0); **redundancy/multi-membership works** (broad 17→50) *with density
anchors*. What's still missing to *dominate* is **signal, not structure**: richer features
+ usage geometry. (Caveat: broad is a 6-query metric; broad 50 = +1 query vs the LLM.)

---

## What was measured (each on the same harness)

### 1. Layer attribution — which of the 3 geometries owns what
Isolated by perturbing one layer, holding the other two at a valid reference (LLM
membership + medoid/contrastive centers + bridges). `scratchpad/layer_bench.py`.

| layer varied | narrow | broad | paragraph | owns |
|---|---|---|---|---|
| **L2 center** centroid→contrastive | +7 (71→79) | 0 | +25 (50→75) | seed precision |
| **L3 bridges** none→50 @ boost 1–2 | 0 | 0 | ~0 | *nothing measurable* |
| **L1 membership** LLM→split | −7 | +17* | 0 | narrow↔broad dial |

\* broad 33→50 = one query; both finer (split) and coarser (em) trigger it → direction-agnostic, not a strategy. Earlier "split tanks broad (33→17)" was a **stale-bridge artifact**; with bridges rebuilt per-partition, split *lifts* broad. Membership is the only thing that moves broad, but the move is fragile.

### 2. Bridges are traversed but inert (`trace_bridges.py`)
Bridges are the highest-weighted edge (`BRIDGE_BOOST=1.3 > MEMBERSHIP_W=1.0`) and **are**
traversed (118 edge expansions over 6 broad queries) and shift 20–60 nodes' salience —
but the **top-12 materialized set is identical with vs without bridges on every query.**
The PE-gate damps flow to non-distinctive targets, and bridged concepts are non-distinctive
by construction. Net: bridges are dead weight for retrieval. To matter they'd have to act
at **seed/selection**, not via damped spread.

> **Correction (2026-06-27, `scratchpad/bridge_tune.py`):** the suppressor is NOT mainly the
> PE-gate — turning it off barely promotes bridge targets. It's the **distinctiveness term**
> (`1/(1+ln degree)^γ`): bridged concepts are low-distinctiveness, and the bridge edge raises
> target degree → lowers it further (self-defeating). Tuning retrieval to value bridges
> (BRIDGE_BOOST 1.3→6, gate off, wider materialize) makes broad **monotonically worse** —
> promoting a vaguer bridged concept evicts a precise covering claim. The suppression is
> deliberate (it protects precision). Same finding (inert/harmful), corrected mechanism.

### 3. Geometric retrieval hops refuted (`geo_hop.py`)
Live k-NN claim→claim adjacency at retrieval (the "members reach non-siblings"
realization). Every config: **narrow 79→64, broad flat 33, paragraph 75→50–62.** Even
sibling-excluded (pure cross-concept) hops left broad at 33. The PE-gate-favors-the-cross-jump
hope didn't translate; diffusion displaces the precise covering claim from top-12.

### 4. No intrinsic geometric KPI proxies coverage
- **Silhouette / Davies-Bouldin** (`struct_kpi.py`): ρ(silhouette, coverage) ≈ 0; **DB
  anti-correlates** (−0.36 narrow, −0.47 broad). All silhouettes negative — concepts aren't
  geometrically separable; the retrieval-best partition is the geometrically *worst*-separated.
- **Masked reconstruction** (`masked_recon.py`): frag-scale ρ = −0.57 narrow, concept-scale
  ρ = −0.61 broad — **anti-aligned with the gold type each was built for.** It's a
  **cluster-size detector** (collapse attractor): frag_recon rewards agglomeration, concept_recon
  rewards fragmentation. Confounded by rate.
- **Rate–distortion / VQ** (`rd_curve.py`): the decisive one. At equal rate K=326, geometric
  VQ has **lower distortion (0.476 < 0.547) but worse coverage (57/33/62 vs 79/33/75).** The
  LLM partition is **rate–distortion-suboptimal yet retrieval-superior.** Minimizing
  distortion walks *away* from retrieval. Information-theory framing: we kept doing **source
  coding** (compress, remove redundancy) when retrieval is a **channel-coding** problem
  (structured redundancy for robust recovery).

### 5. Reconstruction-z membership + redundancy vs the LLM bar (`recon_membership.py`)
Fully geometric: k-center anchors → reconstruction-z (PRD primitive, *not* proximity)
soft multi-membership, centers from primary members only.

| partition | narrow | broad | para | redundant attaches |
|---|---|---|---|---|
| LLM bar | 79% | 33% | 75% | 0 |
| geom single (reconstruction-z) | **71%** | 0% | 62% | 0 |
| geom multi θ0.5 | 71% | 0% | 62% | 1964 |
| geom multi θ1.0 | 71% | 17% | 50% | 3531 |
| geom multi θ1.5 | 64% | 17% | 50% | 5797 |

- **Reconstruction-z beats proximity:** narrow 71 vs VQ's 57 — half the gap closed by using
  the right primitive ("can the region rebuild it?" not "nearest center"). Durable win.
- With **k-center** anchors: redundancy looked refuted (broad stuck 0→17, paragraph dropped).
  **That was an anchor artifact** — k-center seeks outliers, pathological for frames.

### 5b. Anchor strategy is decisive — density anchors revive redundancy (`recon_density.py`)
Same pipeline, anchors = k-means medoids (dense/representative) instead of farthest-first.

| partition | narrow | broad | para | redundant |
|---|---|---|---|---|
| LLM bar | 79% | 33% | 75% | 0 |
| geom single, **k-center** | 71% | 0% | 62% | 0 |
| geom single, **density** | 57% | 17% | 62% | 0 |
| geom **multi θ1.0, density** | **71%** | **50%** | 62% | 3862 |

- **Anchors matter enormously.** k-center tanked broad to 0; density recovers it.
- **Redundancy works with density anchors:** broad 17→50, narrow 57→71. *Not* refuted — the
  earlier null was the k-center pathology. Geometric-neighbour redundancy *does* buy broad
  frame coverage when the anchors sit in dense regions.
- **Best fully-geometric = 71/50/62** — beats the LLM on broad (fragile, +1 query), loses
  narrow (71<79) and paragraph (62<75). Pareto-incomparable, not a win.

---

## Why (the through-line)

Meaning is **non-metric** relative to this embedding: the LLM groups embedding-far things
that mean the same and splits embedding-close things that don't. The rate–distortion result
is near-proof that any **distortion-minimizing** geometry (k-means/VQ/squish/EM) will never
select the meaning partition. Reconstruction-z is *not* distortion-minimizing, so it does
better (57→71) — but it's still bounded by the embedding's meaning content (negation, number,
scope — flattened; PRD's "honest limit"). You can't out-cluster a feature blindness.

**PRD alignment.** Membership = spread-relative reconstruction residual + anchor-present,
soft/overlapping, with the irreversible commit (same vs contradiction) handed to the resolver
(`docs/Slate PRD v2.md` §How, lines 179/207/208/215/235). Geometry is *necessary but not
sufficient*. Our results are the measurement of that line.

**Bio framing** (informative, not load-bearing): dentate-gyrus **pattern separation** = anchors;
CA3 **pattern completion** = reconstruction-membership; sparse **distributed (redundant)** codes
= multi-membership; **Hebbian replay** = usage. Maps onto "centers compress (source coding),
membership redundifies (channel coding)" — though our measurement shows geometric redundancy
alone doesn't deliver.

---

## What's left (the only unexhausted levers)

1. **Richer features** — a stronger / query-aware / asymmetric / LLM-*derived* embedding used
   as a **one-time featurizer** (still geometry, better sensor). Re-run the reconstruction-z
   pipeline on it; measure how much of the 71→79 / 62→75 gap better features close. **Highest
   leverage** — it attacks the root (feature meaning-blindness).
2. **Usage geometry** (Hebbian co-retrieval) — the only signal all session that tracked SR@B;
   PRD carves it out as a separate signal feeding consolidation (lines 200/237). Bootstraps
   from traffic.
3. **Broad is query-side, not structure** — R0 vocabulary-disjoint seeds. Lever is HyDE /
   query decomposition (additive, can't hurt narrow). **Prereq: expand `gold_broad` beyond 6
   queries** before optimizing broad at all.

**Keep:** reconstruction-z membership (narrow 57→71), **density anchors** (not k-center), and
**redundancy/multi-membership** (broad 17→50 with density anchors). **Drop:** distortion-min
partition (k-means/VQ), bridges as a retrieval lever, geometric hops, intrinsic geometric KPIs
(silhouette/DB/masked-recon/rate-distortion — all anti-aligned). Open ceiling on narrow/para
(71/62 < 79/75) is the embedding meaning limit → needs richer features, not more geometry.

---

## Reproduce — harness scripts (`scratchpad/`, run with `PYTHONPATH=.:scratchpad .venv/bin/python`)
Always `cp /tmp/slate_eval.db /tmp/slate_work.db` first. User `usr_01KTXAYR20J4R6F7PT3DP10W3W`.

- `eval_both.py` — Coverage@B + Reach@B, three golds.
- `layer_bench.py <l1|l2|l3>` — isolated per-layer attribution.
- `trace_bridges.py` — bridge traversal + field-diff (inertness proof).
- `geo_hop.py` — geometric retrieval-hop sweep.
- `struct_kpi.py` / `struct_kpi_corr.py` — silhouette/DB vs coverage.
- `masked_recon.py` / `masked_recon_corr.py` — masked-reconstruction vs coverage.
- `rd_curve.py` / `rd_curve_analyze.py` — rate–distortion curve vs coverage.
- `recon_membership.py` — reconstruction-z soft multi-membership vs LLM bar.
- `recon_density.py` — (A) density vs k-center anchors.
