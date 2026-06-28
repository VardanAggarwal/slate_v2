# Write-workflow eval — compression & meaning preservation (v3 vs v2)

**Corpus:** eval user `usr_01KTXAYR20J4R6F7PT3DP10W3W`, 166 real notes, 139,251 input tokens (chars/4).
**Method:** run on a **copy** of `data/engine.db`; v3 fragments materialized via `write.refine_pending` (local MiniLM embedder, stance stubbed — direction doesn't affect token counts). v2 layer was already consolidated (2,040 claims, 326 concepts).
**Scripts:** `scratchpad/compress_eval.py`, `scratchpad/fidelity_eval.py`.

## Headline

| Workflow | What it commits | Committed/Input | Reconstruction fidelity (1–10) |
|---|---|---|---|
| **v3 write** | verbatim fragment spans (NOVEL+AMBIGUOUS; PREDICTED dropped) | **0.96** | **9.1** |
| **v2 consolidate** | paraphrased canonical claims | **0.24** (0.37 with concepts) | **6.25** |

They are **not competing implementations of one operation** — they sit at opposite ends of the compression↔fidelity frontier and run at different times.

## Compression detail (v3)

- 995 fragments stored: **103 NOVEL, 892 AMBIGUOUS** (90%); **37 PREDICTED dropped**, 6 intra-note echoes.
- Per-note ratio: median 0.969, mean 0.952, min 0.523, max 1.0.
- **Why ~no compression:** `predict.decide()` only drops (PREDICTED) when `z ≤ z_echo = −3.5`, i.e. a fragment reconstructs *as tightly as the region reconstructs itself* — a near-verbatim restatement. This corpus has little such redundancy, so almost nothing is dropped. This is **by design**: `predict.py` states the compression boundary "depends on the future use Q … owned and fitted at CONSOLIDATION." **Write is an online router / redundancy filter; consolidation is the compressor.**

### z_echo is the compression dial — REAL causal sweep

Re-running `write.route_fragments` over the whole corpus in ingest order with a growing
memory pool (`scratchpad/zsweep.py`; reproduces the production `refine_pending` result
exactly at the default: 0.958 / 37 dropped / 103 NOVEL / 892 AMBIG):

| z_echo | committed/input | spans dropped |
|---|---|---|
| **−3.5** (default) | **0.958** | 37 |
| −2.0 | 0.801 | 158 |
| **−1.0** | **0.209** | 675 |
| −0.5 | 0.145 | 792 |
| 0.0 | 0.099 | 879 |
| +0.5 | 0.033 | 973 |
| +1.0 | 0.018 | 1013 |

**v3 is strongly tunable.** At `z_echo ≈ −1.0` it commits **0.21 — already below v2's 0.24** —
purely by dropping redundant verbatim spans (no abstraction). 5–10× compression (→0.10) is
one knob away. The curve is **steep in [−2, −1]**, a sensitive calibration region.

> Correction: an earlier stored-z *proxy* estimated `z_echo ≈ +0.1` to match v2 and ~28%
> drop at −1.0. That ignored the routing feedback loop (a dropped span leaves the memory
> later spans are measured against) and badly **understated** tunability. The causal re-run
> above is the figure of record. Monotonicity of the dial is locked by
> `test_z_echo_is_the_compression_dial`.

**Two other levers do NOT reduce committed bytes:**
- *Widen memory M to include canonical claims* (the `write._memory_pool` seam): 0.958 → 0.964 — negligible; paraphrased claims don't embed close enough to push fragments past the strict bar.
- *prox_margin*: zero effect on committed tokens — only reshuffles NOVEL↔AMBIGUOUS, i.e. controls **LLM resolver volume/cost**, not compression.

**Caveat:** this is *selection* compression (drop whole verbatim spans), not abstraction. Below
the default bar, genuinely-novel-but-on-topic spans get dropped, so recall/fidelity falls. The raw
episode is retained either way (nothing is permanently lost), so the right setting is a bet against
SR@B — which consolidation is the layer meant to fit.

### Fidelity ↔ compression frontier (same 8 notes, same judge; `scratchpad/fidelity_zsweep.py`)

| Setting | committed/input | per-note recon fidelity |
|---|---|---|
| v2 claims (abstraction) | 0.24 | **6.25** |
| v3 z=−3.5 (default) | 0.96 | **9.12** |
| v3 z=−1.5 | 0.44 | **5.12** (0/8 fully dropped) |
| v3 z=−1.0 | 0.21 | **3.86** (1/8 fully dropped) |

**v3's meaning-per-byte collapses faster than v2's.** Pushed to v2-like compression, v3 is *worse*
than v2: at z=−1.5 (0.44, ~2× more bytes than v2) fidelity is already 5.1 < v2's 6.25; at z=−1.0
(0.21, beats v2 on bytes) it's 3.9. **In the compressed regime v2's abstraction Pareto-dominates
v3's selection** — rewriting to claims condenses; dropping spans can only keep-or-discard.

**Load-bearing caveat:** this per-note measure is a **lower bound for v3**. A span dropped at
aggressive z_echo is dropped *because it reconstructs from an anchor in another note* — still
retrievable at recall, but invisible to a per-note reconstruction that sees only this note's kept
fragments (e.g. at z=−1.0 one sample note kept 0 fragments — fully redundant-with-memory — scored
None, yet its content lives in the notes it echoed). v2 claims carry no such discount. So the
frontier is biased **against** v3 exactly where it compresses hardest. **The decisive metric is
SR@B** (whole-store recall, anchors included), not per-note reconstruction. n=8, LLM-judged, high
per-note variance.

## Meaning preservation

Reconstruct each note from its committed layer, LLM-judge fidelity vs the raw original (n=8, 138–1686 tok):
- **v3 = 9.1/10** — keeps ~96% verbatim, so reconstruction is near-lossless.
- **v2 = 6.25/10** — 4× abstraction loses distinct points (e.g. note `ep_01KTWXZD…`: 11 claims flatten "I fought hard for the UX but never measured user impact" into a generic claim; judge scored 4/10).

## Cost note

90% AMBIGUOUS ⇒ ~892 **stance-classifier** (`resolve_direction` → `classify_stance`) calls for 166
notes — NOT generative LLM completions. Per `STANCE_PROVIDER`: local CrossEncoder (`nli`, default),
HF zero-shot API (`hf`, prod), or one Haiku call (`haiku`, the only generative case). v2-write uses
the *same* classifier (`encode.py:179`, once per echoing sentence); v3 just fires it more often. No
generative LLM runs at write in either layer — v2's LLM cost is entirely offline at consolidation.
Still, the volume matches the calibration concern in memory (`slate-pe-threshold-calibration`): the
AMBIGUOUS band is over-subscribed under default calibration. Tuning `z_echo`/`prox_margin` at
consolidation reduces both resolver volume and stored bytes.

## Tests

- `tests/test_write.py`: **16/16 pass** (added `test_route_all_predicted_note_stores_nothing`, `test_z_echo_is_the_compression_dial`).
- Methodology checks: chars/4 cancels in the ratio (no tokenization bias; exact for v3 since fragments are substrings of raw); claim dedup factor 1.00× (no double-count); raw episode retained by both layers (correctly excluded from both).

## v3.1 — medoid representation + batched routing + per-cluster knob

Three changes (same 166-note corpus, default `z_echo=−3.5`):

1. **Fragment vector = MEDOID sentence**, not the re-embedded joined span. Selected
   via the spine (`measure(span sentences, exclude_self)` → lowest-residual sentence).
   Reuses encode-time sentence vectors, so **refine makes ZERO embedding calls** (was a
   second HF round-trip per note). Selection, not generation (PRD-faithful). Rebuild
   recomputes the medoid deterministically from the episode's sentence vectors.
2. **Batched routing.** One `measure(all fragments, memory)` gemm vs the fixed corpus,
   replacing the per-fragment `measure(growing-Y)` loop that re-`vstack`-ed the whole
   pool each iteration. Intra-note dedup is now a cheap causal sibling check; the
   union-z is recomputed only for a fragment an earlier sibling actually competes with.
3. **Per-cluster `z_echo`/`prox_margin` made reachable.** Fragments now carry a
   `cluster` (anchor-inheritance bootstrap: NOVEL opens a region, AMBIGUOUS inherits its
   anchor's); `decide()`'s per-region threshold (already coded, previously dead because
   every fragment was `cluster=None`) now resolves. Consolidation re-clusters later.

| metric | re-embed (v3) | medoid+batched (v3.1) |
|---|---|---|
| total fragments | 1032 | 1032 *(scan unchanged)* |
| stored | 995 | 972 |
| dropped (PREDICTED) | 37 | 60 |
| intra-note echoes | 6 | **22 (3.7×)** |
| NOVEL | 103 | 74 |
| regions (clusters) | 0 | 74 |

Medoid sentence vectors match siblings more sharply than re-embedded spans → intra-note
dedup 3.7×, more PREDICTED-drops, fewer NOVEL. `committed/total` stays 0.94 (write is a
redundancy filter; compression is consolidation's). 166 notes refined in 18s, no network.
Tests: `tests/test_write.py` 19/19 (added `test_fragment_medoid_selects_a_real_sentence`,
`test_refine_makes_no_embedding_call`, `test_per_cluster_threshold_decides_predicted_vs_ambiguous`).

> **Blocker for the calibration loop (unchanged by the above):** fragments are still an
> **orphan branch** — `recall`/`assemble_context` read claims/concepts, and `consolidate`
> blueprints from `raw_text`; neither reads fragments. So `z_echo` (and the per-cluster
> knob) cannot move SR@B yet. Wiring fragments into a consumer is the prerequisite before
> a `z_echo`-vs-SR@B fit means anything.

## v3.2 — references instead of duplicate storage

Two redundancies removed; the fragment now stores only what it *adds*, referencing the
immutable episode for the rest:

1. **No fragment vector is stored.** A fragment's vector IS its medoid *sentence's*
   vector, already in `vec_sentences`. The `vec_fragments` table is dropped; the
   fragment row carries a `medoid_idx`, and `fragment_pool`/`knn_fragments` resolve the
   vector by referencing `vec_sentences` via `(episode_id, medoid_idx)`. Kills ~`N_frag
   × EMBED_DIM × 4B` of byte-identical vectors (≈1.5MB on the 972-fragment eval corpus).
2. **No fragment text in the FRAGMENTED event.** The payload carries only spans +
   `medoid_idx` + routing metadata; `apply_fragmented` reconstructs the text from
   `(sent_start, sent_end)` against the episode's sentences. Removes a verbatim copy of
   ~96% of the corpus from the event log.

Net: the working layer's per-note footprint drops from ~3–4× raw to the metadata + the
verbatim `fragments.text` (the one denormalized copy kept for read-speed). No behaviour
change — routing, weights, centre/peak are identical; `tests/test_write.py` 19/19.
**Upgrade path:** deploy then `rebuild` — legacy FRAGMENTED events (text, no `medoid_idx`)
are honoured by the applier, which re-derives the medoid from the span on replay.

## Recommendation

Report **two separate metrics**, never one "compression ratio":
1. **Write (online):** *redundancy filtered* = PREDICTED-dropped fragments / total. Today ≈ 4% — expected on low-redundancy prose; the value shows up on repetitive streams.
2. **Consolidation (offline):** *abstraction ratio* = claim bytes / raw bytes ≈ 0.24, the real compression, traded against ~6.25/10 reconstruction fidelity.

The fidelity gap (9.1 vs 6.25) is the cost of v2's abstraction; the byte gap (0.96 vs 0.24) is its benefit. Whether to push write toward more compression is a `z_echo` calibration decision owned by consolidation — the dial works; the bet is the open question.
