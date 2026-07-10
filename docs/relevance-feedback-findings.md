# Query-tagged relevance feedback — lab findings (2026-07-10)

**Question.** Feedback events (`RELEVANCE_FEEDBACK`, `ENGAGEMENT`) carry the query,
but the only consumer (`_relevance_net` → C13b background bit) throws it away.
Can query-conditioned (pointed) consumption improve retrieval without regressing
the golds?

**Answer.** Yes — one lane wins decisively. A retrieval-time salience rerank that
weights each feedback event by cos(live query, feedback query) more than doubles
paraphrase-transfer coverage (mean .250 → .541) with every original-gold score
byte-identical and the orthogonality gate clean. Both alternative consumption
schemes (qclm attachment, seed injection) lose or regress.

## Method

Lab: `scratchpad/feedback_lab{,2,3}.py` on the traversal-lab base
(`/tmp/slate_trav_base.db`, shipped bar). Deterministic Coverage@B (τ=0.54, B=2000),
golds narrow 16 / broad 19 / paragraph 9.

- **Feedback synthesis** (eval-only, engagement_gen discipline): replay each gold
  query through shipped `resonance.activate`; surfaced window = top-20 nodes +
  frame concepts; node → `relevant` if max-cos(member text, key_facts) ≥ .54,
  `irrelevant` if < .42, else unmarked. 44 events, 292 relevant / 501 irrelevant
  marks, emitted through `record_relevance_feedback`.
- **Eval protocols**: `orig` = the three golds (no-regression bar; same-query
  replay for feedback lanes). `para` = same key_facts, LLM-paraphrased queries
  (`scratchpad/gold_paraphrase_map.json`, median cos(orig,para)=.55 — aggressive
  rewording, baseline drops to .250). `O-gate` = broad-only feedback, eval
  narrow+paragraph — any Δ<0 is disqualifying cross-talk.

## Results (mean of narrow/broad/paragraph SR@B)

| lane | orig | para | O-gate | verdict |
|---|---|---|---|---|
| F0 baseline | .617 | .250 | — | |
| F0fb events-unconsumed | .617 | — | — | no leak (byte-identical) |
| F1 shipped C13b (global bit) | .617 | .250 | — | **coverage-inert** — confirms diagnosis |
| F2 rerank α=.5, W=.55 | .617 | .367 | — | wins |
| F2 rerank α=1.0, W=.55 | .617 | .405 | clean | wins |
| F2 α=1.0, W=.45 sym | .617 | **.541** | **FAIL** (narrow −1) | leak via suppressions |
| F2 α=1.0, W=.50 sym | .617 | .520 | clean | |
| **F2 α=1.0, pos≥.45 / neg≥.55** | **.617** | **.541** | **clean** | **winner** |
| F2 placebo (votes shuffled) | .462 | .250 | — | lift is genuinely pointed |
| F3 feedback-anchored qclm | .617 | .271 | clean | inert (6th "topology inert") |
| F4 seed injection | **.542** | .308 | clean | regresses orig paragraph −2q |

Winner per-gold on para: narrow .312→.688, broad .105→.158, paragraph .333→.778.

## Mechanism + boundary conditions

- The lift is **re-ordering within the activation field**: a reworded query still
  lights the right nodes, just below the materialize cut; votes from a similar past
  query pull them above it. Same-query replay (orig) is flat because voted-relevant
  nodes were by construction already surfaced — feedback cannot add content the
  field never reaches (that stays the oracle-seeding wall, .21→.263).
- **Suppressions are the dangerous half.** At w≥.45 sym the O-gate fails through
  negative votes demoting nodes an unrelated query needed; positive boosts are
  gate-clean down to .45. Hence the asymmetric thresholds.
- **Placebo control**: shuffling vote→node pairing kills the transfer and damages
  orig (.617→.462) — the win is the pairing, not extra salience mass.
- F4's failure is the known flooding mode: force-injecting past-relevant nodes at
  top salience displaces budget from what the live query needs.

## Signal-mix validation (prod = implicit drills + explicit on a subset; `feedback_lab{7..10}.py`)

Engagement drill simulated as the top surfaced-relevant node per query, home-concept
normalized (what `record_engagement` logs); explicit events on a seeded 50%/100%
of queries.

| config (all pos-only, α=1.0, w≥.45) | orig | para | O-gate |
|---|---|---|---|
| eng-only, votes on CONCEPT node | .600 (para-gold −1q) | .464 | — |
| … α_eng .5/.3, band-cap w<.95, mix50-supersede | still −1q somewhere | — | — |
| mix100 (explicit all + concept drills) | .637 | .523 | — |
| **eng-only, drill → member CLAIMS** | **.617 flat** | **.479** | — |
| **mix50, drill → member claims** | **.617 flat** | .442 | **clean** |

**Rule: vote granularity must be CLAIM-level.** A concept-node boost displaces
budget for *other* queries at moderate w (paragraph −1q, α-independent; band caps
and supersede-explicit don't fix it). Normalizing drill votes to the engaged
concept's primary member claims makes every mix pass every gate. Implicit-only
already delivers most of the transfer (.479 of .541) — the flywheel works even if
hosts never call mark_relevance; explicit votes sharpen it.

## Gate (pre-hop reach expansion) — REJECTED (`feedback_lab_gate.py`, reach_diag.py)

Motivation (user): rerank only reorders the already-lit set (the same job the agent
does downstream); it can't rescue a node that never activated. A query-*gate* —
feedback priming a region as extra pre-hop seeds so the PE-gated spread flows into
it — targets the other failure mode. Diagnostic (`reach_diag.py`) confirmed the
mode is real and large: on paraphrases, facts split above 31% / below (rerank's
domain) 28% / **dark (only a gate can reach) 40%**. But "dark" overstates it — the
τ split showed the credible, reach-recoverable share is narrow-regime only (~12
facts); broad's big dark count is the synthesis wall (oracle .263), paragraph is
already lit.

Mechanism tried: `activate(primed={node: w·scale})` injects the feedback region's
concepts as seeds on a dedicated probe channel; the real spread + conductance gate
then vet propagation; rerank + top-N cut filter budget ("gate opens, rerank
filters").

| lane (α=1.0, claim-level rerank) | orig | para | note |
|---|---|---|---|
| rerank_only (reference) | .617 flat | .466 | |
| gate_only s0.6 | .579 (para −1q) | .387 | reach alone = noise |
| gate+rerank s0.3 | .579 (para −1q) | .483 | best para (+1 broad q) |
| gate+rerank s0.6 / s1.0 | .579 (para −1q) | .445 / .466 | non-monotone |
| gate+rerank s0.6, broad-only :O | .579 (**para −1q**) | — | cross-topic LEAK |
| transfer-band [.45,.95) s0.3 | .559 (**narrow −1q** too) | .483 | band-cap worsens |

**Verdict: rejected.** Every gate variant regresses originals (paragraph, and
narrow under band-cap) at every scale, and broad-only feedback leaks into paragraph
(:O). Para gain over rerank-only is one broad query. Root cause: a gate adds
activation *mass*, restructuring the whole field and shifting budget — unlike
rerank, which only reweights existing salience. The "flood" isn't junk the
conductance gate rejects; it's real regional content that legitimately wins budget
slots breadth/exact queries needed. The rerank filter can't demote it (it only
promotes votes). Reach-expansion and coverage trade off here; rerank-only (F2) stays
the winner. Core hook reverted (prod pristine).

Reproduce: re-add to `core/resonance.activate()` (signature `, primed: dict | None = None`)
after the knn-seed loop, before `def strength`:
```python
if primed:                       # feedback gate — dedicated probe channel
    gate_pi = len(p_embs)
    for node, val in primed.items():
        if deposit(node, gate_pi, float(val)):
            seeded.add(node)
```

## Residual seed-shift (mass-neutral reach, "rerank at a lower level") — REJECTED (`feedback_lab_resid.py`)

Motivation (user): rerank reweights node salience *within the lit set* (same level the
agent reranks). The additive gate failed by injecting mass. So intervene one level
DOWN, at the seed embedding, and MASS-NEUTRALLY: re-aim the fixed seed budget along the
**residual** (predict.residuals_against first-principle) of confirmed-relevant content
vs. the query — `q_seed' = norm(q + β·resid(rel_centroid, q))` — while leaving `q_emb`
(fragment scoring) on the true query. Seed/fragment are already decoupled in
`activate()`, so reach and filter separate cleanly. Rocchio-in-the-orthogonal-direction.

| lane (band-capped [.45,.95), positive-only) | orig | para | note |
|---|---|---|---|
| rerank_only (reference) | .617 flat | .466 | |
| shift_only β0.3 | .505 (**paragraph −3q**) | .405 | narrow/broad protected by band, paragraph craters |
| shift+rerank β0.15 | .522 (narrow −1, para −2) | .424 | filter doesn't recover |
| shift+rerank β0.3 | .484 (paragraph −3q) | .462 | |
| shift+rerank β0.5 | .501 (narrow −2) | .462 | |
| shift+rerank β0.3, broad-only :O | .559 (**narrow −1, paragraph −1**) | — | cross-topic leak |

Non-band β0.3 (no cap) regressed originals across all three (narrow/broad/paragraph) —
the exact-match self-harm. Band-cap fixed narrow+broad self-harm but paragraph still
craters (breadth queries are in-band-similar to other feedback, so their seeds get
pulled off the breadth they need), and broad-only feedback leaks into narrow+paragraph.

**Verdict: rejected — and it closes the whole reach-expansion family.** Additive (gate)
and directional-mass-neutral (residual shift) both regress. The general principle: the
true query's own seeding is already optimal for its own facts; feedback from any *other*
query (however similar) can only pull reach toward that other query's targets, which for
multi-fact/breadth queries means dropping a fact — net-negative budget displacement.
**Rerank works precisely because it does NOT change reach** — it only breaks ties within
the already-optimal reached set, at the budget boundary. On this substrate feedback is a
within-reached-set tiebreaker, not a reach lever. Core hook (`seed_shift`) reverted.

## SHIPPED 2026-07-10 (flag ON by default)

Wired, not a sketch. `retrieve.feedback_rerank_index` folds `RELEVANCE_FEEDBACK` +
`ENGAGEMENT` into a per-user `[{q_emb, targets}]` index (cached per event-count so
feedback queries embed once, not per recall — no consolidation table needed yet);
`retrieve.apply_feedback_rerank` multiplies voted nodes' salience by (1+α·w) for
events with cos(live,fb) ≥ w_pos. Applied in BOTH retrieval heads — `recall()`
(core/recall.py) and `assemble_context`→`resonance_recall` (core/resonance.py) —
after `activate()` (which stays PURE: consolidation's query-injection must not see
feedback). Flags in `resonance.DEFAULT_CALIBRATION`: `res_fb_rerank=True`,
`res_fb_alpha=1.0`, `res_fb_w_pos=0.45`. **Kill switch: `res_fb_rerank=False`.**

Granularity (what the lab validated byte-flat): RELEVANCE_FEEDBACK `relevant` ids
used AS-IS (explicit votes are the right grain; episode ids expand to claims);
ENGAGEMENT engaged-concept expands to its primary member claims (concept-node drill
votes regress — signal-mix table). Positive votes only (irrelevant ignored).

Wired-path verification (`scratchpad/verify_feedback_rerank.py`, real events through
`resonance_context`): flag OFF = baseline byte-identical; flag ON = originals
byte-flat (narrow .750 / broad .210 / paragraph .889), paraphrase transfer .250 →
.562. Unit tests: `tests/test_feedback_rerank.py` (flag-off no-op, empty-index no-op,
similar-query boost, dissimilar no-effect, positive-only, ENGAGEMENT member-expansion,
recall-head). Full suite 230 pass.

RISK (per the ON decision): magnitudes fitted on 44 synthetic oracle-labeled queries.
Positive-only bounds the downside to a wrong promote (never a demote), the O-gate was
clean, and 20% label-noise cost ≤1 query — but re-validate α / w_pos against real
logged events before trusting the magnitude. Optimization deferred: precompute the
index at consolidation into a `baselines`-style table if per-recall build latency bites.

## Noise robustness (labels flipped 20%, seed 7 — `feedback_lab{4,5,6}.py`)

| config | orig | para |
|---|---|---|
| asym ±votes, noisy | .562 (broad −1q, para-gold −1q) | .578 |
| **pos-only, noisy** | .599 (broad −1q only) | .578 |
| pos-only α=.5/.3, noisy | .599 (unchanged) | .503 / .424 |

- Suppressions are worthless AND fragile: dropping them entirely keeps the full
  clean transfer (.541, identical) and halves the noise damage. **Final config:
  positive-only, α=1.0, w≥0.45.**
- The residual −1 broad query under noise is a single flipped vote boosting an
  off-topic node into the materialize window; it is α-independent (persists at
  α=.3) — the benchmark's single-query noise floor, not a tunable regression.
  Clean-label originals are byte-flat in every F2 config.

## Caveats

- Synthetic feedback is oracle-labeled (fact-cos split). The placebo and the 20%
  flip bound the mislabel risk: random pairing is harmful, 20% flips cost ≈1
  benchmark query — real hosts should sit well inside that.
- Paraphrase transfer is the proxy for "user asks the same thing differently";
  real query logs may couple tighter (median w likely > .55) → more events above
  threshold, more effect.
- Negative votes still have value for felt-quality via the existing C13b
  background bit (unchanged); they should just stay out of the rerank.
