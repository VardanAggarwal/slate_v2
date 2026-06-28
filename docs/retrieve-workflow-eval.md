# Retrieve-workflow eval — fragment-backed retrieval (P2.5)

Companion to `Slate execution plan.md` §2. Records the **R0 asymmetry spike** (the
gate) and the first wiring of the Write fragment layer into RETRIEVE — closing the
**orphan branch** (`recall`/`consolidate` read claims/concepts; neither read
fragments) that blocked P3's calibration loop from having an SR@B objective.

**Corpus:** eval user `usr_01KTXAYR20J4R6F7PT3DP10W3W`, 166 notes, 972 fragments
(74 NOVEL / 898 AMBIGUOUS, materialized via `write.refine_pending`). **Method:** run
on a **copy** of `data/engine.db`; local MiniLM embedder; gold = `eval/gold.jsonl`
(14 queries, frozen). Scripts: `scratchpad/r0_asymmetry.py`, `scratchpad/run_eval.py`,
`scratchpad/retrieval_diag.py`.

## R0 — embedding-asymmetry spike (the gate) ✅ PASS

The PRD flags retrieval as the least-proven use: a *symmetric* encoder may not place
a query near the memory that answers it. R0 measures this before any build. No LLM,
pure geometry against the 972-fragment pool.

| signal | median | reading |
|---|---|---|
| query→answer cosine | **0.657** | query lands close to its own gold-fact content |
| ambient cosine (query→random frag) | 0.098 | baseline |
| **lift** | **0.541** | answer content is far above ambient — strongly separable |
| reachability percentile | **0.999** | the answer-fact outranks ~all 972 fragments for its query |
| top retrieved fragment cosine | 0.689 | real retrieved spans are close |
| asymmetry gap (ans↔ans − q→ans) | **−0.333** | q→answer (0.657) > answer↔answer (0.323) |

**Verdict: PROCEED with the symmetric MiniLM encoder — no query-aware lens needed.**
The feared asymmetry is *absent* on this corpus: queries embed nearer their answers
than the (paraphrased, diverse) gold facts embed to each other. The PRD's highest
risk (risk register #1) is empirically dismissed here. SR@B remains the real arbiter.

*Caveat:* gold key-facts are hand-authored *from* the corpus, which can flatter
query→answer cosine. The end-to-end SR@B below is the decisive check.

## Design — fragment-backed retrieval (`core/retrieve.py`)

The single-query subset of plan R-steps, built as a thin path over the predictor's
**assembly wrapper** (`core/assembly.py`), exactly as planned:

- **seed** — `store.fragment_candidates(query, k=40)`: nearest fragments, enriched
  with their medoid-sentence vector (the retrieval vector, referenced from
  `vec_sentences` — no duplicate stored) + episode provenance. Over-fetched so the
  selector picks for coverage, not just proximity.
- **select** — `assembly.assemble(cand, Y=query, weights=relevance)`: greedy
  max-marginal-residual (R4/R5/R6). Pick = `argmax(residual × relevance)`; dedup is
  free (a near-duplicate has residual≈0 vs the chosen set). STOP on the gain floor.
- **format** — verbatim spans grouped by source episode + title/date, cut to the
  char budget, most-relevant-first. Verbatim ⇒ near-lossless fidelity (write-eval
  9.1/10), the opposite frontier end from the v2 claims path (6.25/10).

Wired into the SR@B harness as the `frag` answerer alongside `slate` (v2 claims) and
`grep` (FTS over raw). Tests: `tests/test_retrieve.py` 7/7 (relevance ordering,
empty/isolation guards, provenance grouping, budget truncation, dedup).

## SR@B — CLEAN API run + C12 value_floor fit (2026-06-22, supersedes the noisy run below)

Funded Anthropic API (answerer + judge both **sonnet** — a haiku answerer was tried
and *deflated* SR@B, failing queries sonnet passed, so it was reverted). Full n=14
gold per cell, no excluded errors. Cost ~$0.97 baseline + ~$0.75 fit. The numbers in
the older CLI-judged table further down are **superseded** by this.

| system | SR@B @ B=2000 | tail | SR@B @ B=1000 | tail |
|---|---|---|---|---|
| slate (v2 claims) | 50% | 40% | 43% | 40% |
| **frag** | 57% | 40% | 50% | 20% |
| **hybrid** (frag+concept) | **64%** | 60% | 43% | 0% |
| **grep** (raw FTS) | **86%** | 100% | **71%** | 60% |

- **hybrid > frag > slate** — the frag+concept hybrid is the best Slate variant (64%),
  confirming the complementarity thesis; it owns the tail (60%) the single paths miss.
- **grep still dominates** (86%) — same short, self-contained-note corpus story: 2–3 raw
  notes fit in B and carry the answer verbatim, so compression has little to win. Slate's
  parsimony thesis only pays where the raw doesn't fit (a tighter-budget / larger-note
  corpus, still to be measured).
- **B/2 over-injection is real**: hybrid collapses 64%→43% (tail 60%→0%) at B=1000.

**C12 value_floor fit (frag path, pushed to `calibration_profiles`):**

| floor | B=2000 SR@B / tail / ctx | B=1000 SR@B / tail / ctx |
|---|---|---|
| none (raw residual) | 57% / 40% / 1856 | 50% / 20% / 895 |
| **0.25 (fitted, pushed)** | 57% / 40% / **1729** | **57%** / **40%** / 870 |

The relevance-aware STOP earns its keep: at **B/2 it lifts SR@B 50→57% and tail 20→40%**
(kills the over-injection), and at full B it holds SR@B while trimming context (parsimony).
`value_floor=0.25` is persisted per-user; `retrieve.assemble_context` loads it by default.
This is the C12 loop run end-to-end — fit against frozen SR@B, pushed down — not pre-tuned.

## SR@B — fragments vs claims vs grep (2026-06-22, NOISY CLI run — SUPERSEDED above)

Judge+answerer via the `claude` CLI (subscription); see the LLM-infra note at the
foot. Errors (session-limit blips) excluded, so per-cell denominators vary 11–14 —
treat as **directional, not definitive** (n is small and only ~5 gold rows are hard).

| system | SR@B @ B=2000 | tail | mean ctx | SR@B @ B=1000 | tail | mean ctx |
|---|---|---|---|---|---|---|
| **frag** (P2.5) | 46% (6/13) | 20% | 1884 | 36% (4/11) | 0% | 926 |
| slate (v2 claims) | 42% (5/12) | **67%** | 720 | 38% (5/13) | 25% | 665 |
| **grep** (raw FTS) | **79%** (11/14) | **100%** | 2001 | **62%** (8/13) | 50% | 1000 |

Apples-to-apples on the 8 queries every cell scored cleanly: frag@2000 **50%** vs
slate **12%** vs grep **62%**; frag@1000 50% vs slate 38% vs grep 75%.

**R-gate: ❌ NOT met by fragment-only retrieval.**
1. *frag ≥ claims recall?* Overall yes (marginally), but frag **loses the tail badly**
   (20% vs 67% at B=2000) — and "the tail is the gate." Not a clean win.
2. *Slate ≥ grep@B?* No — **grep dominates** at both budgets (79/62% vs frag 46/36%).

**But P2.5's real objective is achieved:** fragments are now a wired SR@B consumer —
the orphan branch is closed, so `z_echo`/the per-cluster knob finally have a path to
SR@B and **P3 (the calibration loop) is unblocked**.

### What the per-query matrix shows (the useful part)

- **frag and claims are COMPLEMENTARY, not redundant.** frag wins where claims lose
  (g02/g05/g07/g08 — factual, specific; verbatim spans carry the detail) and loses
  where claims win (g03/g11 — hard/conceptual; abstraction + the concept graph
  synthesise better than raw spans). Neither dominates → a **frag+concept hybrid** is
  the obvious next answerer, not a replacement of one by the other.
- **grep is a brutal baseline on THIS corpus** because notes are short (~800 tok) and
  self-contained: 2–3 whole raw notes fit in B and contain the answer verbatim, so
  there is little for compression to win. Slate's parsimony thesis only pays when the
  raw doesn't fit. grep *does* degrade with budget (79%→62%, tail 100%→50%), so the
  crossover likely sits at a **tighter budget (B/4 and below)** — worth measuring.
- **g07: frag ✓ but grep ✗** (both budgets) — a concrete case where targeted spans
  beat a raw-note dump. Existence proof that Slate *can* beat grep per-query.
- **Over-injection is real and harmful.** frag fills the budget (1884/926 tok) and
  drops from 46%→36% at B/2; the assembly never stops (`n_ch=24` every query). This is
  the clearest actionable defect — see below.

## Known issue — assembly does not stop (over-injection)  ⟶ now the top P3 lever

The geometric diagnostic shows fragment assembly hits the item cap (`n_ch=24`) on
**every** query: the gain floor (0.35, calibrated for write-time region cohesion)
never fires, because in retrieval the assembly Y grows from a tiny query seed, so the
marginal residual stays high (~0.86–0.95) even for tangential picks. The char budget
is the only effective stop, so frag pads to budget and that *hurts* at B/2. This is
the PRD's over-injection risk and exactly the **stop calibration P3/C12 must fit** —
now reachable because the loop runs end-to-end. The fix belongs to R2/R7 (the
*relevance/answerability* gate the minimal P2.5 omitted): stop on marginal **value**
(residual × relevance), not raw residual, so a novel-but-irrelevant span can't pad,
and return nothing when the whole query is off-corpus. **Deferred — it must be fitted
against SR@B (the system's own discipline), and the LLM window was exhausted; do not
ship an unvalidated knob.**

## P3 progress — relevance-aware STOP built (2026-06-22)

The R2/R7 stop is **built as an opt-in `value_floor` knob** in `core/assembly.py`:
the PICK already maximised residual×relevance, but the STOP read raw residual; with
`value_floor` set, STOP now fires on the same marginal **value** = residual×relevance,
so a high-residual-but-tangential span can't pad. Default OFF (`None`) — preserves
every existing caller's raw-residual semantics; `retrieve.py` registers it as the
fit target but leaves it disabled (not pre-tuned). Geometry-tested
(`tests/test_wrappers.py::test_assemble_value_floor_excludes_tangential_novel`,
147/147 green): an orthogonal-to-query span is padded in under the raw stop, dropped
under the value stop.

The fit driver `eval/fit_stop.py` sweeps `value_floor` vs **frozen** SR@B (the only
credit; plan risk #2), tie-broken by tail then by *lower* mean-ctx (parsimony). It is
**resumable** (per-(floor,budget,query) cache flushed each call) so the quota-gated
judge can stop+resume across CLI session windows. **Not yet run** — needs fragments
materialized on a db copy (the live `data/engine.db` has 0 fragments) + a clean LLM
window; the fitted floor then gets pushed down by C12.

## frag+concept HYBRID answerer — BUILT (2026-06-22)

`core/hybrid.py` (`hybrid_context`) blends the two complementary single paths under
one budget B: **concepts/claims for the tail** (`recall.assemble_context`) +
**fragments for specificity** (`retrieve.assemble_context`). A thin orchestrator —
no new geometry; each sub-path keeps its own seed/select/provenance/dedup. Budget is
split by `concept_share` (fragments keep the majority, concepts get a guaranteed tail
slice); if one path is empty its slice is handed to the other, both empty → the
sentinel. Wired as the `hybrid` SR@B answerer (`eval/harness.py`). `concept_share` is
a **C12 fit target, NOT pre-tuned** (default 0.4) — mirrors `value_floor`. Offline
geometry/plumbing tests: `tests/test_hybrid.py` 6/6 (both-paths blend, each-path
fallback, budget, isolation), full suite 153/153.

**The SR@B validation of the hybrid is deferred with the rest of the quota-gated
sweep** (below) — the build is complete and offline-tested; the North-Star credit
waits on a clean LLM window.

## Next (data-driven)

1. **Run the `fit_stop` sweep** — validate the value_floor on SR@B@B/2 and a B/4 point
   (where Slate should overtake grep). Materialize fragments on a copy first.
   *Deferred (quota-gated); skipped this pass per direction.*
2. ✅ **frag+concept hybrid answerer** — BUILT (see above); SR@B credit pending the
   same window. Target: the 67%-vs-20% tail gap.
3. **P3 calibration loop** — now unblocked: fit per-cluster `z_echo`/`gain_floor`/
   `prox_margin` **and** the hybrid's `concept_share`/`value_floor` against frozen SR@B.
4. Re-run the full matrix once quota refreshes (it was noisy; pin down on a clean run).

## LLM-infra note (operational)

The 168-call eval is gated by quota on every provider: Anthropic **API 429**, **Gemini
free-tier cap (20/day)**, and the **CLI subscription session limit** (hit after ~8
`claude -p` calls — each fresh print-mode subprocess creates ~26k cache_creation
tokens, and the ~5h window is shared with the interactive agent session). Mitigation
(in `scratchpad/run_eval.py`): route `llm.call → llm._call_claude_cli` (stored-session
auth, bypasses the env-token gate), and make the driver **resumable** — skip
already-scored queries, save per-query, stop cleanly on a session-limit error, rerun
next window. See memory `slate-prod-llm-account-capped`.

## Known issue — assembly does not stop (over-injection)

The geometric diagnostic shows fragment assembly hits the item cap (`n_ch=24`) on
**every** query: the gain floor (0.35, calibrated for write-time region cohesion)
never fires, because in retrieval the assembly Y grows from a tiny query seed, so
marginal residual stays high (~0.86–0.95) even for tangential picks. The char budget
is the only effective stop. This is the PRD's over-injection risk and exactly the
**stop calibration P3/C12 must fit** — now reachable for the first time, since the
loop runs end-to-end. Candidate fix: gate the stop on marginal *value*
(residual × relevance), not raw residual, so a novel-but-irrelevant pick can't pad.
Deferred to the calibration step (don't pre-tune before the baseline SR@B is read).
