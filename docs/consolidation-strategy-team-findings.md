# Consolidation strategy race — what we tried and what we learnt (2026-06-27)

A plain-language summary of the agent-team experiment that searched for ways to make
Slate's consolidation produce better retrieval (Coverage@B). Evidence + scripts at the
bottom; this is the story in simple words.

---

## The goal (in one paragraph)

Slate breaks your notes into ~2000 small facts ("claims") and groups them into ~317
topics ("concepts"). When you ask something, it pulls back the relevant claims. We grade
the grouping with **Coverage@B**: across three kinds of question — **narrow** (precise
lookups), **broad** (synthesize across many notes), **paragraph** (rebuild a passage) — does
the retrieved text actually contain *every* fact the answer needs (cosine ≥ τ=0.54)? A
question "passes" only if all its facts are covered.

The grouping ("consolidation") has three layers:
- **Membership** — which claim goes in which topic.
- **Center** — the single vector that represents each topic.
- **Bridges** — links between topics.

**The bar to beat** (today's shipped system: LLM-decided membership + medoid + contrastive
centers): **narrow 78.6 / broad 36.4 / paragraph 75.0, mean 63.3** (317 concepts, biggest 36).

---

## How we ran it

Rather than guess one approach, we raced **5 strategies as a team of agents**, each betting
on a different idea from **information theory + brain science**, with a strict rule to
**minimize LLM use** (the agents reasoned over the claim text themselves; zero calls to the
billed LLM). The team:

1. **Round 1 — independent.** Each strategy built and tested itself in isolation.
2. **Blackboard.** Everyone's findings were pooled into a shared digest.
3. **Round 2 — cross-pollinate.** Each agent borrowed the most useful idea from a peer and refined.
4. **Synthesis.** We stacked the winning pieces into one combined pipeline.
5. **Adversarial verify.** A final agent tried to *break* the winner — reproduce it, check for
   cheating (using the test answers to tune), check for "collapse" (everything merging into blobs).

All strategies plugged into one shared test bench (`scratchpad/strat_lab.py`) so the
comparison was apples-to-apples and the test questions were never used to tune anything.

---

## The 5 strategies, in plain words

| | Name | The bet | Inspiration |
|---|---|---|---|
| **A** | Backup copies in the right places | Put a few claims into a *second* topic too — but only where a topic has a real **gap** (a claim it should cover but currently can't reconstruct). | Error-correcting codes: smart redundancy so nothing is lost |
| **B** | What's used together belongs together | Watch which claims surface together for similar questions; group by that co-use. | Brain: "neurons that fire together wire together" |
| **C** | Give it better senses | The text-reader is blind to *not*, numbers, and scope ("all" vs "some"). Add explicit signals for those, then regroup. | Richer input alphabet |
| **D** | Pack it tightly (but don't over-merge) | Minimize description length, with a guard against collapsing everything into blobs. | Minimum description length / predictive coding |
| **E** | Sharper topic labels | Make each topic's center point at what makes it **distinctive**, not just its average. | Surprise / IDF weighting |

---

## The five strategies in detail

Each agent got the same brief, the same shared test bench, and a rule to avoid the LLM. Here
is what each one actually *did* under the hood, and what came of it.

### A — Channel-code redundancy  ✅ WINNER (mean 67.5)
**Principle (information theory): channel coding, not source coding.** Earlier work proved
something counter-intuitive: the LLM's grouping is *not* the tightest possible compression, yet
it retrieves best. The reason is that retrieval is a noisy-channel problem — a question is a
garbled version of what you stored, and you want **structured redundancy** so the right claim is
still recoverable. Compression (removing redundancy) is exactly the wrong instinct.

**What it did mechanically:** keep the LLM grouping and the topic centers *exactly as they are*.
Then look for **coverage holes** — a (claim, topic) pair where the claim is **reachable** from
the topic (its embedding is close to the topic centre, cosine ≥ 0.55, so a question landing on
that topic could plausibly want this claim) **but** the topic's current members **cannot
reconstruct** it (reconstruction-z between 1 and 3 — measurably above the topic's normal internal
spread, so it's a real gap, not noise). Score each hole by `cosine × z` (near the topic *and*
genuinely uncovered = the best "parity bit"), and add the globally top-ranked few as a **second
membership** for that claim, at most one extra per claim. In this corpus exactly **33** holes
clear the bar. Filling them lifts paragraph 75 → 87.5 with nothing else touched.

**Why it's a real win:** non-destructive (centers fixed first, copies added after, so a topic's
identity can't be diluted), no collapse (biggest topic stays 36), reproducible, no test leakage.

### B — Hebbian usage replay  (moved broad, taxed narrow — Pareto trade, not a win)
**Principle (brain + information theory): "neurons that fire together wire together," scored with
mutual information.** The embedding is blind to meaning the words don't carry, but **usage**
recovers it: two claims that keep getting pulled up *together* for the same need belong together,
even if their text looks different. There's no live traffic, so usage was **bootstrapped**: treat
each claim's own text as a stand-in question, run it through the real retrieval/activation path,
and record which claims **light up together**. Co-activation counts become a Hebbian affinity,
turned into **PMI** (so a claim that co-fires with *everything* — merely popular — is discounted;
only genuinely informative co-firing counts). That affinity then either re-votes membership or,
safer, adds usage-based redundant memberships after centers are fixed.

**Leakage guard (important):** the stand-in questions are **claim texts only** — never the gold
questions, which live solely inside the scorer. The script never opens the gold files.

**Result:** the only lever all session that moved **broad** (36.4 → 45.5) — but its broad lift
comes bundled with ~775 diffuse extra memberships that **blur precise topics**, dropping narrow
78.6 → 64.3. So it's a trade across axes (Pareto-incomparable), not a mean win. Still the most
interesting *signal*: it proves broad responds to usage meaning the embedding lacks.

### C — Facet featurizer  (refuted — inert on this benchmark)
**Principle (information theory): enrich the input alphabet.** The text-reader (MiniLM) flattens
exactly the things that distinguish precise claims: **negation** ("not"), **number/magnitude**,
**scope** ("all" vs "some"), **named entities**. So bolt on 9 hand-built dimensions that carry
those signals, glue them onto the 384-dim embedding (scaled by a weight), and re-run the gap/
membership test **in that enriched space** — while keeping the centers written for retrieval in
the plain 384-dim space (so only the *grouping decision* changes, not retrieval).

**Result:** the facet dimensions *are* real — they change which claims get picked at the
boundary — but the picks they favour are claims the **gold questions don't ask about** (negation/
number/scope-distinct facts that no probe targets). So Coverage@B comes out **bit-identical** to
plain space, and as you add more facet-selected copies it gets *worse* (dilution). The blindness
is real; this particular cure doesn't touch what the benchmark measures.

### D — Free-energy / MDL consolidation  (refuted — ties bar, no variant beat it)
**Principle (brain + information theory): minimum description length with an anti-collapse term.**
Form topics to minimise total "description length" = cost of the model (how many topics × how
expensive each centre) + cost of the leftovers (how badly each claim is reconstructed by its
topic) — but add a **redundancy reward** so the objective doesn't just compress everything into
mega-blobs (which earlier "minimise distortion" experiments proved is the failure mode). Start
from the LLM grouping and make only greedy merge/split moves that lower the objective.

**Result:** every merge/split move was a one-way ratchet toward agglomeration (the refuted
collapse). Once D adopted A's idea of realising the redundancy term as scarce, budgeted copies
*instead of* re-grouping, it just **reproduced A**. Net: MDL re-grouping adds nothing; the only
useful part of D's objective was the redundancy term, which is A.

### E — Surprise-weighted centers  (refuted — inert or harmful)
**Principle (information theory): point the label at what's distinctive (PMI / IDF).** The shipped
centre is a medoid plus a contrastive pick of one representative member. E tried to build the
centre as a **surprise-weighted** combination — down-weight members that look like the whole
corpus (common, low information) and up-weight rare/distinctive members — so the centre points at
what most *discriminates* this topic. Three variants: weighted sum, surprise-tilted medoid, and a
small distinctiveness nudge on the shipped vector.

**Result:** inert at best, harmful at worst (the aggressive variant cost −7.2). The reason is
structural: in the retrieval path a claim is reached through its **membership link**, not through
how well the topic's centre vector points at it — so re-aiming the centre changes nothing the
benchmark sees. This is *why* the whole team converged on working the **membership/redundancy**
layer, not the centre layer.

---

## What won

**Strategy A — "backup copies in the right places."**

| | narrow | broad | paragraph | **mean** |
|---|---|---|---|---|
| Bar (shipped) | 78.6 | 36.4 | 75.0 | 63.3 |
| **Winner (A)** | 78.6 | 36.4 | **87.5** | **67.5** |

- **+4.2 mean**, entirely from paragraph (75 → 87.5). Nothing else got worse.
- **Nothing broke**: same 317 topics, biggest topic still 36 (no blob collapse).
- **Passed the adversarial check**: reproduced exactly on a fresh copy, used no test answers
  to tune, no collapse. Verdict: **confirmed win**.

How it works: leave the existing grouping and centers untouched. Then add a **small, budgeted**
set of redundant memberships — a claim joins a second topic **only** if it's *reachable* from
that topic (close to its center, cos ≥ 0.55) **yet** that topic's current members *can't*
reconstruct it (reconstruction-z between 1 and 3). That's a genuine **coverage hole**; the
extra copy is a "parity bit" that fills it. This is the long-standing idea from the
rate-distortion work made concrete: retrieval wants **structured redundancy** (channel coding),
not maximum compression (source coding).

---

## What we learnt (the lasting lessons)

1. **A little redundancy in the right gaps is the whole game.** The corpus has exactly **33**
   such gaps. Fill those and the score jumps; do anything else and it doesn't move.

2. **The "smarter" ideas added nothing — and we now know why.** It's a **floor phenomenon**:
   the gap test (cos ≥ 0.55 AND z ∈ [1,3]) already selects exactly those 33, and they all get
   admitted. So B's usage ranking, C's facet ranking, and E's center tweaks were just
   re-sorting a set that was being taken whole. Mathematically irrelevant. *(Every Round-2
   strategy independently converged on the identical 67.5 once it adopted A.)*

3. **A claim is found through its membership link, not through where the topic's label points.**
   That's why re-pointing centers (E) did nothing — and the over-eager version actively hurt
   (−7.2). Reachability has to be measured in the **standard embedding space**, because that's
   where real questions land.

4. **"Broad" questions are a hard wall (36.4).** The only thing that moved them was B's
   heavy-handed grouping (→45.5) — but that wrecked the precise questions (narrow → 64.3), so
   it's a trade, not a win. The deep reason: the text-reader is **genuinely blind** to meaning
   the words don't carry (negation, numbers, scope). No regrouping fixes a blind sensor.

5. **The next real gains are NOT in grouping.** They're in better **content/senses** (a richer
   or usage-aware embedding) or a **smarter retriever** that reads beyond plain cosine. This
   re-confirms the earlier "geometry vs LLM membership" conclusion from a completely fresh
   angle.

---

## Did we try combining the winners? (A + C, asked 2026-06-27)

Yes — A's gap-finder run **inside C's facet-augmented space** (so the "can this topic rebuild
the claim?" test sees negation/number/scope too). It is **strictly worse than A alone**:

| | extra copies | narrow | broad | paragraph | mean |
|---|---|---|---|---|---|
| A (standard space) | 33 | 78.6 | 36.4 | 87.5 | **67.5** |
| A+C (facet weight 0.5) | 80 | 71.4 | 36.4 | 62.5 | 56.8 |
| A+C (facet weight 1.0) | 80 | 71.4 | 36.4 | 75.0 | 60.9 |

Two things break: (1) facets shift the z-values so the gap-band **stops self-limiting to 33** and
admits 80; (2) the facet-distinct picks are claims **the gold questions don't ask about**, so
narrow and paragraph drop. Plain standard-space A picks strictly better parity bits. Run:
`CHAN_CONFIGS="0.0,80,0.55,1.0,3.0;0.5,80,0.55,1.0,3.0;1.0,80,0.55,1.0,3.0" PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/strat_C_facets.py chan`

---

## Did we try combining A + B? (asked 2026-06-27)

This is the *most* interesting combination, because A and B move **different axes** — A lifts
paragraph, B lifts broad — so in principle they could stack. In the original race, B's Round 2
set this up but the (network-bound) scorer timed out under 12-agent contention and only B's
Round-1 number was captured. So it was run cleanly afterward (`scratchpad/strat_AB.py` for the
union; `strat_B_hebbian.py round2` for the floor-ranking version). Result (bar mean 0.6331):

| variant | extra copies | narrow | broad | paragraph | mean |
|---|---|---|---|---|---|
| **A alone** | 33 | 78.6 | 36.4 | 87.5 | **67.5** |
| B alone (red_b3, broad-lifter) | 773 | 64.3 | **45.5** | 75.0 | 61.6 |
| **A ∪ B** (A's 33 ∪ B's 773) | 805 | 64.3 | **45.5** | **87.5** | 65.8 |
| A ∪ B (A's 33 ∪ B's tight 33) | 66 | 71.4 | 36.4 | 87.5 | 65.1 |
| A-floor, holes ranked by usage | 33 | 78.6 | 36.4 | 87.5 | 67.5 |

**Verdict: A+B does NOT beat A — but it tells us exactly why, and it's the richest finding here.**

1. **The two lifts genuinely stack.** `A ∪ B` is the *first and only* configuration all project
   where **broad 45.5 AND paragraph 87.5 appear together** — proof the axes are orthogonal and the
   two redundancy ideas compose. The mechanism works.
2. **But B's broad lift is inseparable from a 773-copy narrow tax.** Those diffuse adds blur the
   precise topics, dropping narrow 78.6 → 64.3. That single regression sinks the union mean to
   65.8 — above the bar, **below A's 67.5**. The narrow loss outweighs the broad gain on the mean.
3. **B's *tight* version doesn't help either:** with only 33 usage adds, broad doesn't move
   (stays 36.4) and narrow still slips to 71.4 — pure downside on top of A.
4. **Ranking A's holes by usage instead of `cos×z` just ties A** (67.5). Usage picks equivalent
   gold-movers; it doesn't find better ones. (Confirms the "floor phenomenon": within the
   reachable+uncovered set, *which* ranking you use barely matters.)

**The real takeaway:** broad and paragraph CAN be won at the same time — the blocker is purely
that B's only known way to move broad drags in hundreds of narrow-blurring copies. The open lever
is a **targeted** broad signal (usage adds that lift breadth *without* the diffuse narrow tax). If
that existed, A+B would beat A. That's the concrete next experiment for breaking past 67.5.
Run: `PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/strat_AB.py`

### Recovering narrow in A∪B — the inflow cap (2026-06-27)

Question: in A∪B narrow drops 78.6 → 64.3; can we recover it without losing broad? Diagnosis
(`scratchpad/narrow_recover.py`): narrow is scored over **14** queries; A∪B breaks exactly **2**
of them (`g06`, `g13`). The narrow tax is **concept flooding** — a precise concept that a narrow
query lands on receives several of B's diffuse copies, and they push the query's exact covering
claim out of the fixed retrieval budget.

**What does NOT work — selecting B's adds by geometry.** Every add-set filter (keep
embedding-far adds, keep adds into big concepts, keep the globally strongest N) recovers narrow
but **erases broad**, landing *exactly* back on A (0.786 / 0.364 / 0.875). Broad's lift is an
emergent **volume/recall** effect — it needs the full diffuse set; remove a chunk and it reverts.
So at the membership-set level, narrow-tax and broad-gain are **coupled**.

**What works — cap inflow per concept.** Don't filter *which* adds; cap *how many* any one
concept may receive (keeping the most bridge-like, lowest-cosine ones). This spreads B's volume
thinly across many breadth concepts (broad survives) while no single precise concept floods
(narrow recovers):

| variant | narrow | broad | para | mean | adds |
|---|---|---|---|---|---|
| A alone (prior best) | 0.786 | 0.364 | 0.875 | 0.6748 | 33 |
| A∪B full | 0.643 | 0.455 | 0.875 | 0.6575 | 805 |
| **A∪B, inflow ≤ 2/concept** | 0.714 | **0.455** | 0.875 | **0.6813** | 505 |
| A∪B, inflow ≤ 1 | 0.714 | 0.364 | 0.875 | 0.6510 | 297 |
| A∪B, inflow ≤ 3 | 0.643 | 0.455 | 0.875 | 0.6575 | 640 |

**Inflow ≤ 2 is the knee** and is the project's new best on the 3-gold mean (0.6813). ≤1 starves
broad; ≤3 floods narrow again. It recovers **1 of the 2** broken narrow queries while holding
broad's full lift.

**Two honest caveats:**
1. **It's a reshuffle, not net-more coverage.** A alone answers 11+4+7 = **22/33** total queries;
   A∪B inflow≤2 answers 10+5+7 = **22/33** — the *same 22*. It trades one narrow pass for one
   broad pass. The 3-gold *mean* prefers it only because broad has a smaller denominator
   (+1/11 − 1/14 = +0.0065). It is a genuine win **on the agreed mean metric**, and a good trade
   if broad (the hardest wall) is the priority — but it is not answering more questions than A.
2. **The 2nd broken narrow query is not cheaply recoverable.** Cohesion-aware caps (give tight/
   precise concepts cap 0–1, diffuse ones cap 3) either revert to A or drop broad — concept
   *tightness does not separate* "narrow-critical" from "broad-critical" concepts (they overlap,
   echoing the resonance thread's "signals overlap" routing refutation). Full recovery (keep all
   of narrow AND broad's gain = a true net +1) would need the **retrieval-layer split**: let
   redundant edges feed activation/breadth-budget but not the depth-budget a narrow query draws
   from. Untested; needs a `core/resonance.py` change.

Run: `PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/narrow_recover.py`

### Prototype: the retrieval-layer split (2026-06-27, `scratchpad/sec_proto.py`)

The idea was: mark B's redundant edges "secondary" — let them light concepts (broad) but
keep them out of the budget a narrow query draws from. Prototyped it properly (a
secondary-aware re-implementation of `resonance_recall`/`_context`; with `secondary=∅` it
reproduces stock retrieval bit-for-bit). **The diagnosis matters more than the original idea:**

1. **Where the narrow tax is NOT.** Tracing the two broken narrow queries (`g06`, `g13`)
   showed their covering fragments are *present as candidates* under A∪B; they get truncated.
   - Routing secondary fragments to the breadth budget → **no effect** (the covering frags
     weren't sourced from secondary members).
   - Gating, or even *killing*, secondary-edge activation spread → **no effect** (narrow
     stays 0.643 even with all secondary spread off). So the tax is not in spread.
2. **Where it IS.** The redundant memberships inflate node **degree**, and
   `distinctiveness = 1/(1+ln(degree))^γ` feeds salience — so degree inflation reshuffles the
   salience ranking, changes the top-12, and crowds the covering fragment out of the budget.
3. **The fix that works.** Exclude secondary edges from the degree count **for claim nodes
   only** (protect precise-claim ranking; leave concept degrees intact so breadth concepts
   stay bright). This **decouples**: keeps the **full** 773-add broad lift *and* recovers a
   narrow query.

| variant | narrow | broad | para | mean | raw |
|---|---|---|---|---|---|
| A alone | 0.786 | 0.364 | 0.875 | 0.6748 | **22/33** |
| A∪B full (stock retrieval) | 0.643 | 0.455 | 0.875 | 0.6575 | 21/33 |
| A∪B + claim-only degree fix | 0.714 | **0.455** | 0.875 | **0.6813** | **22/33** |
| A∪B inflow≤2 (membership fix) | 0.714 | 0.455 | 0.875 | 0.6813 | 22/33 |

**Verdict on the prototype:** it works — the retrieval split genuinely separates broad-gain
from narrow-tax, which membership filtering could not. But it lands on the **same 0.6813
ceiling** as the simpler membership inflow-cap, and stacking the two adds nothing. Crucially,
the raw accounting shows **A∪B by any lever tops out at 22/33 — exactly tied with A**; the
+0.0065 mean is the denominator effect (broad's 11 vs narrow's 14), a 1-narrow-for-1-broad
reshuffle, not net-new coverage. The 2nd broken narrow query is unrecoverable by every lever
tried (membership cap, fragment routing, activation gating, degree correction).

**So:** A+B is genuinely additive *on the axes* and now cleanly tunable at *either* layer, and
if you value broad it's the better operating point — but it does not answer more questions than
A. Net-new coverage still requires the broad **content/query-side** lever (next section), not
more A+B engineering. Run: `PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/sec_proto.py`

---

## Who actually fixes "broad"? — clearing up an apparent contradiction (2026-06-27)

The retrieval thread once said *consolidation* would fix broad; this thread says *retrieval/
content* will. That's not two teams punting — it's one hypothesis that died:

1. **2026-06-24 (retrieval):** *hypothesis* = broad is capped because consolidation hasn't split
   the mega-hub concepts. "Fix consolidation → broad lifts."
2. **2026-06-26:** that hypothesis was **tested and refuted** — splitting mega-hubs *lowers*
   broad. Consolidation geometry is not the lever.
3. **2026-06-27 (this race):** re-confirmed a third way — no scarce consolidation lever moves
   broad; only B's heavy diffuse grouping did, at narrow's expense (a trade, not a win).

So the retrieval thread's **final** position (after its own hypothesis failed) is the same as
this one: broad is the **R0 vocabulary-disjoint seed problem** — query words don't overlap note
words, so graph navigation never gets a foothold. The lever is **query-side**: HyDE (hypothesize
a passage from the query, embed *that*), query decomposition, or richer/usage content.

**Honest open status (now updated — the query-side fix was tried, 2026-06-27):** see below.

### Broad levers tried and refuted (2026-06-27)

Five distinct attempts to move broad, all refuted (figures are the 33q values they were run on):

| lever | what it does | result |
|---|---|---|
| **B as membership** | usage co-activation → redundant members | broad 0.364→0.455 BUT narrow 0.786→0.643 (trade) |
| **HyDE** (`scratchpad/hyde.py`) | hypothetical passage → extra/replacement seed | broad 0.364→0.273; narrow also dropped |
| **Query decomposition** (`hyde.py`) | broad query → precise facet sub-seeds | broad 0.364→0.273; narrow held 0.786 |
| **Hebbian bridges** (`hebbian_bridges.py`) | usage → concept↔concept L3 bridges | inert at N=50; broad drops at N≥150 |
| **Corpus-grounded query expansion** (`query_expand.py`) | first-pass top claims/concepts → extra seed probes (PRF / concept-label) | broad never gains; best (PRF k=5) holds 0.308, taxes narrow 0.75→0.688; concept-label craters both |

HyDE/decomposition were expected to be *additive* (can't hurt narrow); they weren't — generic or
facet seeds pull plausible-but-off content into the fixed budget and **crowd out** the precise
cross-note fragment. Hebbian bridges (the "use usage as L3, not membership" idea — should avoid
the degree-inflation narrow-tax) reproduced the **exact** geometric-bridge result: traversed but
inert at low count (top-12 materialized set unchanged), harmful at high count (salience boost
displaces precise claims). Bridges crowd the *budget* even though they don't inflate *degree*.

**Corpus-grounded query expansion** is the most informative null. It's the *grounded* fix HyDE
should have been — PRF expands the query using the user's OWN nearest claims/concepts, not generic
knowledge — and it *still* can't move broad. Why: PRF expands from the **query's neighbourhood**,
but the reach-miss covering claims are far from the query *by construction* (that's what makes them
reach-misses in a symmetric embedding). First-pass results bridge to more on-topic-but-not-covering
content, which crowds the budget (narrow tax). This proves the broad reach-miss is **not** a "generic
vs grounded expansion" problem — the concrete covering claims are simply **unreachable from the query
side in a symmetric embedding**. Only an asymmetric (query/passage) embedding puts the abstract query
near the concrete claim. (Run on the 38q set: A 0.750/0.308/0.889; PRF k=5 0.688/0.308/0.889.)

### WHY bridges are no-op, and whether tuning retrieval to value them helps (`scratchpad/bridge_tune.py`, 2026-06-27)

Bridges enter retrieval at one place — `_Graph.neighbours` weights a `bridges` edge
`min(1,weight)·BRIDGE_BOOST` — then spread activation one damped hop. A bridge matters only if
it pushes its target concept across the **top-12 materialize cutoff** (only the top-12 nodes pull
verbatim fragments). Diagnosis with 150 Hebbian bridges over the broad golds:

- Bridges **are** traversed — 13–41 bridge-target concepts light up per query — but they rank
  **below the cutoff** (ranks 12–53). That's the no-op.
- **It is NOT the PE-gate** (the prior docs' explanation — now corrected). Turning `res_pe_gate`
  off barely moved bridge targets. The actual suppressor is the **distinctiveness term**:
  `salience = strength·(1+ln conf)·distinctiveness`, `distinctiveness = 1/(1+ln degree)^γ`.
  Turning *that* off promoted bridge targets into the top-12 (in_top12: 0 → 4–6). It is
  **doubly self-defeating**: bridges link to related/hub concepts (low distinctiveness), and the
  bridge edge *raises* the target's degree → lowers distinctiveness *further* → suppresses the
  very concept it boosts.
- **Tuning to value bridges makes broad strictly, monotonically WORSE** (A-only broad = 0.364):

  | | BRIDGE_BOOST 1.3 | 3.0 | 6.0 |
  |---|---|---|---|
  | PE-gate ON | 0.273 | 0.091 | 0.091 |
  | PE-gate OFF | 0.273 | 0.182 | 0.091 |

  (wider `res_materialize_nodes` didn't help either). You *can* force bridge targets into the
  top-12 (kill the distinctiveness term), but the concepts promoted are by construction **less
  distinctive (vaguer)** than the precise covering claims they evict — so it costs coverage. The
  distinctiveness term that suppresses bridges is **doing its job: protecting precision.** Bridges
  aren't inert because retrieval *under*-values them; they are suppressed *deliberately*, and
  overriding that is exactly what hurts. (Same budget wall, viewed from the salience side.)

### The budget hypothesis — TESTED AND REFUTED (`scratchpad/breadth_lab.py`, 2026-06-27)

I hypothesised broad's wall was the **fixed 2000-token budget** (contexts run 75–98% full,
5/11 truncate) and built the two levers to "fit more": (1) fragment compression — cross-fragment
sentence-level dedup; (2) breadth-budget reallocation — lower `depth_share`, raise frame cap.
A no-op control (dedup_tau=1.01) reproduces stock context byte-for-byte, so the harness is faithful.

**Both refuted. broad = 0.3636 is INVARIANT under every downstream knob:**

| lever | broad | note |
|---|---|---|
| compression (dedup_tau 0.97→0.72) | 0.364 | frees space on NON-truncated queries; truncated ones (b_religion, b_ideas) have ~no dup sentences to drop — assembly already deduped the pool |
| breadth budget (`depth_share` 0.30→0.0) | 0.364 | even breadth getting 100% of specifics |
| frame cap ↓ (0.35→0.20) | 0.364 | only *hurts* narrow/para (0.786→0.714, 0.875→0.750) |
| materialize cutoff (`res_materialize_nodes` 12→45) | 0.364 | surfacing 4× more bright nodes changes nothing |

**The truncation was a red herring** — contexts truncate, but the missing broad facts are not in
the truncated tail; reallocating/compressing budget just rearranges fragments that are all "not
the covering one." So broad is **not** budget-bound, not selection-cutoff-bound, not reach-in-
general bound.

### The corrected diagnosis: broad is bound at the ACTIVATION FIELD (upstream of everything)

broad = 0.3636 is invariant under bridges, budget, compression, and the materialize cutoff —
i.e. under **everything downstream of the activation field**. The covering fragments for the
failing broad queries simply **aren't in the surfaced set**, because either their concepts never
light up (R0 vocabulary-disjoint seeds — query words don't overlap note words) or the verbatim
note text doesn't cosine-match the abstract broad fact (embedding meaning limit, τ=0.54).

This finally unifies every broad result in the project: **the only lever that ever moved broad
(B's redundant memberships, 0.364→0.455) worked by changing the FIELD** — extra members make a
concept brighter, pulling in fragments no downstream knob can reach — and it taxed narrow because
brightening costs precision. Bridges, budget, compression, cutoff are all downstream of the field
and therefore inert. The remaining levers must act on the field/sensor: a query-aware or
asymmetric **embedding** (so abstract broad queries match verbatim notes), or **usage content**
that re-shapes which concepts co-activate — not anything in retrieval assembly.

### Anatomy of broad failure — what can actually solve it (`scratchpad/broad_anatomy.py`, 2026-06-27)

To answer "what can solve broad," decomposed each of the 44 broad facts by an ORACLE (best
cosine of the fact vs ALL claim chunks in the corpus) vs the in-context score:

| mode | facts | of the 17 FAILURES | fix |
|---|---|---|---|
| COVERED (≥τ in context) | 27 (61%) | — | — |
| **REACH_MISS** (corpus-best ≥τ at 0.57–0.85, context-best ~0.31) | 10 | **59%** | close abstract-query→concrete-claim gap at the SEED |
| **SUBTHRESHOLD** (in context at 0.46–0.53) | 5 | **29%** | asymmetric scoring / τ; or LLM-judged-covered anyway |
| NOT_REPRESENTABLE (corpus-best <τ) | 2 | 12% | nothing — genuinely unrepresentable |

**This CORRECTS the "embedding meaning limit" framing above: 88% of broad failures are fixable —
the embedding represents the content fine (oracle 0.57–0.85); it just isn't surfaced.** The
reach-miss pattern is structural: broad queries are **abstract** ("the various ideas I explored"),
covering claims are **concrete** ("a marketplace for X"). Concrete claim is FAR from the abstract
query (never seeded) but CLOSE to the concrete fact (oracle high). MiniLM is **symmetric** — it
maps a terse question and a verbatim passage into one space and expects a match; they don't match.

**What can solve it (ranked):**
1. **Asymmetric / instruction-tuned embedding** (e5/bge/gte with `query:`/`passage:` prefixes).
   Directly attacks both fixable modes: better query→claim seeding (the 59%) and lifting borderline
   scores over τ (the 29%). The oracle makes this a **measured prediction, not a guess** — content
   is matchable at 0.57–0.85; what's missing is a mapping that puts abstract queries near concrete
   notes. (= the "richer features" lever from `docs/geometry-membership-handoff.md`, now quantified.)
2. **Corpus-grounded query expansion** (no model swap): expand the abstract query into the user's
   OWN related concept labels/claims, seed from those. A HyDE in real vocabulary, not generic
   (generic HyDE was refuted — it seeds off-target). Can reach part of the 59%, not the 12%.
3. **τ / asymmetric scoring** for the 29% subthreshold — cheap, partial, and arguably a metric
   conservatism (τ=0.54 is a proxy for the LLM judge, which may score these covered).

**Why no retrieval-side lever can do it** (all tested null: HyDE, decomposition, enumeration,
budget, materialize cutoff): they all operate INSIDE the symmetric embedding space and cannot
manufacture a query→claim proximity the embedding doesn't encode. Only a better sensor (1/2) or
field-reshaping content can. Side-finding: `res_frame_claims_per=12` gives paragraph **1.000**
(mean 0.6926) — but it's another narrow↔para reshuffle (still 22/33 raw), not net coverage.

---

## Eval expanded 33 → 38 (2026-06-27)

Golds grew **14/11/8 = 33 → 16/13/9 = 38**: appended 5 grounded golds authored *note-first*
from previously-uncovered notes (brain, reservation, taxation, social contracts, UX, destiny),
never from retrieval output (keeps the eval non-circular). IDs: narrow `g15`/`g16`, broad
`b_fairness`/`b_systems_mind`, paragraph `p15`. Every key_fact grounds ≥0.60 to a real corpus
claim (validated via `broad_anatomy.py`'s oracle). The full 2x (+33) was scoped but declined —
130+ notes remain untapped if revisited.

**Re-based baseline (38 queries):**

| | narrow | broad | paragraph | mean |
|---|---|---|---|---|
| Bar | 0.750 | 0.308 | 0.778 | 0.6118 |
| **A (channel-code)** | 0.750 | 0.308 | **0.889** | **0.6489** |

**A's win REPLICATES** (+0.037, entirely paragraph, narrow/broad held) — robust to 5 fresh golds,
not a one-probe artifact. All numbers ELSEWHERE in this doc are the pre-expansion 33-query values
(bar 0.6331 / A 0.6748); re-measure on 38q going forward.

---

## Session arc — every lever tried, in order (2026-06-27)

The whole session in one table. Bar = **78.6 / 36.4 / 75.0, mean 0.6331** (22/33 raw at
narrow 11/14, broad 4/11, para 7/8). The eval-DB the bar is measured on had to be run through
the shipped medoid+contrastive write path first (the raw DB sits on stale centroids → 71/27/50).

| # | lever | result | verdict |
|---|---|---|---|
| 1 | **A — channel-code redundancy** (33 coverage-hole parity attaches) | 78.6 / 36.4 / **87.5** = **0.6748** | **WIN** (+para, verified, no collapse/leakage) |
| 2 | B — Hebbian usage as membership | 64.3 / 45.5 / 75.0 = 0.6158 | moves broad, taxes narrow — trade |
| 3 | C — facet featurizer | ties bar | refuted (inert on these golds) |
| 4 | D — free-energy / MDL | ties bar | refuted (only its redundancy term = A) |
| 5 | E — surprise-weighted centers | 78.6 / 27.3 / 75.0 | refuted (centers inert; membership-edge is what's read) |
| 6 | A + C (facet-space hole test) | 0.57–0.61 | worse than A (breaks the 33-band) |
| 7 | A + B union (+ inflow cap, degree-fix) | **0.6813** | beats A on MEAN but same **22/33** raw — reshuffle |
| 8 | HyDE (hypothetical-passage seed) | broad 0.27 | refuted (generic seeds crowd budget) |
| 9 | Query decomposition (facet sub-seeds) | broad 0.27 | refuted |
| 10 | Hebbian as bridges (L3) | inert / harmful | refuted (distinctiveness term suppresses) |
| 11 | Compression (sentence dedup) | broad 0.364 | refuted (no dups to squeeze where it matters) |
| 12 | Breadth budget / materialize cutoff | broad 0.364 | refuted (broad isn't budget-bound) |
| — | `res_frame_claims_per=12` | para **1.000**, mean 0.6926 | side-find; still 22/33 (narrow↔para reshuffle) |

**Two hard ceilings discovered:**
- **22/33 raw coverage.** A and every A+B variant top out here. Mean can be reshuffled between
  axes (narrow↔broad via inflow cap; narrow↔para via frame enumeration) but total passes don't
  rise. Net-new coverage needs broad, and broad needs a better sensor.
- **broad = 0.3636** under every retrieval-side lever. Anatomy proved 88% of broad failures are
  *fixable* (oracle 0.57–0.85), dominated by abstract-query↔concrete-claim mismatch in a symmetric
  embedding. **The one untried lever that the data predicts will work = an asymmetric / instruction
  embedding (e5/bge/gte, query:/passage:).** PARKED for another day.

## Status & next steps

**SHIPPED into `core/` (2026-06-27).** Strategy A (channel-code redundancy) is now wired into
production consolidation + retrieval. Verified, reviewed, 220/220 tests pass.
- `core/store.py`: `concept_members.kind` column ('primary'|'redundant') via idempotent migration
  + base schema; `concept_member_ids(primary_only=)`; `set_channel_redundancy()` (idempotent
  replace); `recompute_concept_embedding` + `claim_in_any_concept` are primary-only.
- `core/consolidate.py`: `_channel_holes` + `_channel_redundancy` (step 6c, after
  `_reanchor_concepts`, before `_fit_baselines`, gated on `CHANNEL_REDUNDANCY=True`); a
  `CHANNEL_REDUNDANCY` event + `apply_event` handler (event-logged → rebuild-reproducible,
  rollback-reversible); every consolidation-internal `concept_member_ids` read is primary-only
  (a no-op when no redundant rows exist, so the default path is unchanged).
- Retrieval needs NO change — resonance reads all members, so it consumes the redundant attaches
  automatically; A's 33 attaches lift paragraph without taxing narrow (no degree-fix needed).
- Verification (`scratchpad/verify_channel_prod.py`): eval lift reproduces exactly through prod
  code (n_red=33, paragraph 0.778→0.889 on 38q, hole-set identical to scratchpad jaccard 1.0);
  idempotent; centers unmoved by redundancy; rebuild 33→33, rollback 33→0; prune-vs-redundancy
  regression PASS. Adversarial 4-lens review found one real bug (`claim_in_any_concept` counted
  redundant rows → could orphan a pruned claim across runs) — fixed + regression-tested.
- **Not yet applied to live `engine.db`** — shipped code only affects FUTURE consolidation runs;
  existing users keep their current partitions until the next run (or a one-time backfill).
- Optional add-on (NOT shipped): A+B with the **claim-only degree-fix** (`sec_proto.py`) or
  **inflow≤2 cap** — reaches 0.6813 mean if you value broad over narrow, but it's a reshuffle, not
  net coverage.

**PARKED — the real lever for net gains (broad):** re-run `broad_anatomy.py` + Coverage@B with an
**asymmetric embedding** (e5/bge/gte via the HF inference API — scratchpad test first; prod host is
HF-API-only, no torch). If reach-miss (59%) + sub-threshold (29%) collapse, broad is solved and the
production embedding swap is justified. Pre-check: spot-check the 29% sub-threshold facts against a
real LLM judge — τ=0.54 may already under-credit them (metric conservatism).

---

## Reproduce (scripts)

Run with `PYTHONPATH=.:scratchpad .venv/bin/python`. Corpus `/tmp/slate_eval.db`, user
`usr_01KTXAYR20J4R6F7PT3DP10W3W`. The lab copies the corpus per run, so nothing is clobbered.
Bar reproduced via `strat_lab.py` (medoid+contrastive write path), NOT the raw DB.

**Strategy team (the race):**
- `scratchpad/strat_lab.py` — shared test bench: load claims, set membership/redundancy, score
  Coverage@B over all 3 golds via the production resonance path. Establishes the bar.
- `scratchpad/strat_A_channelcode.py` — **the winner.** Channel-code redundancy.
- `scratchpad/strat_B_hebbian.py` — usage co-activation (`round2` = A+B floor-ranking).
- `scratchpad/strat_C_facets.py` — facet featurizer (`chan` = A+C).
- `scratchpad/strat_D_freeenergy.py` — MDL + redundancy.
- `scratchpad/strat_E_surprise.py` — surprise-weighted centers.
- `scratchpad/strat_SYNTH.py` — combined pipeline (equals A; ablations prove nothing adds).
- `scratchpad/consolidation_team.workflow.js` — orchestration (R1 → blackboard → R2 → synth → verify).

**A+B narrow recovery + retrieval split:**
- `scratchpad/strat_AB.py` — A∪B union + inflow-cap variants.
- `scratchpad/narrow_recover.py` — inflow-cap / cohesion-cap sweep (membership-level fix).
- `scratchpad/sec_proto.py` — secondary-edge retrieval split; the claim-only **degree-fix** is here.

**Broad investigation:**
- `scratchpad/hyde.py` — HyDE (augment/replace) + query decomposition.
- `scratchpad/query_expand.py` — corpus-grounded query expansion (PRF / concept-label) — refuted.
- `scratchpad/hebbian_bridges.py` — Hebbian co-activation as L3 bridges.
- `scratchpad/bridge_tune.py` — WHY bridges are no-op (distinctiveness, not PE-gate) + tuning sweep.
- `scratchpad/breadth_lab.py` — compression + breadth-budget reallocation (faithful no-op control).
- `scratchpad/broad_anatomy.py` — **the diagnosis**: decomposes broad failures into
  reach-miss / sub-threshold / not-representable via a corpus oracle.

## Follow-up: bridges removed, and the channel-code depth/locality questions settled (2026-06-27)

After the race shipped channel-code, we closed out the **Bridges** layer entirely and pinned
down *how much* redundancy to add and *how* to gate it. Three results, all on the 38q eval.

### Bridges are dead — removed from prod
The shipped "bridge" was a **center-proximity** link (two concepts whose medoids sit in a
residual band). It does nothing:
- **Ablation at B and 2B is byte-identical.** Deleting all 47 bridge relations changes
  Coverage@B by exactly zero at the 2000-tok budget *and* at 4000. Not even a budget-crowding
  effect — the bridge targets never rank into the assembled set either way.
- **The corrected definition doesn't save it.** We rebuilt "bridge" as the *structural hole*
  the user actually meant — a claim that spans **two far-apart concepts** (centers mutually
  distant, so they'd never merge). Tested two ways:
  - **As a `bridges` relation** (claim→concept *and* concept→concept): inert — identical to
    bar. So the dead part is the **bridge-relation mechanism itself** (a concept-layer edge is
    damped by the distinctiveness term `1/(1+ln deg)^γ` and never reaches the top-12 cutoff),
    not the near-vs-far definition.
  - **As membership**: *regresses* (narrow tax, no paragraph gain). Why: a claim
    "reconstructable from both" far concepts is **blind duplication** — it carries info both
    already have. That's the `budgeted_flat` baseline the race already refuted. Channel-code
    wins precisely because it attaches **unreconstructable** holes (z∈[1,3]), not shared ones.
- **`BRIDGE_BOOST` (×1.3) is an inert knob.** Wiring the 33 channel attaches as *boosted*
  bridge relations scores **identically** to plain membership (weight 1.0). The win is binary:
  connect the spanning claim to the reachable concept and the paragraph gold surfaces; the
  extra edge weight can't surface a *second* thing because nothing else is eligible. Gating is
  the PE/distinctiveness field, not edge-weight magnitude.

→ **Removed in prod**: `_bridges` generation, the `BRIDGED` event path, `BRIDGE_BOOST`, the
`list_bridges` MCP tool, and the recall/digest 🌉 read surfaces. `synthesize()` survives as a
general two-concept drafter. Channel-code redundancy (membership) is the *only* structural
lever that moves Coverage@B.

### Depth of redundancy is threshold-driven, not a basis count
- The per-claim cap `R_MAX=1` is **inert**: the gate yields 33 eligible pairs across **33
  distinct claims** — no claim has two simultaneous holes (a claim close enough to a second
  center is usually already reconstructable there, or too far). Setting `R_MAX` to 2/3/99
  changes nothing.
- So depth is set by the **z-band threshold** (which self-limits to ~33), *not* by a count.
  A fixed basis count is refuted both ways: the blanket θ=1.0 multi-membership (3862 attaches,
  ~12/claim) floods and taxes narrow; scarce threshold-gating is the win. "Channel coding spends
  parity bits only on the fragile symbols" — and the threshold is what finds them.

### The reachability floor should NOT be localized
We asked whether `cos≥0.55` should be a *local* threshold (relative to each concept's density)
rather than absolute. Tested member-spread-relative, margin/Voronoi, and rank-only floors —
**every local form regresses to bar.** The headroom diagnostic explains why in one number:
**99.7% of z-band holes are already below cos 0.55** (33 of ~10,900), and the excluded ones
live in **normal-density** concepts (μ_cos 0.62–0.75, median 0.68 ≈ the global 0.70 median),
**not** sparse pockets. So `cos≥0.55` isn't really a "reachability accuracy" gate — it's a
**scarcity/precision** gate, and its value is *staying at 33*. Relaxing it (local-spread,
rank-only) explodes the pool to 160–10,900, inflates concept degree, and crowds the covering
fragment out of budget → paragraph win dies. Tightening it (margin) admits only 11, and not
the gold-covering ones. The signal where locality *does* help — the hole test `z` — is
**already** local (z-scored to each concept's own LOO spread).

**Still open:** expressing depth as a scarcity **rate** (top-X% of holes by cos·z, or ∝
n_claims) rather than an absolute z-band, so "33" generalizes to larger corpora. Untested.

**New scripts:** `scratchpad/struct_hole_bridge.py` (corrected-bridge test, relation vs
membership), `scratchpad/channel_locality.py` (local vs global reachability floor + headroom).

## Related
- `docs/geometry-vs-llm-membership-findings.md` — prior thread; its bridge "PE-gate" claim is
  corrected here (the suppressor is the distinctiveness term).
- `docs/consolidate-contrastive-handoff.md` — the shipped medoid + contrastive center work.
- Memory: `slate-channel-code-redundancy` (the full close-out), `slate-geometry-vs-llm-membership`,
  `slate-consolidate-contrastive`.
