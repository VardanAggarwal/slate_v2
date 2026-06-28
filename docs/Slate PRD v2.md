# Slate PRD v2

## Why
LLMs are great at processing content, especially text. But more the content to process, noisier the outcome. In a world where attention is all you need, attention on the right things matters the most. The goal for Slate is to reduce the amount of content an LLM needs to process while still ensuring that everything that deserves attention is covered.

## Things to consider
### Possible alternate memory structures
1. Native md style used by Claude — too much generalisation or too long files.
2. Grep and surrounding text — exact text match dependency, but semantic keywords can be used for multiple searches.
3. RAG — closest to the desired model but scales badly. Most of the content being processed ends up too close to each other in a larger universe, leading to saturation with generic content. The same would happen with Grep on certain keywords.

### What kind of situations to handle
- Reducing repeated context without losing salience. Common themes repeat often, but each instance carries some additional nuance that might deserve attention.
- Being able to drill down based on how specific the information needs to be. Often handled with projects, clusters, decision trees — but ideally more free-flowing, like a graph, while being able to decide when to stop navigating further.

### Where is cost
- All LLM calls — tokens exchanged, what is pushed and what comes back.
- Embedding calls — input tokens.

### What is stored vs processed
- Raw content is stored as it is, and should never have to be processed.
- Pre-processed structured data and various indexes are what get processed for retrieval and consolidation.

## The core principle
At each stage, are we able to figure out where to dedicate attention to?

## North Star
Slate succeeds when it answers a query as well as a ground-truth reference would, within the token budget it is given.

**Sufficiency Rate @ Budget (SR@B):** the share of queries for which Slate, using at most budget `B`, produces the same correct answer as the reference — *credited only when amortized build-cost stays within the cost gate* (below).

- `B` is set by the host — a slice of the context window (~10% of a 200k window, ~5–10% of a 1M window). Slate is told `B` at call time and must answer within it.
- **The reference is the ground-truth answer, not Slate's own and not, by default, the full corpus's.** Where a gold answer exists, use it (human-adjudicated). The full-corpus answer is a *floor, not the target* — Slate may legitimately beat it by synthesising across fragments, and matching a wrong full-corpus answer is not success. For corpora too large to read in one pass, the reference is a **retrieval oracle: RAG/grep run at 3× the budget (3B)**, taken as the best answer reachable with a generous read. Queries the corpus genuinely cannot answer are tagged and excluded.
  - Note the role split, decided purely by budget: **RAG/grep at 3B are the *reference* (privileged read, an oracle proxy); RAG@B and grep@B are *competitors* (same budget as Slate, scored against that reference).** The same tools cannot be both — a budget-matched retriever as truth would be circular.
- **"Same answer" is binary**, scored against a *pre-registered key-facts checklist* per gold query — fixed, not judge-picked per run — by two judges with a tiebreak, reporting inter-rater agreement (κ). A graded score is kept only as a tiebreaker.
- Reported **overall and on a hard/tail slice**, where the **tail is the gate and the head a floor** — you may not trade the hard cases for easy ones.

**Parsimony is enforced, not assumed.** SR@B on an isolated query does not punish filling the window — so we make padding cost. Evaluate at a **budget distribution (B/2 and B)**, and withhold full credit when a smaller assembly would also have passed. Over-injection that flips an answer is already a failure; padding that merely crowds the window now costs too.

**Faithfulness folds in:** an answer that contradicts the source cannot match the reference, so an unfaithful answer is already a miss. (A separate *store-integrity* check — derived memory that contradicts the raw record without changing today's answer — stays inside consolidation, not as a headline metric.)

**The cost gate.** SR@B is credited only while amortized build-cost (write + consolidate tokens, measured **per query at steady state**, not cumulative from zero) stays within `k×` a near-zero-build baseline. A score *bought* with unlimited precompute cannot be cheapened later without losing it — so cost is a gate from the start, not a deferred clean-up.

**Measurement — two query sets, never mixed.** If the same queries both *teach* the system and *score* it, the score lies — Slate memorises its own test, and a shifting test makes a rising number meaningless. So we split them, like a fixed exam versus a practice pile:

- **The frozen benchmark — the scoreboard.** A fixed, stratified set of queries that Slate is *never* tuned on. It is the reported headline (with confidence intervals). Because the questions don't change, a rise in SR@B is real improvement, not an easier batch — which is also why ΔSR@B and the forgetting rate are computed *only* here, on the identical queries before and after a run.
- **The adaptive probe set — the practice pile.** Constantly refreshed from production failures and the long tail. This is where consolidation learns and where new weaknesses are hunted. It is never trended and never the headline — its difficulty shifts every cycle, so a number off it would mean nothing.

The two run on **different clocks**. Consolidation runs every cycle, learning fast from the probe set. The frozen set changes *slowly* — only when a vetted batch of hard cases graduates from probe into it, stamped as a new version. Between versions it stays fixed, so cycles stay comparable; at each version bump the exam gets harder and more representative, and the comparison baseline resets to the new version. The flow: a failure surfaces in the probe set → consolidation fixes it → the frozen set confirms the fix *generalised* rather than just patching that one query.

## How SR@B breaks down by stage — diagnostics, not a factorisation
Information about the answer can only be *lost* as it flows raw → written → consolidated → retrieved; no stage adds back what an earlier one dropped. So each stage has a no-loss diagnostic. These do **not** cleanly multiply into SR@B — consolidation rewrites memory every cycle, so Write's content-presence rate itself drifts, and the rates are measured on different conditioning sets. Treat them as conditional rates that *locate* a regression, not as an exact decomposition. **Every stage is gated on end-to-end SR@B**, never on its own sub-metric alone — otherwise a stage can show green while the whole stalls.

- **Write — content presence.** Is the answer-relevant content present in the fragments at all? Whatever Write drops cannot be retrieved later (the raw episode survives for re-derivation, but not for retrieval). A presence check, reported as a diagnostic — but because Write is gated on SR@B *at budget*, it cannot game the number by emitting more, finer fragments: that raises presence while crowding retrieval, which lowers SR@B.
- **Retrieve — sufficiency @ budget.** Of the queries whose content *is* present, what share does the budgeted assembly answer — at the assembly size that *maximises* the answer, not the one that fills `B`? Misses, over-injection, and bad stopping land here.
- **Consolidate — ΔSR@B per run, on the frozen set.** Consolidation reshapes memory between cycles; its metric is the *change* it causes — SR@B after minus before, measured only on the frozen benchmark (100% overlap, so it is comparable). Net positive means it earned its tokens — usually by raising retrieval-sufficiency (better structure surfaces the right thing in fewer tokens, so more fits the budget). Its hard floor is the **catastrophic-forgetting rate**: previously-succeeding frozen-set queries that now fail. A nuance-averaging merge or an over-eager prune shows up here. A run must lift SR@B without regressing the set.

Each cycle the eval recomputes all three *after* consolidation (so the drifted rates are current) and reports the residual between the conditional rates and observed SR@B, rather than assuming it is zero.

---

## The signal underneath everything: prediction error
One signal underlies fragment boundaries, salience, and what to consolidate: **prediction error** — how much a new piece of content differs from what memory already expects.

The rule it gives is **predictive coding**: before encoding anything, ask what memory already predicts, and **store only the residual — the part it got wrong**. The predictable part is not stored again.

But "predictable" does not mean "ignore." A prediction that comes true is *evidence for the memory that made it*, so that memory is **reinforced** — confidence and recency bumped — with nothing new stored. Encoding lays down new content; reinforcement strengthens what is already there. A confirmed prediction does the second, not the first.

So every incoming piece routes one of three ways:
- **Predicted** → reinforce what predicted it; store nothing.
- **New, and it fits** what we already know → store it, attached to that theme. Cheap.
- **New, and it clashes** with what we know → the expensive case: either a real discovery worth restructuring for, or noise to reject. It gets a judgement, not an automatic write.

The same signal answers each downstream question:
- Where to cut a fragment — where content stops being predictable.
- How much to encode — only the residual; how much it matters is how much it surprised.
- What to revisit later — the surprising things, first.
- What to let go of — when something stops being surprising on repetition, it has become background.

How prediction error is computed is committed in **§How: the predictor** below; what is fixed here is its *shape* — **predict, then diff** — and that the same signal is reused everywhere. This is the spine the rest of the design hangs from.

## Two kinds of memory
Two stores, with different rules, because they do different jobs.

- **Raw memory** is the record of what actually came in. It is never edited, never merged, never thrown away. It is the ground truth.
- **Working memory** is everything we derive from it — the fragments, the themes, the links, the weights. This is what gets processed, and it is allowed to be wrong, because it can always be rebuilt.

This split protects against saturation: merging or reweighting working memory risks collapsing things that should stay distinct, but the untouched raw record lets anything corrupted be re-derived from what was originally said — not from a log of past decisions, which would repeat the mistake. So "store raw, never process it" is not a storage detail — it is what lets the rest of the system take risks.

---

## Write
The first stage, where a new piece of content is pushed. A long piece has to be:
- broken into smaller independent pieces,
- with relationships kept only where they add critical value,
- and checked against existing memory to see what matches something already known, and what is genuinely new and deserves more attention.

The entire focus is on identifying what *in this particular content* deserves attention. The boundaries between fragments fall where the content stops being predictable. How strongly each fragment is held is set by how surprising it is. A piece that contradicts something already believed is the most surprising thing that can arrive, and is held the strongest — and flagged for later resolution.

**Owns — content presence:** every answer-relevant fact must survive into the fragments, or it is lost to retrieval for good. Boundaries and salience are judged by end-to-end SR@B *at budget* — not by unbudgeted reproduction, so Write cannot win by emitting more, finer fragments that then crowd retrieval.

## Retrieval
A query has to be decomposed. Typically it covers some background plus some specific nuances. A nuance might be covered in a different context and be easily borrowed here; a deep dive into background might add depth to an otherwise shallow problem.

The goal is *not* to fetch in everything. It is to focus on what needs further elaboration — what is incomplete or would benefit from more depth, given the query. That is a function of two things: what matters here, and what memory can actually provide. Each fragment of the query gets treated on its own:
- where memory already covers it, there is nothing to add — drop it,
- where memory can deepen it, fetch that,
- where memory can't help, return nothing rather than padding the context.

And there has to be a point where we stop navigating. We keep pulling on a thread only while the next step is expected to add more than it costs. When the marginal gain falls below that bar, we stop. Decomposing the query is what *lets* us ask this question per fragment; it does not answer it on its own.

Every retrieval also leaves signals — what was fetched, what was dropped, what got cut for budget. Those are not discarded; they feed consolidation.

**Owns — sufficiency @ budget:** of what memory holds, surface enough to answer within `B`, and stop at the point that *maximises the answer* — not the point that fills the window. Over-injection that dilutes attention is a failure, not a cost.

## Consolidation
Every write and every retrieval creates signals — what repeated, what was new, what deserved attention, what didn't. Consolidation processes both, and folds them back so future writes and retrievals aim attention better. It runs as a batch job, offline, like sleep — never inline, so signals settle instead of thrashing. It spends its effort where surprise was highest.

It is the stage where the structure actually changes, so it is also where the risks live:

- **Forgetting is a feature.** Without it the store only grows, and saturation — the exact thing we are fighting — comes back from the inside. One-off things that never recur are pruned. Things that have become pure background are folded into the theme they belong to.
- **But we never lose a nuance.** Forgetting and merging are governed by one rule: only let go of what the remaining structure can reconstruct. If a piece carries a nuance that would be lost when it is folded in, it stays standalone.
- **Beliefs get reconciled, and versioned when they can't be.** When two things genuinely conflict: if one supersedes the other on better or newer grounds, the old one is marked as past, not deleted. If they are both true under different conditions, each is scoped to its condition — it was never really a conflict. If they genuinely both stand, both are kept as versions of one belief, one current and the rest held. We never average a conflict into mush, and we never silently drop the loser. At retrieval, the current view is returned, but the fact that it is contested is always surfaced. A belief that flips back and forth must clear a margin before the current view changes, or it oscillates forever.
- **Bad runs must be undoable.** Because this is the one stage that can corrupt the structure, a consolidation run has to be reversible — undone back to the state before it, and in the worst case re-derived from the raw record.

The retrieval signals close the loop here: was something treated as salient but never actually retrieved (demote it)? Was something dropped that should have surfaced (promote it)? Did a held belief get contradicted (resolve it)? One caution: "wasn't retrieved" is not the same as "wasn't useful" — rare-but-correct memories must not be suppressed just for being quiet. The external yardstick matters more than raw retrieval frequency.

For a personal memory engine, versioning is not just cleanup — the trail of how a view changed, and when, is itself signal.

**Owns — ΔSR@B per run, floored by the catastrophic-forgetting rate:** a run must lift SR@B (or its parsimony) without dropping any query that previously succeeded. This is also the stage where the SR@B eval is run each cycle, on the frozen set — Slate vs RAG vs grep against the gold/3B-oracle reference.

---

## What this model still has to address
These are part of the *what*. Two now have a mechanism in §How (noted inline); the rest are not yet pinned down:
- **Cold start.** On an empty store nothing can be predicted, so everything looks maximally surprising. §How gives the shape of the answer — trust the residual in a region only once it holds enough members for its spread to be stable — but the warm-up curve itself, and the per-region graduation, still need pinning.
- **Deletion and privacy.** The raw record is immutable on purpose, but a person must be able to say "forget my note about X." §How supplies the *propagation* rule — a derived memory survives a redaction only if it is still reconstructable from the remaining raw — but the sanctioned redaction mechanism itself, which the rebuild path must honour, still needs building.
- **Many people, one engine.** Memory is per person. Isolation between users is a requirement, not an afterthought.
- **Writing while consolidating.** New content arrives while a batch run is in progress. What that run sees, and what it ignores until next time, has to be defined.

## How: the predictor
Everything above deferred one decision — how prediction error is actually measured. It is now committed, and its **write-side behaviour validated against the live corpus**; the retrieval and consolidation uses below are designed on the same primitive but not yet measured. One estimator answers every question the spine raises, because each of those questions turns out to be the same question asked of different inputs.

### The functions it must serve
Before the mechanism, the work it has to do. **Status** legend: `built` — the primitive does it today; `wrapper` — the primitive exists, needs a loop/scan/guard around it; `LLM` — the resolver's job, not the predictor's.

**Write**

| Function | Status | How |
|---|---|---|
| Define logical chunks | wrapper | residual run *sequentially* — cut where the next unit stops being predictable from the running span. Primitive built, scan loop not. |
| Variable-resolution encoding | wrapper | not just *where* to cut but *how finely* — fine where surprise is high, folded where low. |
| Match against corpus | built | `measure()` nearest anchor → `decide()` route. |
| Intra-note dedup | built | run the residual against *earlier fragments of the same note*, so within-note echoes collapse before they hit the store. |
| Label store / don't-store | built | PREDICTED → reinforce (store nothing); NOVEL → store; AMBIGUOUS → store + flag. |
| Hold-strength / salience | built | relative weight + rank within the batch — never an absolute magnitude. |
| Which theme it attaches to | built | the anchor pointer. |
| Centre / essence | built | lowest-residual fragment = the note's core; highest = its most novel point. |
| Distil the claim text; sign a contradiction | LLM | the resolver's, not the predictor's. |

**Retrieve** — here the predictor *inverts*: run it role-swapped (X = a memory item, Y = the query), and the same signal answers "fetch or drop."

| Function | Status | How |
|---|---|---|
| Break query into independent fragments | wrapper | the same sequential residual scan as Write chunking, run on the query. |
| Drop / deepen / return-nothing per fragment | built | residual of *memory against the query fragment*: ≈0 → memory only echoes the query, **drop**; >0 & anchored → memory adds depth, **fetch the residual**; no anchor → **return nothing** (don't pad). The three cases above. |
| How deep / when to stop | wrapper | iterate: each candidate's residual against the *assembled-so-far*; when the next item is PREDICTED by what's already pulled, **stop**. VOI = marginal residual. |
| Borrow nuance cross-theme | built | the anchor needn't be same-topic; a low-residual fit *across* a theme boundary is transferable nuance. |
| Per-fragment budget allocation | wrapper | split `B` across fragments by how much residual each can fill. |
| Prioritise final context | wrapper | greedy **max-marginal-residual** — rank by unique information per token, drop items already reconstructable from the chosen set. |
| Answerability triage | built | whole query NOVEL (high residual, no anchors) → stop before fetching; saves the retrieval. |
| Read / synthesise the fetched text | LLM | |

**Consolidate** — where the primitive pays off most: the merge/forget nuance-loss is a reconstruction-residual question by definition.

| Function | Status | How |
|---|---|---|
| Dedup / canonicalize | built | "already represented?" = PREDICTED route. Replaces the cosine bands. |
| Concept attach (where a claim belongs) | built | nearest anchor + anchor-present, against concept medoids. |
| Concept split / create | wrapper | a concept's cohesion (member-residual baseline) is precomputed; high internal residual → members don't reconstruct each other → two themes → SPLIT; low → coherent. |
| MERGE safety (the nuance guard) | wrapper | before folding X in, check X's residual against the *post-merge* structure — low → reconstructable, safe to fold; high → carries a nuance the merge would erase → keep standalone. Fixes "merge averages, loses nuance." |
| Forget / prune safety | built | leave-one-out residual against the rest; low + old + quiet → prune; high → irreplaceable, protect even if quiet. |
| What to revisit first | built | rank by surprise; AMBIGUOUS items (unresolved residuals on anchors) are the priority — they need the LLM. |
| Background detection (decay) | wrapper | track a claim's surprise over repeats; when it trends to PREDICTED, it has become background → fold into theme. |
| Concept drift / re-anchor | wrapper | members' residual against the *current* medoid creeping up → drift → re-anchor the medoid or split. |
| Bridge candidacy | ~~built~~ **removed 2026-06-27** | was a residual band between concept medoids. Removed: proven retrieval-inert (no Coverage@B effect at B or 2B, concept- or claim-edge level). The structural win lives in channel-code redundancy instead. |
| Conflict / contradiction detection | built | AMBIGUOUS + opposite-stance anchor flags *where* reconciliation is needed. |
| Contradiction clustering / oscillation margin | built | group flags on one anchor into one reconciliation; detect a belief that flip-flops over time. |
| Store-integrity check | built | residual/route between a derived claim and its source episode. |
| Corpus saturation / health | built | track the global residual distribution over time — falling average = everything becoming predictable = saturating. |
| Reconcile / version a conflict (which supersedes, scope it, "real bridge?") | LLM + schema | the predictor detects, never resolves. |

**Cross-cutting**

| Function | Status | How |
|---|---|---|
| Regional warm-up graduation | built | cold-start is per-region: trust residual routing once a region has enough members + stable cohesion; sparser regions stay in warm-up. |
| Redaction propagation | wrapper | after a raw delete, re-check which derived claims are still reconstructable from the remaining raw; drop those that aren't — the rebuild path honours the deletion. |

*The one honest limit:* retrieval-frequency signals ("demote what's salient but never retrieved") are **not** the predictor — they are logged usage data that feeds consolidation *alongside* it, never from it, because rare-but-correct memory must not be suppressed for being quiet.

### How the predictor works
One primitive underlies all of it: **the residual of X against a context Y** — how much of X cannot be reconstructed from Y, judged relative to Y's own internal spread.

- **Predict.** Reconstruct X from its nearest neighbours in Y, trusting the closest most — so a fragment counts as predicted only when things *genuinely near it* can rebuild it, not when a combination of distant neighbours happens to span it. What they rebuild is the predictable part.
- **Diff.** The residual is what is left — the part those neighbours could not rebuild. That is the surprise, its size the magnitude.
- **Calibrate to spread, not to an absolute.** A fixed similarity cutoff fails, because the same distance reads as "near-duplicate" in a dense theme and "loosely related" in a sparse one. So the residual is scored as a z-score against the residuals of Y's *own* members — how well the region reconstructs itself, leave-one-out — shrunk toward a prior where a region is too thin to judge alone, and where the corpus spans several domains, toward a prior over *clusters* rather than a pooled average, so a small domain is not dragged toward a large one. This makes the residual *comparable* across regions; where the routing line is then drawn on that comparable scale is a separate choice — the calibration, defined just below.
- **Locate.** The nearest member is the **anchor**: the real piece of text the residual sits on. Memory is never decoded back out of a vector — the residual always points at a sentence that already exists (selection, not generation). A fragment is treated as sitting *on* memory only when it is as close to its anchor as the region's members are to each other; otherwise it is in open territory.

Three routes fall out:
- **Predicted** — residual within the region's normal spread. Nothing new; reinforce what predicted it.
- **Novel** — residual beyond what any region explains, with no anchor present. Genuinely new; store it.
- **Ambiguous** — a real residual that sits *on* an existing anchor. Store it, and flag it: it may contradict, refine, or version what it sits on.

One boundary is absolute: **the predictor measures magnitude and structure; it never decides direction.** Whether an ambiguous residual contradicts its anchor or merely refines it is *geometrically identical* — both are "close but not the same." Sign is a question of meaning, so it is handed to a separate resolver, and only for the ambiguous cases. The predictor cannot be wrong about a contradiction because it never claims one — which is exactly what lets this cheap layer run on everything: it only ever measures.

### Two layers: measure, then decide
The predictor is a **pure sensor**. It reports the residual, the anchor, and how the local region is spread — and stops there. It holds no thresholds and makes no store-or-forget decision, because the residual answers only *how new is this against memory (M)* — a thing that can be measured. *How much of that residual to keep* is a different question: it depends on what will later be asked (Q), which cannot be known when the content arrives. So that boundary is not measured, it is **chosen** — the **calibration**, a prior over Q and the budget B. The two contexts are never conflated: M is what the residual is measured against; Q is what decides whether a given compression was right.

The knob that matters most is **how aggressively to compress** — where, on the comparable residual scale, the line between *keep* and *drop* falls. That is the calibration's critical setting; the rest are operating points (how often to invoke the resolver) and bookkeeping. Because it is a bet, it is owned where the bet can be revised: at **consolidation**, which can re-derive from the immutable raw record — so a wrong compression is recoverable, not fatal — and which sees what retrieval actually used, the closest proxy for Q we get. Consolidation fits the calibration and pushes it outward, *down* to write and *up* to retrieve, where the decision is taken. That decision layer is fed by the predictor but is not part of it: swapping the embedding model changes what the sensor reports and forces only a re-fit of the calibration; the sensor's contract is unchanged.

What the predictor hands forward is therefore a **route** (predicted / novel / ambiguous), an **order** (which fragments surprised most), and a **relative weight** within a single batch — never a standalone "importance" number. An absolute magnitude would claim a precision the residual does not have and would drift as memory grows; a rank and a within-batch share say all that attention needs — *hold more of what surprised more* — without inventing that precision.

### What it can handle
Every function above is that one primitive with X and Y reassigned:

| Stage | X | Y | the residual tells us |
|---|---|---|---|
| Write | the incoming fragment | memory | what to store |
| Retrieve | a memory item | the query, then the growing assembly | what to fetch, and when to stop |
| Consolidate | a claim, or a member | the rest of memory, or its concept | what is safe to merge, forget, or split |

Most functions are this measurement read directly. A few need only a thin wrapper over it: a **sequential scan** (segmentation and query decomposition), an **iterative residual-against-the-assembly loop** (retrieval depth and stopping), and a **reconstruction guard** (safe merge and safe forget — fold or prune only when the residual against what remains is low). The same residual that decides "store only the new part" at write decides "fetch only what adds" at retrieve and "keep only what cannot be rebuilt" at consolidation.

Two honest limits on that reach. The residual is *geometric*, and the embedding can flatten precisely the nuances that matter most — a negation, a number, a scoped condition. So a low residual is **necessary but not sufficient** to merge or forget: the irreversible commit is confirmed by the resolver, never by geometry alone. And retrieval assumes a query embeds near the memory that answers it — which a symmetric encoder does not guarantee, since a question and its answer can differ more in form than in content. Until that is measured, the retrieval uses are the least proven part of this mechanism, and may need a query-aware embedding to hold.

Two things it deliberately does not do. **Direction** is the resolver's, as above — contradiction versus refinement, distillation, reconciliation, synthesis. And **usage frequency** — what was retrieved, what was ignored — is a *logged* signal, not a measured one; it feeds consolidation alongside the predictor, never from it, because rare-but-correct memory must not be suppressed for being quiet.

The *what* was settled first so that committing the *how* would be one decision unlocking many — and it is: a single estimator, reused at every stage, with one clean line it does not cross.
