# Broad-lift via user-traversal signal — implementation plan

**Status:** planning (implement in a fresh chat). **Primary goal:** move broad Coverage@B off its ~33–36 wall using a *demand-side* signal we didn't have before — the user's own drill-down/traversal — without regressing narrow/paragraph (forgetting = 0).

## 0. Why this exists (one paragraph)

Broad = cross-note synthesis (tension, lineage, cross-scale arcs). The store has three ways to connect concepts (`cpt↔cpt`):
- **Lane 1 — explicit relation edge** (`relations` table): within-episode only, weight 1.0, no usage reinforcement, and the ranking layer *demotes* high-degree endpoints (`distinctiveness = 1/(1+ln degree)^gamma`, `resonance.py:309`). **Historically inert for broad.**
- **Lane 2 — shared-member 2-hop** (`concept_members.kind ∈ {redundant, query}`): a claim/qclm attached to multiple concepts makes them mutually reachable. **This is what has moved broad** — channel-code redundancy (6c) + query injection (6d, with `REANCHOR`).
- **Lane 3 — embedding proximity**: ephemeral, recomputed each retrieval, never persisted. How cross-time adjacency actually surfaces today, then discarded.

We now capture the user's traversal, which is a **concept walk** (`cpt→cpt` bridges with `clm`/`ep` leaves; never `clm→clm`). We route that signal into all three lanes + salience + calibration and measure which lifts broad.

**Success criteria (per workstream):** ΔBroad Coverage@B > baseline AND Δnarrow ≥ 0 AND Δparagraph ≥ 0 (forgetting gate). Each lane ships only if it clears this. Deterministic `eval/coverage.py` is the primary metric; `eval/harness.py:judge()` (Opus-4.8) confirms; gemini judge is known-floored, ignore.

---

## Phase 0 — Signal capture (PREREQUISITE for everything)

### 0.1 New event `ENGAGEMENT`
Append-only, log-only (NOT materialized into the semantic store — `apply_event` ignores it, consolidation reads it straight from the log). Mirror `RETRIEVAL_SIGNAL` / `RELEVANCE_FEEDBACK`.

Payload:
```json
{
  "query": "the seeding query text",
  "surfaced": ["cpt_a", "clm_x", ...],   // what recall offered
  "engaged": "cpt_a",                      // the node the user drilled first
  "path": ["cpt_a", "cpt_b", "cpt_c"],    // walk order (concepts; claims/eps dropped)
  "spawned_write": false,                  // did a new note follow in-session
  "ts": "..."
}
```

- **New recorder** `retrieve.record_engagement(conn, user_id, query, *, surfaced, engaged, path, spawned_write, run_id=None)` → `store.append_event(..., "ENGAGEMENT", ...)`. Put next to `record_retrieval_signal` (`retrieve.py:146`), add to `__all__`.
- Path is normalised to **concepts** (lift claim→home concept via `_claim_concept`) — matches the traversal analysis (concept is the only pivot).
- Host/UX emits it. **No UX exists in eval**, so see 0.2.

### 0.2 Eval traversal generator (so we can test with no live UX)
`eval/engagement_gen.py`:
- For each `gold_broad` probe, build the "ideal walk" = the concepts the gold `key_facts` span (map each gold fact → nearest concept). Emit a synthetic `ENGAGEMENT` per probe.
- Also ingest the `slate_use_case.txt` transcript as ONE real trace: path = `Compromise&SD → Rel-Identity-Erosion → Society-vs-Individuality → Personal-Identity-Preservation`, engaged = `Compromise&SD`.
- Flag `--synthetic` in the harness so gold stays clean in prod (mirrors 6d's "synthetic queries are eval-only" discipline).

**Deliverable of Phase 0:** a corpus + a set of `ENGAGEMENT` events on it, replayable, so consolidation workstreams below have input.

---

## Workstream A — Lane 2 (PRIMARY BET): traversal → query-claim + reanchor

Extend step 6d (`consolidate.py:1262`, `_inject`/`QUERY_INJECTED`, applier `store.set_query_claims` `:982`).

- **New source:** in addition to `RETRIEVAL_SIGNAL` queries, read `ENGAGEMENT` events.
- **New attach set:** attach the `qclm_` to the concepts on the **walked path** (`payload.path`), not `activate()`'s geometric top-K. This is the demand-validated concept set — the whole point.
- **Query text for a multi-hop walk:** reuse the seeding `query`. (Optional v2: LLM-synthesize a one-line "arc query" spanning the path; keep behind a flag, test separately.)
- **`REANCHOR` ON** — the medoid pull toward the query is the broad lever (`:1268`); a bare bridge without it under-delivers. Ablate on/off to confirm.
- Applier is already replace-whole / idempotent / rebuild-safe — just a new input source. No new machinery.

**Test:** baseline vs +A(reanchor off) vs +A(reanchor on). Expect the biggest broad lift here.
**Kill criteria:** if +A(reanchor on) doesn't beat baseline broad, or regresses narrow/paragraph, do not ship.

---

## Workstream B — Lane 1 (LOW-PRIOR TEST): cross-episode Hebbian edges

New consolidation step (call it 6f / `TRAVERSAL_EDGES`).

- Read `ENGAGEMENT`; extract consecutive `cpt→cpt` pairs from each `path`; count frequency across ALL engagement events.
- Emit `RELATED{from_c, to_c, relation:'associates', weight=f(count)}`. **Compute weight from all events and SET it** (replace-whole, idempotent) — do NOT rely on `insert_relation`'s overwrite-on-conflict for accumulation. Add a `set_traversal_edges` applier that wipes prior `relation='associates'` rows then re-inserts (mirror `set_channel_redundancy` `store.py:960`).
- Remove the within-episode assumption: none in storage (`insert_relation` takes any ids); the constraint is only that `_relations` sources from one episode's spine. This step sources from cross-episode traversal — that's the removal.
- **Sub-variant to test the known failure mode:** the distinctiveness penalty demotes bridged endpoints. Add an optional walk flag `res_walked_edge_exempt` — a walked `associates` edge does NOT count toward the target's degree penalty (`resonance.py:309`). Test with/without.

**Test:** baseline vs +B vs +B(exempt). Prior: skeptical. This is the experiment the traversal data newly *enables*, not a broad fix we're betting on.
**Kill criteria:** inert or regressive → confirm Lane 1 dead even with real traversal, document, move on.

---

## Workstream C — Idea 3 (MINE): arc / schema node synthesis (persist Lane 3)

Turn the *ephemeral* cross-time synthesis into a durable node — the "joint" that coverage needs.

New consolidation step (`ARC_SYNTHESIZED`):
- For each walked `path` of length ≥ 3, LLM synthesizes a short concept summary ("the compromise→erosion→identity arc, 2017→2026").
- **Mint a new concept** whose members are the claims along the path (attach `kind='primary'` OR a new `kind='arc'` — decide based on whether it should move member medoids). Deterministic id from the sorted path-node set → idempotent, rebuild-safe. Applier wipes prior arc nodes + re-inserts (replace-whole).
- Concept embedding = LLM-summary embedding (captures the abstraction the members' medoid can't).
- A future broad query near the arc theme seeds the arc concept → reaches all path members in **1 hop** (vs 2-hop shared-member, vs unreachable today).

**Guardrails:**
- Must not create a new mega-hub: cap arc member count; check distinctiveness after mint.
- Gate behind forgetting check like everything else.

**Test:** baseline vs +C. This is the "add structure" option — highest upside, watch narrow/paragraph closely.
**Kill criteria:** creates hubs / regresses forgetting / no broad lift.

---

## Workstream D — Salience flag (engagement → C13/C13b)

Cheapest; improves felt quality; NOT expected to move broad (ranking layer, FLOOR-blocked). Ship because it's near-free.

- On engagement, emit a relevance vote: `record_relevance_feedback(relevant=[engaged_id])`. C13b (`consolidate.py:1453`) already UNBACKGROUNDS it. Surfaced-but-not-engaged nodes: **do not** auto-demote (respect existing "rare-but-correct must not be suppressed" caution, `:1378`).
- **Unlock C13's deferred promote branch** (`_consume_retrieval_signals` `:1370`): the code comment says promotion was deferred for lack of a gold "needed" signal — engagement IS that signal. Add a promote branch gated on engagement presence (mirror the existing background/BACKGROUNDED guard for idempotency).

**Test:** narrow/paragraph must stay ≥ baseline (this is a forgetting check, not a broad bet). Optional: does the engaged node rank higher (felt-quality proxy)?

---

## Workstream E — Calibration fit (engagement as growing gold)

Deepest "tune, don't add" — feed the existing fit loop a real, self-growing gold set.

- Convert `ENGAGEMENT` → `(query → wanted-node)` pairs = a synthetic gold file in the `gold.jsonl` shape.
- Run `eval/fit_stop.py:sweep()` + `pick_best()` against it (offline/nightly) → fit `value_floor` (+ friends) → `calibration.push(conn, user_id, value_floor=...)` (`calibration.py:30`). The docstring already anticipates the "relevance-aware `value_floor`" (`calibration.py:11`).
- Compare the demand-fitted floor vs the hand-14-gold floor on Coverage@B.

**Test:** does the demand-fitted `value_floor` beat the hand-gold floor on broad + narrow? **Kill criteria:** if the fitted floor regresses the frozen gold SR@B (fit_stop's own gate), keep the hand floor.

---

## Test plan (the core ask)

### New broad queries to add to `eval/gold_broad.jsonl`
Author ~6 new cross-note-synthesis probes (the transcript is the source material). Each needs `key_facts` spanning ≥2 notes/concepts so Coverage@B can score span:

1. **Tension:** "Where do my notes disagree about whether problems are the system's fault or my own responsibility?" (Compromise&SD vs reduced-personal-responsibility)
2. **Lineage:** "How has my view on compromise changed from 2017 to now?" (Social Contracts 2017 → today's system note)
3. **Cross-scale:** "What single pattern shows up in my notes on relationships, career, and society?" (self-erosion at increasing scale)
4. **Concept assembly:** "What have I concluded about identity across everything I've written?" (Blue Walls + Life:Survival + Society-vs-Individuality + Personal-Identity-Preservation)
5. **Meta-recurrence:** "Which of my ideas recur at more than one scale?"
6. **Bridge:** "How does my thinking on personal responsibility connect to my thinking on identity preservation?"

Keep the existing 11; these add the tension/lineage/arc shapes the current gold under-probes.

### Harness
- Primary: `eval/coverage.py:coverage_eval(gold, conn, user_id, context_fn, tau, budget_tok)` — deterministic, no LLM, cheap. `context_fn` = the retrieval assembly under each config.
- Confirm the winners with `eval/harness.py:judge()` under Opus-4.8 (real judge). Ignore gemini.
- Run each on: `gold_broad` (primary), `gold.jsonl` (narrow — forgetting guard), `gold_paragraph.jsonl` (paragraph — forgetting guard).

### Ablation matrix
| Config | broad | narrow | paragraph | ship? |
|---|---|---|---|---|
| baseline (current) | — | — | — | ref |
| +A Lane2 reanchor OFF | | | | |
| +A Lane2 reanchor ON | | | | |
| +B Lane1 edges | | | | |
| +B Lane1 edges + degree-exempt | | | | |
| +C arc node | | | | |
| +D salience | | | | |
| +E calibration fit | | | | |
| best-of combo | | | | |

Report ΔBroad and Δnarrow/Δparagraph for each. **Ship a lane iff ΔBroad > 0 and both forgetting deltas ≥ 0.** Log any coverage cap / dropped items (no silent truncation).

---

## Sequencing / dependencies
1. **Phase 0** (event + eval generator) — blocks all.
2. **Workstream A** (Lane 2) — the bet; do first after Phase 0.
3. **Workstream C** (arc node) — second-highest upside.
4. **Workstream B** (Lane 1) + **D** (salience) — cheap, run alongside as controls.
5. **Workstream E** (calibration) — after A/C land, so the fit is against an improved store.

## File touchpoints (quick map)
- Event/record: `core/retrieve.py` (`record_engagement`, ~`:146`), `core/store.py` (`append_event`; new appliers `set_traversal_edges`, arc mint), `core/consolidate.py` (`apply_event` `:149`; new steps near 6c/6d `:269`–`:298`; C13/C13b `:1370`/`:1453`).
- Retrieval scoring: `core/resonance.py` (`activate` `:208`, `_Graph.neighbours` `:173`, distinctiveness `:309` for the degree-exempt sub-variant).
- Calibration: `core/calibration.py` (`push`/`merged`), `eval/fit_stop.py` (`sweep`/`pick_best`).
- Eval: `eval/gold_broad.jsonl` (+new probes), `eval/coverage.py`, `eval/harness.py`, new `eval/engagement_gen.py`.
- Config flags: extend the `QUERY_CLAIM_*` / `res_*` family (attach-to-path, reanchor ablation, walked-edge-exempt, arc min-len/max-members). All new behaviour default OFF, flip via calibration profile.

## Risks / notes
- All new consolidation output must be **replace-whole, idempotent, rebuild-safe** (deterministic ids) — the store is a materialized view of the event log.
- Keep synthetic engagement **eval-only**; prod signal is genuine usage, gold stays clean.
- Every lane is still a bet that must clear the forgetting gate on real data. The new variable is the demand signal each lane was previously missing — not a guarantee.
- Expected result ranking (priors): A > C > (B, E) for broad; D is felt-quality only.
