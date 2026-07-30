# Evidence Lane — Implementation Plan

> Add an external-evidence thread alongside the user's own notes: research/facts that
> back (or refute) claims. Designed 2026-07-30. Treat as a **critical change** — it
> touches the write path, the concept graph, and retrieval. Not an optimisation.

---

## 1. What this is

Evidence is an **episode with `source='research'`** that runs the *same* encode path as
a note. It mints claims like a note does. Origin stays recoverable via
`claim_support.episode_id → episodes.source`, so no `voice` column is needed anywhere.

Provenance: this is the first routing branch of the obligation graph in
`~/job-hunt/frontier-memory-blog.md` §6 — *"fillable from public fact → a background
research task."*

## 2. Invariants — must hold at every step

1. **Recall stays local and free.** No API call on the read path. Stance is precomputed,
   never computed at query time.
2. **Episodes are immutable.** Nothing here edits or deletes an episode. Rollback is
   always "stop emitting + `rebuild`", never a data fix-up.
3. **A source's words never become the user's.** Evidence may be a concept *member*;
   it may never be a concept *representative* (medoid / anchor / `concepts.canonical`).
4. **Self-lane Coverage@B does not regress.** Evidence competes for retrieval budget;
   the partition is what contains it. This is a measured gate, not an assumption.
5. **No new LLM billing.** Stance runs on `nli` (local) or `hf` (Inference API).
   `STANCE_PROVIDER=haiku` + the nightly sweep is a hard no — it bills per pair.

## 3. Settled — do not relitigate

Killed during design, with reasons:

| Rejected | Why |
|---|---|
| `claims.voice` column | origin is already joinable via `claim_support → episodes.source` |
| separate evidence store | evidence is an episode; the substrate already fits |
| declared `supports` edges / `backs_claim_ids` as FK | claim ids are md5 of text (`consolidate.py:126`) — any pinned edge dangles on canonicalisation |
| skipping consolidation for research episodes | minting buys free write-time evidence surfacing via `knn_claims`; the breaks were smaller than first claimed |
| forked strength (`n_external_support`) | sources only accumulate because you returned to the topic — research *is* investment |
| attachment-breadth scalar | decay already implements "useful if it resurfaces"; degree-boosting a retrieval target is the bridges failure mode |
| decay exemption for evidence | unreferenced evidence is useless; refresh on usage **or** new attachment |

---

## 4. P0 — Blocker, do before anything else

**Verify `STANCE_PROVIDER` in production.**

`.env.example:33` ships `STANCE_PROVIDER=nli`. The server image has no torch
(`Dockerfile:12`). If the server's `.env` doesn't override to `hf` with an `HF_TOKEN`,
then `_get_nli()`'s import fails → caught at `encode.py:121-122` → returns `"neutral"`
→ `_build_receipt` buckets it into `echoes` (`:187-190`).

Net: **the ⚡ contradiction line never fires in prod, silently.** Local is fine (torch is
in `.venv`), so this does not reproduce in tests.

- Check the server `.env` for `STANCE_PROVIDER` and `HF_TOKEN`.
- If unset → set `STANCE_PROVIDER=hf` + `HF_TOKEN`, redeploy, verify a known
  contradicting note produces a ⚡ line.
- Add a startup assertion: if `STANCE_PROVIDER=nli` and `sentence_transformers` is
  unimportable, log loudly rather than degrading to `neutral` for every pair.

**This is an existing production bug independent of the evidence work.** It also makes
E2/E5/E7 inert on arrival if unfixed, so it gates the whole plan.

---

## 5. Phasing

**Phase A — working save path (E1, E6, E2, E7).** No retrieval impact, no concept-graph
impact. Shippable and observable on its own.

**Gate A→B:** hand-feed 10–15 real sources. Read every receipt. Confirm the stance
verdicts are sane — specifically that genuine backing reads as 📎 and a deliberately
contradicting source reads as ⚡. Do not proceed until the labels are trustworthy;
everything in Phase B assumes the verdict is correct.

**Phase B — graph + retrieval (E3, E4, E5).** Touches concept membership and the
retrieval budget. Eval-gated.

**Gate B (exit):** Coverage@B re-baselined, self-lane flat, suite green.

Master flag `EVIDENCE_LANE` (default **off**) gates E3/E4/E5 behaviour. E1/E2/E6/E7 are
inert without a research episode existing, so they need no flag of their own.

---

## 6. Tasks

### E1 — `episodes.citation_json`

Nullable column: source URL, source title, retrieved-at.

- **Touch:** `core/store.py` — add to the `episodes` DDL (`:70`) **and** to `_ADD_COLUMNS`
  (`:361`), which is the house pattern for a column on an existing table. No separate
  migration script.
- **Must not:** prepend citation text to `raw_text`. It would pollute the sentence
  embeddings and mint a junk sentence.
- **Verify:** fresh DB and an existing DB both end up with the column; `PRAGMA table_info`.
- **Rollback:** additive and nullable — harmless if unused.
- **Tests:** `tests/test_store.py`.

### E2 — Stance-direction flip for `source='research'`

- **Touch:** `core/encode.py:183`. Branch on source: notes keep
  `classify_stance(claim_text, sent)`; evidence uses `classify_stance(sent, claim_text)`.
- **Why:** premise = the warrant, hypothesis = what is on trial. Evidence ⊨ claim, not the
  reverse — specific entails general, not vice versa. Unflipped, real backing scores
  `neutral` and falls into `echoes` (`:187-190`) indistinguishably from mere topical
  adjacency.
- **Keep `k=3`** (`:180`). One evidence sentence legitimately attaches to several claims
  with different verdicts.
- **Depends on:** P0. Inert if stance returns `neutral` for everything.
- **Verify:** a fixture pair (specific finding, general claim) returns `entailment`
  flipped and `neutral` unflipped. Assert both directions so the regression is caught.
- **Rollback:** single branch, revert.
- **Tests:** `tests/test_encode.py`.

### E6 — `save_evidence` MCP tool

- **Touch:** `mcp_server.py`, alongside `save_note` (`:314`).
- **Contract — the mirror of `save_note`'s:** only the *source's* words, verbatim. Both
  tools ban paraphrase, from opposite sides. A model-recollected "fact" is worse than no
  fact — it looks like verification and isn't.
- **Required args:** `text` (verbatim excerpt), `source_url`, `source_title`,
  `retrieved_at`. Reject empty citation — evidence without provenance is an assertion.
- **Calls** `encode(..., source="research")`.
- **Does not call** `trigger_refine_async` (fragmentation is working memory about *your*
  surprise — a wasted async LLM call here) and does not log `spawned_write`.
- **Depends on:** E1.
- **Verify:** episode lands with `source='research'` + citation; no `fragments` rows; no
  `ENGAGEMENT` event.
- **Tests:** `tests/test_mcp.py`.

### E7 — Evidence variant of `receipt_markdown`

Same receipt dict, inverted narration. `mcp_server.py:251-269`.

| field | note (current) | evidence |
|---|---|---|
| `contradictions` | ⚡ *Contradicts a stored claim … your new line* | ⚡ *A source refutes your claim* |
| `echoes` (entailment) | 🔁 *Echoes stored claim* | 📎 *Backs your claim* |
| `echoes` (neutral, ≥ τ) | — | *Relates to* — unverified |
| `prior_episode_matches` | 🕰️ *Resonates with your note* | 🕰️ your note **or** 📚 another source |
| `n_novelties` | ✨ *N new claims* | 🗄️ *Backs nothing you've written yet* — orphan, parks for the sweep |
| empty | *Saved. No overlaps…* | *Filed. Nothing in your corpus touches this yet.* |

- **`"your new line"` (`:256`) is a misattribution on the evidence path** — it quotes the
  incoming sentence as the user's. Fix, not restyle.
- **Not confined to the new variant.** `knn_sentences` returns matches from all episodes,
  so once research episodes exist a **note** receipt will render a paper as
  `Resonates with your note "…"`. The source-aware branch at `:259-263` must land on
  **both** paths. This requires `knn_sentences` to return `source` (`core/store.py`).
- **Why it matters:** the receipt is the only surface the user sees, and it's written for
  the model to narrate back. Correct rows narrated with the wrong verb read as a broken
  feature.
- **Depends on:** E2 (the entailment/neutral split).
- **Tests:** `tests/test_mcp.py`, `tests/test_encode.py`.

### E3 — `concept_members.kind='evidence'`

Evidence claims join concepts as members but are excluded from medoid/anchor candidacy.

**Verified: the exclusion is already free.** Every centre/baseline path *allowlists*
`kind`, it does not denylist:

- `store.py:1057` — concept vector: `AND kind IN ('primary', 'query')`
- `store.py:1086`, `:852`, `consolidate.py:917`, `:1229` — `kind = 'primary'`

So a new value is excluded from the concept vector, centres and baselines **by
construction**, exactly as `'redundant'` is (documented `store.py:365-368`). Unfiltered
paths — retrieval spread/materialize — still see it, which is what we want: evidence
stays reachable through the concept and can counter bias, without ever representing it.

- **Touch:** `core/consolidate.py` membership step — pass `kind="evidence"` when the
  claim's supporting episode is `source='research'`. `store.py:132` comment + `:369`.
- **Risk:** volume asymmetry. Evidence is cheap to add, notes aren't. Medoid/anchor
  exclusion protects concept *identity*; it does not stop the membership mix from
  shifting what a concept is made of. Log the evidence:self member ratio per concept
  each run and watch it.
- **Verify:** an evidence claim appears in `concept_members` with `kind='evidence'`, and
  the concept vector is byte-identical to its value before the evidence was added. If the
  vector moves, evidence is leaking into a centre path — stop and find it.
- **Rollback:** stop emitting `kind='evidence'` + `rebuild` (replays events, regenerates
  derived state without them). No episode is touched.
- **Tests:** `tests/test_consolidate.py`.

### E4 — `evidence_attachments` + nightly sweep

One row per (evidence sentence, claim) pair that cleared threshold, with its stance.

```
evidence_attachments(evidence_episode_id, sentence_idx, claim_id, stance, similarity)
```

- **Derived state** — truncated and recomputed like `claims` / `concept_members`. A stale
  claim id cannot strand anything because the table is rebuilt against current ids.
- **Durable record is the event:** `EVIDENCE_ATTACHED` carries the claim **text**, not
  just the id — the same reason `MERGED` stores both concept snapshots. That is what
  survives re-canonicalisation and lets `rebuild` replay.
- **Sweep:** k-NN (local, free) → `classify_stance` above `ECHO_THRESHOLD` (0.72,
  `config.py:25`).
- **Watermark — load-bearing, not an optimisation.** Nightly work is
  `(new evidence × current claims) + (all evidence × claims new-or-re-canonicalised since last run)`.
  Without it the sweep re-evaluates every pair every night, and since the table is
  truncated there is no memo to skip them.
- **Dormant evidence stays sweep-eligible.** Dormancy gates *recall*, not the sweep.
  This is the only thing letting a 2026 source back a 2028 claim.
- **Decay:** no exemption. Evidence decays on the normal clock; refresh on usage **or**
  a new attachment — the sweep bumps `last_seen` when it emits `EVIDENCE_ATTACHED`.
- **Cost:** zero LLM on `nli`/`hf`. `ECHO_THRESHOLD=0.72` against a MiniLM NN-cosine
  median ≈0.59 means most candidates never reach a stance call. **Assert
  `STANCE_PROVIDER != 'haiku'` before the sweep runs** — per-pair billing at sweep volume
  is the trap that drained credits twice before.
- **Spike risk:** a consolidation run that re-canonicalises many claims makes that night's
  sweep proportionally large. Cap it and `log()` what was dropped — no silent truncation.
- **Depends on:** E2 (stance direction), Gate A.
- **Verify:** run twice with no corpus change → second run does zero stance calls (count
  them). Add a claim → only the affected pairs are evaluated.
- **Tests:** new `tests/test_evidence.py`.

### E5 — Recall: three labels, budget-partitioned

- **Single pool, origin as a display attribute.** No filter, no promotion threshold.
- **Three labels:** 🧠 yours · 📎 evidence, backs · ⚡ evidence, contradicts. Collapsing
  ⚡ into 📎 is the one genuinely harmful outcome — a refutation rendered as support.
- **"Might" vs "does":** evidence surfacing *standalone* has no verdict for this pairing →
  *"might back you."* Evidence surfacing alongside a claim that is also in the result set
  has a precomputed verdict in `evidence_attachments` → *"backs this"* / *"refutes this."*
- **Budget-partition it** the way `depth_share` splits the budget in resonance retrieval.
  Evidence gets its own share of B rather than bidding freely — so self-lane coverage
  stays measurable against the existing gold sets and evidence can't crowd out the user's
  own thinking.
- **Touch:** `core/recall.py`, `core/retrieve.py`, MCP formatting.
- **Depends on:** E4.
- **Rollback:** `EVIDENCE_LANE=off` restores the current single-lane behaviour.
- **Tests:** `tests/test_recall.py`, `tests/test_retrieve.py`.

---

## 7. Eval protocol (Gate B exit)

Coverage@B against the existing gold sets: narrow (20), broad (6), paragraph (14), plus
the grep baseline.

1. **Baseline before Phase B**, on the current corpus with `EVIDENCE_LANE=off`.
2. **After E3** — concept membership changed. Expected: **flat**. Evidence is excluded
   from centres, so movement here means leakage. Treat any change as a bug, not a result.
3. **After E5** — evidence competes for budget. Acceptance: **self-lane coverage flat vs
   baseline**. That is what the partition is for.
4. **Broad = 6 scored queries.** A one-query move is one query. Report as a hint, never
   as a clean number.

Full suite green (currently ~230 tests) at every gate.

## 8. Out of scope

The obligation graph proper; an agent that goes and *fetches* sources; any UI. This builds
the lane, not the loop.
