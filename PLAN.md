# Slate Engine — Implementation Plan

> Build plan for a new, headless, MCP-first rewrite of Slate's storage & retrieval engine.
> Self-contained: written to be executed in a fresh session with no prior context.
> Source repo for reference/porting: `/Users/vardanaggarwal/slate` (keep it running untouched).

---

## 1. Why this exists (context)

Slate's goal is a brain-like memory system: **hippocampus** (awake-time encoding: identify what's known / new / worth remembering), **sleep consolidation** (push to neocortex: dedupe, integrate, find connections), **minimal unique storage** (recreate documents from the fewest stored pieces), and **dynamic correlation** (surface emerging connections between unrelated topics, including temporal "A leads to B" structure).

The current implementation gets the *representation* right (fragments, concepts, bridges, spine links, health decay) but the *process* wrong:

- All semantic decisions happen **synchronously at save time** in one greedy LLM call with only top-5 neighbors visible. No sleep phase, no replay.
- Concepts can merge but never **split**; merges destructively `DELETE` rows — no history, no undo, no "how did my thinking evolve".
- Claims are **never deduplicated** — no compression pressure, no canonical-claim layer.
- Spine ("A leads to B") links are extracted then **never read** by search or ranking.
- Bridges only form when one note touches both concepts; **latent** drift between concepts is invisible.
- ChromaDB + SQLite dual-store causes consistency pain (manual rollback in `ingest.py:129-134`, dual-write of `concept_id`).

## 2. Core architecture decisions (settled — do not relitigate)

1. **New project, not a refactor.** New repo `slate-engine` (sibling dir to `slate/`).
2. **Migration = replay, not data port.** Old `slate.db` `sources.raw_text` holds every note in full. Feed each through the new `encode()` as an episode; first full `consolidate()` rebuilds the semantic store. Repeatable, no cutover risk.
3. **Single store: one SQLite file with `sqlite-vec`** for vectors + graph + event log + FTS. No ChromaDB.
4. **Two memory stores, one transfer process:**
   - **Episodic store** (append-only): raw content + embeddings + novelty markers. Written at save time. Immutable. Source of truth.
   - **Semantic store** (derived, rebuildable): canonical claims, concepts, typed relations. Written ONLY by consolidation, ONLY via the event log.
5. **`encode()` is cheap and synchronous** — embed locally, nearest-neighbor against canonical claims, return a novelty receipt (echoes / novelties / contradictions). **No LLM graph decisions at save time.** At most one small LLM (or local NLI) call for stance classification.
6. **`consolidate()` is the nightly batch** — blueprint extraction, claim canonicalization (dedupe with provenance), concept merge/split/create with global context, latent bridge detection, decay/strengthen. All decisions emitted as **events**, never destructive updates. All LLM calls via the **Batch API** (50% off).
7. **Headless, MCP-first.** FastMCP server (port OAuth 2.1 scaffolding from old repo) is the only interface. Save from Claude.ai / Claude Code / mobile. A web UI, if ever, is just another client.
8. **Nightly job = plain code on cron**, not an agent. Deterministic pipeline: collect episodes → submit batch → poll → apply events.
9. **Morning digest is a required feature**, not a nice-to-have (Phase 6). It reads last night's events and tells the user what emerged.
10. **Embeddings stay local and free** (all-MiniLM-L6-v2 on CPU, same as today). Never move embeddings to an API.
11. **Model tiering:** Haiku-class for mechanical work (extraction, canonicalization), Sonnet-class for the global merge/split judgment pass. Reuse the existing provider fallback chain (claude → gemini → local).

## 3. Target project layout

```
slate-engine/
  core/
    store.py        — SQLite + sqlite-vec; schema init; event-log append/apply;
                      the ONLY module that touches the DB file
    encode.py       — embed, segment, novelty receipt, episode write
    consolidate.py  — sleep phase pipeline (see §5)
    recall.py       — spreading-activation retrieval
    reconstruct.py  — regenerate a doc from its blueprint; synthesize() from bridges
    digest.py       — morning digest: summarize last night's events
    llm.py          — provider chain + Batch API helpers
    config.py       — env vars (port pattern from old engine/config.py)
  mcp_server.py     — FastMCP + OAuth: save_note, recall, timeline, reconstruct,
                      synthesize, digest
  server.py         — thin FastAPI app mounting /mcp + /health
  cli.py            — `encode`, `consolidate`, `replay`, `digest`, `rebuild`
  migrate.py        — old slate.db → episodes (replay driver)
  tests/            — pytest; the old project has no tests — this one does
  Dockerfile, docker-compose.yml, .env.example
```

## 4. Data model (single SQLite file `data/engine.db`)

```sql
-- EPISODIC STORE (append-only, immutable)
episodes(
  id TEXT PRIMARY KEY,            -- ep_<ulid>
  ts TEXT NOT NULL,               -- ISO; for replayed notes use original created_at
  raw_text TEXT NOT NULL,
  title TEXT,
  source TEXT,                    -- 'mcp' | 'replay' | 'import' | ...
  receipt_json TEXT               -- novelty receipt computed at encode time
)
episode_sentences(episode_id, idx, text, embedding)   -- vec table via sqlite-vec

-- SEMANTIC STORE (derived; rebuildable from episodes + events)
claims(
  id TEXT PRIMARY KEY,            -- clm_<md5 of canonical text>
  text TEXT NOT NULL,             -- canonical distilled claim (deduped)
  embedding ...,                  -- sqlite-vec
  strength REAL DEFAULT 1.0,      -- bumped on re-encounter (spaced repetition)
  created_at TEXT, last_seen TEXT
)
claim_support(claim_id, episode_id, verbatim_sentence, PRIMARY KEY(claim_id, episode_id))
  -- provenance; this IS the minimal-unique-storage mechanism

concepts(
  id TEXT PRIMARY KEY, label TEXT, canonical TEXT, embedding ...,
  state TEXT,                     -- 'active'|'grounded'|'stale'|'dormant' (port health.py model)
  strength REAL, created_at TEXT, last_activity TEXT
)
concept_members(concept_id, claim_id, weight, PRIMARY KEY(concept_id, claim_id))

relations(
  from_id TEXT, to_id TEXT,       -- claim or concept ids
  relation TEXT,                  -- 'leads_to'|'contradicts'|'supports'|'bridges'|...
  weight REAL, created_at TEXT, evidence_episode_id TEXT
)
-- spine relations get promoted here at concept level → the temporal structure

-- EVENT LOG (backbone; semantic store is a materialized view of this)
events(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, run_id TEXT,
  type TEXT,                      -- ENCODED|CANONICALIZED|CONCEPT_CREATED|MERGED|
                                  -- SPLIT|BRIDGED|RELATED|DECAYED|STRENGTHENED
  payload_json TEXT
)
consolidation_runs(id, started_at, finished_at, n_episodes, batch_id, status, cost_estimate)
```

Rules:
- `MERGED` events record both concept snapshots — never delete the loser's history.
- `timeline(concept_id)` = filter events by concept id. `rebuild` = truncate semantic tables, replay all events (or re-consolidate all episodes with a better model).

## 5. Pipelines

### encode(text, ts?, title?) → receipt   [synchronous, <1s]
1. Append episode row; split sentences (port splitter from old `extract.py`); embed locally; store sentence vectors.
2. kNN each sentence against `claims` (vector) → classify matches:
   - **echo**: high sim, same stance → note claim_id (will bump strength at consolidation)
   - **novelty**: no claim above threshold
   - **contradiction**: high sim, opposing stance — via local NLI cross-encoder
     (e.g. `cross-encoder/nli-deberta-v3-small`, CPU) or one Haiku call as fallback
3. Write receipt to episode + `ENCODED` event. Return receipt — this is what the MCP tool shows the user immediately ("echoes your March note on X", "contradicts claim Y").

### consolidate(since?) → report   [nightly cron; Batch API]
1. Collect unconsolidated episodes.
2. **Blueprint extraction** per episode (port prompt + schema from old `extract.py` — best-tested asset). Haiku-class, batched.
3. **Canonicalize claims**: each blueprint claim vs kNN existing claims → LLM judges same/new → `CANONICALIZED` events; new rows or `claim_support` + strength bump. Haiku-class, batched.
4. **Concept pass** (the one judgment-heavy call, Sonnet-class): affected concepts + their members + new claims, full context → decisions: CREATE / MERGE / **SPLIT** / attach / NOOP → events.
5. **Relations**: promote blueprint spine links to claim/concept-level `relations` rows (`leads_to` etc.). Contradiction receipts → `contradicts` relations.
6. **Latent bridges**: pure embedding math — concept-pair similarity drift + 2-hop co-activation candidates; verify top candidates with one small LLM call each → `BRIDGED` events.
7. **Decay/strengthen**: port `health.py` state model as `DECAYED`/`STRENGTHENED` events; echoes from step 3 bump strength.
8. Write `consolidation_runs` row with token/cost accounting.

### recall(query, k) → results   [$0, local]
Vector seed over claims/concepts → spreading activation through `relations` and `concept_members` (decay per hop, weight by strength/recency/state) → ranked results with why-now signals (port: bridge 🌉, frequency 🔁, time-gap 🕰️). Two-hop activation is what surfaces non-obvious connections.

### Read/browse API   [$0, local SQL — lives in recall.py or a small read.py]
Direct lookups, no LLM, no embeddings. These back the MCP tools and replace v1's
panel browse routes (`GET /sources/{id}`, `GET /concepts/{id}`) and v1 MCP tools
(`get_note`, `list_recent_notes`, `assemble_context`):

- `get_episode(id)` — raw_text, title, ts, receipt, blueprint (post-consolidation), claims it supports
- `list_episodes(limit, before?)` — recent notes (id, title, ts, essence)
- `get_concept(id)` — label, canonical, state/strength, member claims **with provenance**
  (which episodes, verbatim sentences), relations incl. bridges, last_activity
- `get_claim(id)` — text, strength, supporting episodes + verbatim sentences
- `assemble_context(topic)` — the Claude-first read: recall(topic) → group hits by
  concept → return canonical + claims + provenance as compact markdown, sized for
  injection into a conversation. Port intent (not code) from v1 `mcp_server.py`.
- `stats()` — counts (episodes, claims, concepts, relations), last consolidation run

### MCP usage pattern (the headline use case)
In any Claude conversation with the connector active, Claude should *proactively*
pull relevant older concepts into context. This is engineered, not automatic:

- **Trigger conditions live in tool descriptions.** Models under-reach for tools;
  write descriptions that say WHEN to call, e.g. `recall`: "Call this whenever the
  user shares an opinion, idea, plan, or draft on a topic they may have thought
  about before — before composing your response." Same for the MCP server-level
  `instructions` string.
- **Two-stage retrieval.** `recall` returns compact headlines only (concept label,
  one-line canonical, why-now signal, similarity — ~50 tokens/hit) so it's cheap
  for Claude to call speculatively. `assemble_context` / `get_concept` are the
  escalation for full claims + provenance. Budget assemble_context output
  (~1-2K tokens max, most-relevant-first).
- **Receipts close the loop at save time.** `save_note`'s response (echoes /
  contradictions / novelties) is written FOR Claude to narrate back to the user
  ("this contradicts your March note on X") — phrase it as renderable markdown.
- Document a recommended claude.ai project instruction in the README, e.g.
  "When we discuss ideas, check Slate (recall) for my prior thinking first."

### reconstruct(episode_id) / synthesize(concept_ids|bridge_id)   [on demand]
- reconstruct: essence + spine + claims (+ verbatim sentences as style residue) → regenerate doc; report fidelity vs raw_text. North-star metric: unique-claim bytes ÷ reconstructable bytes.
- synthesize: pull two bridged concepts' claims + provenance → draft a NEW document about the connection. This is "create new docs from emerging learnings".

### digest(date?) → markdown   [REQUIRED — the daily payoff]
Read last night's events → one short LLM call → e.g.:
> 🌉 New bridge: "Conformity as Safety" × "Platform Lock-in" (via yesterday's note)
> ⚡ Contradiction: yesterday you argued X; on Mar 12 you claimed not-X
> 🔁 "Battery-saver travel" strengthened (3rd encounter)
> 🕰️ Going dormant: "Spaced repetition" — last touched 70 days ago

Exposed as MCP tool + CLI. Delivery options (pick during Phase 6): cron → ntfy/email/Telegram, or a Claude scheduled routine that calls the MCP tool and messages the user (agent fits here — judgment + prose).

## 6. What to port from `/Users/vardanaggarwal/slate` (don't rewrite)

| Asset | From | To |
|---|---|---|
| Extraction prompt + blueprint schema | `engine/extract.py` (PROMPT, sentence splitter, local KMeans fallback) | `consolidate.py` |
| LLM fallback chain (claude→gemini→local, per-model retry) | `engine/extract.py` / `engine/concepts.py` (recently unified) | `llm.py` |
| OAuth 2.1 + FastMCP scaffolding (`SlateOAuthProvider`) | `engine/mcp_server.py` | `mcp_server.py` |
| Health/decay state machine | `engine/health.py` (`_determine_state`) | consolidate step 7 |
| Bridge semantics + why-now ranking ideas | `engine/concepts.py`, `engine/search.py` | recall/bridges |
| Embedder singleton setup | `engine/db.py` | `store.py`/`encode.py` |
| Config pattern | `engine/config.py` | `config.py` |

Deliberately NOT ported: ChromaDB, save-time `update_concepts()`, Chroma/SQLite dual-writes, `ingest.py` rollback dance.

## 7. Build phases (each independently verifiable)

### Phase 0 — Scaffold
Repo, venv, deps (`fastapi`, `fastmcp`, `sqlite-vec`, `sentence-transformers`, `anthropic`, `google-genai`, NLI cross-encoder optional), config, Dockerfile. ✅ `pytest` green on a trivial store test.

### Phase 1 — Store + encode + replay
`store.py` schema/events; `encode.py` full path; `migrate.py` replaying old `slate.db` (preserve original `created_at` as episode ts; order chronologically so receipts are causally sane).
✅ episode count == old sources count; receipts non-empty on later notes; re-running replay is idempotent.

### Phase 2 — Consolidation
All of §5-consolidate. Build sync-mode first (direct calls) for fast iteration; add Batch API mode after the prompts settle. Run on replayed corpus; iterate on merge/split prompt against real data.
✅ claims deduped (count < raw claim instances); ≥1 sensible merge; SPLIT path exercised in a test; `cli.py rebuild` reproduces identical semantic store from the event log; cost-per-run logged.

### Phase 3 — Recall + read API
Spreading activation + ranking + why-now, plus the full read/browse API (§5: get_episode, list_episodes, get_concept, get_claim, assemble_context, stats).
✅ side-by-side eval vs old `search()` on ~10 real queries — new engine must surface at least one 2-hop result old search can't; `get_concept` on a real concept shows claims with correct episode provenance.

### Phase 4 — MCP server
Port OAuth; tools: `save_note` (→ encode, returns receipt), `recall`, `assemble_context`, `get_note`, `list_recent_notes`, `get_concept`, `timeline`, `digest` — v1 tool names (`get_note`, `list_recent_notes`, `assemble_context`, `search_corpus`→`recall`) kept compatible where sensible so existing Claude.ai connector habits carry over. Deploy via docker-compose; connect from Claude.ai.
✅ save a note from a Claude conversation and get a receipt with a real echo from the replayed corpus; `assemble_context` on a known topic returns concept-grouped context usable in-conversation.

### Phase 5 — Nightly cron
Host cron (or sidecar loop) → `python -m cli consolidate` (Batch mode). Failure handling: a failed run leaves episodes unconsolidated → safely picked up next night. Log to `consolidation_runs`.
✅ two consecutive unattended nightly runs complete; second night's digest reflects first night's saves.

### Phase 6 — Morning digest + reconstruct/synthesize
`digest.py` + delivery channel; `reconstruct.py` with fidelity score; `synthesize` MCP tool.
✅ digest arrives every morning; reconstruct on 5 old notes with fidelity self-rated ≥ 7/10; one synthesized doc from a real bridge.

(Then: retire or repoint the old Slate web UI; old repo becomes read-only archive.)

## 8. Cost guardrails (from analysis; 5 saves/day assumption)

- encode: ~$0–0.003/save (NLI local = $0; Haiku stance fallback ≈ $0.003)
- consolidate: ~$0.03–0.05/night batched (Haiku mechanical + 1 Sonnet merge/split call)
- full-corpus re-consolidation (~500 episodes): ~$3–5 batched — cheap enough to redo when models improve
- recall/timeline: $0 (local)
- Pricing refs: Haiku 4.5 $1/$5 per MTok; Sonnet 4.6 $3/$15; Batch −50%; cache reads ~0.1×, min cacheable prefix Haiku=4096 tok. Only LLM-call count should scale with daily writing volume, never with corpus size.

## 8b. Explicit non-goals (v2.0)

- **Discover subsystem** (RSS/Reddit/YouTube feeds, external-item scoring, interests, influence detection — v1 `engine/discover.py`) is OUT of scope for the initial build. The episodic model supports it later without schema change: consumed content becomes episodes with `source='feed'`, and consolidation treats them like any other episode. Don't build it now; don't design it out either.
- **Web UI.** Headless only; any future UI is a client of the MCP/HTTP API.
- v1's `respond.py` (URL fetch + corpus-grounded response generation) — revisit after Phase 6; `assemble_context` covers most of its value inside Claude conversations.

## 9. Open decisions (decide during build, don't block on them)

1. NLI cross-encoder locally vs Haiku for contradiction detection (start with NLI; fall back if quality poor).
2. Digest delivery channel (ntfy / email / Telegram / Claude scheduled routine).
3. Whether `encode()` also runs a fast local-KMeans blueprint for a richer instant receipt (extra polish, not required — full blueprint happens at night).
4. Repo name + whether old Slate UI gets repointed or retired.

## 10. Invariants (enforce in code review every phase)

- Episodes are immutable. Never UPDATE an episode.
- **Only `consolidate.py` writes to the semantic store, and only through `events`.** Any "quick" direct write from the MCP path re-creates the old architecture.
- Every consolidation decision must be an event before it is a row.
- `rebuild` from the event log must always reproduce the semantic store exactly.
- Embeddings local. Search local. LLM calls only at consolidation + on-demand generation + (optionally) one small call at encode.
