# slate_v2 — Slate Engine

Headless, MCP-first rewrite of Slate's storage & retrieval engine, modeled on
hippocampus/neocortex memory: cheap synchronous **encode** at save time, nightly
batch **consolidation** into an event-sourced semantic store, local-only recall.

**Start here: [PLAN.md](PLAN.md)** — full architecture, schema, pipelines,
build phases with success criteria, and the invariants. §2 lists settled
decisions; do not relitigate them.

## Setup

```bash
# Requires a Python with sqlite loadable-extension support
# (Homebrew python@3.12 on macOS; python.org builds will NOT work)
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env            # add ANTHROPIC_API_KEY / GEMINI_API_KEY

.venv/bin/python -m pytest      # everything green before touching data
.venv/bin/python -m cli replay  # rebuild episodes from the old slate.db
.venv/bin/python -m cli consolidate --all --max-episodes 25
.venv/bin/python -m cli stats
```

Serve (Docker): `docker compose up -d` → MCP endpoint at
`$SLATE_BASE_URL/mcp`, health at `/health`. OAuth login is enabled when
`AUTH_USER`/`AUTH_PASS` are set; leave them empty for local dev.

## CLI

```
encode --text|--file [--title]   save a note, print the novelty receipt
replay [--old-db --limit]        idempotently replay the v1 corpus
consolidate [--all]              run the sleep phase now (sync mode)
digest [--since-hours --polish]  what emerged from recent consolidation
reconstruct <episode_id>         regenerate a note from its blueprint + fidelity
rebuild                          rebuild semantic store from the event log
stats                            counts + last consolidation run
```

## Nightly consolidation (Phase 5)

Plain cron, not an agent — deterministic pipeline; a failed run leaves its
episodes unconsolidated and the next night picks them up:

```cron
30 2 * * * cd /path/to/slate_v2 && .venv/bin/python -m cli consolidate --all >> data/cron.log 2>&1
45 7 * * * cd /path/to/slate_v2 && .venv/bin/python -m cli digest --polish >> data/digest.log 2>&1
```

## Using Slate from Claude (the headline use case)

Connect the MCP server in Claude.ai (or `claude mcp add`), then add this to
the project's instructions so Claude reaches for memory proactively:

> When we discuss ideas, opinions, plans, or drafts, check Slate first: call
> `recall` with the topic before composing your response, and weave my prior
> thinking in (cite note titles and dates). After saving a note with
> `save_note`, always relay the receipt — echoes and contradictions with my
> past notes are the point. Only save text I wrote myself, never your own
> summaries.

Two-stage retrieval keeps it cheap: `recall` returns ~50-token headlines
(with why-now signals: 🌉 bridged, 🔁 recurring, 🕰️ resurfacing, 2-hop =
non-obvious connection); escalate to `assemble_context` / `get_concept` for
full claims with provenance.

## Layout (PLAN.md §3)

```
core/
  store.py        — SQLite + sqlite-vec; schema; event log (sole DB module)
  encode.py       — embed, novelty receipt, episode write
  consolidate.py  — nightly batch pipeline; event emit/apply; rebuild
  recall.py       — spreading-activation retrieval + read/browse API
  reconstruct.py  — doc regeneration / synthesis from bridges
  digest.py       — morning digest from last night's events
  llm.py          — provider chain + Batch API helpers
  config.py       — env vars
mcp_server.py     — FastMCP + OAuth: 12 tools
server.py         — FastAPI app mounting /mcp + /health
cli.py            — encode | replay | consolidate | digest | reconstruct | rebuild | stats
migrate.py        — replay old slate.db sources as episodes
```

## Deploy-readiness audit (2026-06-11)

Verified locally: NLI stance label order confirmed against the real
cross-encoder (contradiction detection will work in production); Gemini
fallback exercised live through `llm.call`; Batch API helpers match the
installed anthropic SDK (0.109.1); a save during a held write lock waits and
succeeds (30s busy timeout), and consolidation no longer holds transactions
across LLM calls (regression-tested). **Not verified: the Docker image** —
Docker is not installed on this machine; `docker compose up` + `/health`
must be the first step of the production deploy (watch first-boot model +
punkt downloads in the slim image; consider baking them into the image or
mounting a cache volume).

## Status (2026-06-11) — deferred to production

All phases are implemented and tested locally (40 tests; partial corpus:
156 episodes replayed, first 25 consolidated → 63 concepts, 5 bridges,
rebuild verified byte-identical, fidelity 6-7/10 on 2013 notes, one real
synthesized doc). Deliberately deferred to the production deploy:

1. **Full migration**: `cli replay && cli consolidate --all --max-episodes 25`
   (~$0.03/episode sync; resume-safe — failed/killed runs reuse their logged
   blueprints). Then verify: claim dedupe fires on the duplicate notes
   (episodes ~140-156), `cli rebuild` checksum-identical, cost in
   `consolidation_runs`.
2. **Claude.ai connect** (Phase 4 acceptance): save a note, get a receipt
   with a real echo; set AUTH_USER/AUTH_PASS for OAuth.
3. **Two unattended nightly cron runs** (Phase 5 acceptance) + digest
   delivery channel choice (PLAN.md §9.2).

## Invariants (PLAN.md §10)

- Episodes are immutable — enforced by SQLite triggers.
- Only `consolidate.py` writes to the semantic store, only through `events`.
- `rebuild` from the event log must reproduce the semantic store exactly
  (covered by a test).
- Embeddings and search are local; LLM calls only at consolidation,
  on-demand generation, and (optionally) one small stance call at encode.
