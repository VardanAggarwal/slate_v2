# slate_v2 — Slate Engine

Headless, MCP-first rewrite of Slate's storage & retrieval engine, modeled on
hippocampus/neocortex memory: cheap synchronous **encode** at save time, nightly
batch **consolidation** into an event-sourced semantic store, local-only recall.

**Start here: [PLAN.md](PLAN.md)** — full architecture, schema, pipelines,
build phases with success criteria, and the invariants. §2 lists settled
decisions; do not relitigate them.

Reference repo for porting (extraction prompt, LLM fallback chain, OAuth/MCP
scaffolding, health model): `/Users/vardanaggarwal/slate`.

## Layout (target — see PLAN.md §3)

```
core/
  store.py        — SQLite + sqlite-vec; schema; event log (sole DB module)
  encode.py       — embed, novelty receipt, episode write
  consolidate.py  — nightly batch pipeline
  recall.py       — spreading-activation retrieval
  reconstruct.py  — doc regeneration / synthesis
  digest.py       — morning digest from last night's events
  llm.py          — provider chain + Batch API helpers
  config.py       — env vars
mcp_server.py     — FastMCP + OAuth tools
server.py         — FastAPI app mounting /mcp + /health
cli.py            — encode | consolidate | replay | digest | rebuild
migrate.py        — replay old slate.db sources as episodes
```

## Invariants (PLAN.md §10)

- Episodes are immutable.
- Only `consolidate.py` writes to the semantic store, only through `events`.
- `rebuild` from the event log must reproduce the semantic store exactly.
- Embeddings and search are local; LLM calls only at consolidation,
  on-demand generation, and (optionally) one small call at encode.
