# Multi-user auth — task scope

> Status: **implemented** (2026-06-12, all phases §9.1–9.6). 86 tests green,
> including the §7 isolation suite and an end-to-end two-user OAuth check
> against a live server. See [PLAN.md](PLAN.md) for the engine architecture
> this builds on. Implementation notes vs. this scope:
>
> - **Token binding detail:** FastMCP 3.4.2 drops `AccessToken.subject` between
>   issuance and tool context; the binding rides in `AccessToken.claims`
>   (`{"user_id": ...}`) instead — verified end-to-end incl. the refresh chain.
> - **Persisted token store:** DCR clients, access tokens (with their binding
>   claims), and the refresh chain live in `oauth_*` tables in the engine DB,
>   write-through on every issue/rotation/revocation and reloaded on boot — so
>   connected MCP clients survive deploys/crashes (verified with a SIGKILL +
>   restart e2e: same bearer works, refresh stays bound, rotated-out pairs stay
>   dead). Refresh tokens never expire; expired access tokens are purged on
>   boot without killing their refresh counterpart, so sessions outlive
>   downtime longer than the 1h access-token lifetime. Auth codes / pending
>   logins stay in-memory (5-min mid-browser-flow state). Tokens are stored
>   plaintext — bearer capabilities in the same trust domain as the corpus
>   they unlock.
> - **Composite PKs not needed:** ids stay globally unique instead — vec0
>   PRIMARY KEYs are unique *across* partitions, so per-user PKs were not an
>   option anyway. New claim ids are user-salted
>   (`clm_<md5(user_id + "\0" + text)>`, `consolidate.claim_id_for`); legacy
>   unsalted ids stay valid (single owner) and ride in immutable event payloads.
> - **Live cutover:** stop the container, run
>   `python migrate_multiuser.py --username $AUTH_USER` (keeps a
>   `.pre-multiuser.bak`, swaps atomically, verifies row parity first; `--dry-run`
>   to rehearse), restart. `store.connect()` hard-refuses a legacy DB, so the
>   server cannot silently run against an unmigrated file. The DEPLOY.md §7
>   cron lines work unchanged — `cli consolidate --all` / `digest` now default
>   to looping every user.
> - **Keep `AUTH_USER`/`AUTH_PASS` set in production.** The pair is the
>   OAuth-enabled switch (`mcp_server.py` only constructs the provider when
>   both are set) — unsetting it disables auth on `/mcp` entirely. After
>   bootstrap only its *credential* role is dead: logins validate solely
>   against the users table.
> - New dep: `bcrypt` (both requirements files).

## Decisions (locked)

- **Isolation:** shared DB with a `user_id` column on every table (not
  DB-per-user). DB-per-user was rejected on scale grounds — schema migrations,
  `VACUUM`, backups, and integrity checks would each loop over N files, and it
  is a one-way door away from any future cross-user query.
- **User store:** a `users` table with password hashes.
- **Provisioning:** admin-provisioned via CLI. No public self-signup.

The one feasibility risk in the shared-DB model — scoping vector search per
user — is solved: `sqlite-vec 0.1.9` supports `PARTITION KEY` on `vec0` tables,
so kNN can be restricted to one user's partition.

## 1. Data model (`core/store.py`)

Add `user_id` to **every** table and scope **every** query.

- New table: `users(id, username UNIQUE, pw_hash, created_at, is_admin)`.
- Add `user_id TEXT NOT NULL` to: `episodes`, `episode_sentences`, `claims`,
  `claim_support`, `concepts`, `concept_members`, `relations`, `events`,
  `consolidation_runs`, `episode_consolidations`, `replay_map`.
- Vec tables → redeclare with `user_id TEXT PARTITION KEY`
  (`vec_sentences`, `vec_claims`, `vec_concepts`); every kNN query adds
  `AND user_id = ?`.
- FTS (`episodes_fts`) → add `user_id UNINDEXED`; filter it in every match.
- **Composite PKs:** `claims.id` is `clm_<md5 of canonical text>` today — the
  same text from two users must not collide. PK becomes `(user_id, id)` (or the
  id is user-prefixed). Same fix for `claim_support`, `concept_members`,
  `relations`.
- Immutability triggers (`episodes_no_update`/`_no_delete`) stay unchanged.

**Threading model:** every `store.py` function gains a `user_id` parameter
(~40 functions) and filters on it. Explicit parameter — **not** ambient /
thread-local state — so the threaded server cannot leak by a forgotten context.
This is the bulk of the mechanical work and the main leak-risk surface.

## 2. Auth layer

- **Users table validation** replaces the single-credential check in both
  places: the MCP `SlateOAuthProvider` login (`mcp_server.py`) and the
  Basic-Auth `/status` (`server.py`). Hash with `argon2` or `bcrypt` (new dep).
- **Identity propagation (trickiest part):** FastMCP tools take no auth context
  today. The OAuth provider must bind the issued token → `user_id` at authorize
  time, and each tool resolves "who am I" from the request token via FastMCP's
  dependency/context API, then forwards `user_id` into `core`. **This is the
  spike to de-risk first** (see Phasing §9).
- The env `AUTH_USER`/`AUTH_PASS` path is kept only as a bootstrap admin (or
  dropped once a user row exists).

## 3. Code paths to scope

Thread `user_id` through: `encode.py`, `recall.py` (`recall`,
`assemble_context`, `get_episode`, `list_episodes`, `get_concept`),
`consolidate.py`, `digest.py`, `reconstruct.py`, plus `timeline` and `stats`.
Every MCP tool in `mcp_server.py` resolves `user_id` from auth and forwards it.

## 4. Consolidation & cron (per-user)

- `cli consolidate --all` and the nightly job loop **over all users** — each
  user's unconsolidated episodes produce their own run rows and LLM cost.
- `digest` becomes per-user. The `/run` button on `/status` runs only the
  requesting user's corpus (admins may run all).
- `consolidation_runs` / "last run" on `/status` filter by the logged-in user.

## 5. Migration of the existing corpus

- Designate today's single corpus as the first user (the current `AUTH_USER`).
- One-shot migration: create that user row, backfill `user_id` on all existing
  rows, rebuild vec/FTS into the partitioned tables.
- `migrate.py` replay and `replay_map` gain a target `user_id`.

## 6. Admin CLI

`cli user add <username>` (password prompt), `user list`, `user passwd`,
`user rm`.

## 7. Tests

- Existing tests assume a single corpus — update fixtures to seed a user.
- **Isolation tests are the priority:** user A's `recall` / `assemble_context`
  / `get_note` / `timeline` / `digest` must never surface user B's data — one
  leak test per tool, plus a vec-partition test and an FTS-scope test.

## 8. Risk register

| Risk | Severity | Mitigation |
|------|----------|------------|
| Cross-user data leak (any unfiltered query) | **High** | Per-tool isolation tests (§7) |
| vec `PARTITION KEY` kNN doesn't truly restrict in 0.1.9 | Medium | Verify in the phase-1 spike |
| Claim-id collision across users (`md5(text)` PK) | Medium | Composite PK; check every event applier |
| `_conn()` re-opens + re-inits schema per tool call | Low | Light connection cache (not blocking) |

## 9. Phasing

1. **Spike** — FastMCP token→`user_id` propagation + vec partition kNN.
   De-risks the two unknowns. *(~½ day, gate)*
2. **Schema + `store.py`** — `user_id`, composite PKs, partitioned vec/FTS,
   migration script. *(~1–1.5 days)*
3. **Thread `user_id`** through `core` + all MCP tools + `/status`/`/run`.
   *(~1–1.5 days)*
4. **Users table + auth** in OAuth and Basic-Auth; admin CLI. *(~1 day)*
5. **Per-user consolidation / digest / cron.** *(~½ day)*
6. **Isolation test suite** + migrate the live corpus. *(~1 day)*

Roughly **5–6 focused days**. Phase 1 is the gate: if FastMCP cannot cleanly
hand a tool the authenticated identity, that reshapes the rest.
