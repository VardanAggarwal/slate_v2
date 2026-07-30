# Deploying slate-engine alongside the v1 Slate

> **2026-06-12 — multi-user deployed:** AUTH.md implemented end-to-end on this
> host. The legacy corpus was migrated to first user `slate` (admin) via
> `migrate_multiuser.py`; backups on the server:
> `engine.pre-multiuser.bak` + `engine.db.pre-deploy` in
> `/home/ubuntu/slate-engine-data`. OAuth tokens now persist in the engine DB
> (verified in prod: same bearer survives `docker compose restart`). The
> deploy also restored `SLATE_BASE_URL=https://myslate.duckdns.org` to the
> server `.env` — it had been lost in the 2026-06-12 .env clobber, leaving
> OAuth discovery advertising localhost. Keep `AUTH_USER`/`AUTH_PASS` set:
> the pair enables OAuth on /mcp (credentials themselves now live in the
> users table). Cron unchanged (`consolidate --all` / `digest` are per-user
> by default).
>
> **Final state (2026-06-11, cutover complete):** v2 owns the root of
> `https://myslate.duckdns.org` (health `/health`, UI `/status`, MCP `/mcp`);
> `/engine/*` remains as an alias; `myslate2.duckdns.org` 301-redirects to
> `myslate`. v1 container is **stopped** (not removed) — restart anytime with
> `cd /home/ubuntu/slate && docker compose up -d`; its `slate.db` stays on
> disk and read-only-mounted in v2. LLM calls are subscription-first: the
> image ships Claude Code CLI, `CLAUDE_CODE_OAUTH_TOKEN` in the server `.env`
> (chain: claude-cli → claude API → gemini). The sections below describe the
> original side-by-side bring-up for reference.

Target: the existing v1 host (`ubuntu@140.245.216.42`, SSH key
`~/.ssh/slate_server`). v1 keeps running untouched on 127.0.0.1:8000 /
`myslate.duckdns.org`; v2 lands on 127.0.0.1:8100 behind its own subdomain.
Migration is replay-based and reads v1's `slate.db` read-only — v1 is never
written to and never goes down.

## 0. Pre-flight (on the server)

```bash
free -h          # v2 embeds via HF Inference API (HF_TOKEN) — no torch.
                 # Container RSS ~200MB; fits the 1GB box alongside v1.
df -h /          # ~1GB free is plenty (slim image, no model downloads)
docker ps        # confirm v1 'slate' container is the only thing on :8000
```

Surveyed 2026-06-11: 956MB RAM (~500MB free), 25GB disk free, Docker 29 +
Compose v5, Caddy active, v1 healthy on :8000 — all compatible.

## 1. Ship the repo (git-based since 2026-07-09)

`/home/ubuntu/slate-engine` is a git checkout of `VardanAggarwal/slate_v2`,
authenticated by a read-only deploy key (`~/.ssh/slate_engine_deploy` on the
server, wired via per-repo `core.sshCommand`; the key is registered under the
repo's Settings → Deploy keys). Runtime files (`.env`, backups, cron scripts)
are untracked and survive checkouts.

Deploy = push, pull, rebuild:

```bash
git push origin <branch>                          # from the laptop
ssh -i ~/.ssh/slate_server ubuntu@140.245.216.42 \
  'cd /home/ubuntu/slate-engine && git fetch origin && \
   git reset --hard origin/<branch> && docker compose up -d --build'
curl -s https://myslate.duckdns.org/health        # myslate serves v2 since the cutover
```

Prod tracks the branch that was last reset to (check with `git rev-parse
--abbrev-ref HEAD` on the server). Legacy rsync (pre-2026-07-09, kept for
emergencies — bypasses git, leaves the checkout dirty):

```bash
rsync -av --exclude .venv --exclude data --exclude .git --exclude .env \
  -e "ssh -i ~/.ssh/slate_server" \
  /Users/vardanaggarwal/slate_v2/ ubuntu@140.245.216.42:/home/ubuntu/slate-engine/
```

## 2. Server env

```bash
ssh -i ~/.ssh/slate_server ubuntu@140.245.216.42
mkdir -p /home/ubuntu/slate-engine-data/hf-cache
cd /home/ubuntu/slate-engine
cp .env.example .env && nano .env
```

Set in `.env`: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, **`HF_TOKEN`** (required
on the server — copy from `/home/ubuntu/slate/.env`; it switches embeddings to
the HF Inference API), `AUTH_USER`/`AUTH_PASS` (reuse v1's),
`SLATE_BASE_URL=https://myslate2.duckdns.org`, and **`STANCE_PROVIDER=hf`**
(the server image has no torch, so the local NLI cross-encoder isn't available;
`hf` runs zero-shot MNLI over the Inference API on the `HF_TOKEN` you already
set — no LLM call, no per-save cost). The compose file injects `DB_PATH` and
`SLATE_V1_DB`.

> **This one is load-bearing and fails silently.** If `STANCE_PROVIDER` is
> unset it defaults to `nli`, `_get_nli()` throws on the torch-less image,
> `classify_stance` degrades to `"neutral"`, and `_build_receipt` files every
> contradiction as an echo — the ⚡ line never fires and nothing errors. This
> was live in prod 2026-07-09 → 07-30. Verify after any deploy:
>
> ```bash
> docker exec slate-engine python -c "
> from core import config, encode
> print(config.STANCE_PROVIDER,
>       encode.classify_stance('I love the office.', 'I hate the office.'))"
> # want: hf contradiction
> ```

## 3. Build + boot (the image's first real test)

```bash
docker compose up -d --build          # slim image: requirements-server.txt, no torch
curl -s localhost:8100/health         # {"status":"ok","episodes":0,...}
```

## 4. Public URL

Register the subdomain (same IP, existing DuckDNS token from v1 access.md):

```bash
curl "https://www.duckdns.org/update?domains=myslate2&token=<TOKEN>&ip=140.245.216.42"
```

Reverse proxy — Caddy (`/etc/caddy/Caddyfile`):

```
myslate2.duckdns.org {
    reverse_proxy 127.0.0.1:8100
}
```

or nginx: copy the existing `myslate` server block, change `server_name` to
`myslate2.duckdns.org` and `proxy_pass` to `http://127.0.0.1:8100`, then
`certbot --nginx -d myslate2.duckdns.org`. Reload the proxy.

## 5. Migrate (replay, resume-safe, v1 read-only)

```bash
docker compose exec slate-engine python -m cli replay          # all sources → episodes
docker compose exec slate-engine python -m cli consolidate --all --max-episodes 25
docker compose exec slate-engine python -m cli stats
```

~$0.03/episode sync mode. Killed/failed runs resume free (logged blueprints
are reused). Verify after:
- `stats`: episodes == v1 non-empty sources; claims < CANONICALIZED event count
  (dedupe fires on the duplicate notes around episodes 140-156)
- `docker compose exec slate-engine python -m cli rebuild` → re-run `stats`,
  counts unchanged
- digest renders: `docker compose exec slate-engine python -m cli digest --since-hours 48`

## 6. Connect Claude.ai (Phase 4 acceptance)

Add connector `https://myslate2.duckdns.org/mcp` in Claude.ai → OAuth login
with AUTH_USER/AUTH_PASS → save a test note → expect a receipt echoing a
replayed note. Add the project instruction from README "Using Slate from
Claude".

## 7. Nightly cron (Phase 5)

```cron
30 2 * * * cd /home/ubuntu/slate-engine && docker compose exec -T slate-engine python -m cli consolidate --all >> /home/ubuntu/slate-engine-data/cron.log 2>&1
45 7 * * * cd /home/ubuntu/slate-engine && docker compose exec -T slate-engine python -m cli digest --polish >> /home/ubuntu/slate-engine-data/digest.log 2>&1
```

Check after two nights: `consolidation_runs` has two `ok` rows; morning
digest reflects the previous day's saves.

The evidence sweep needs no cron line of its own — `consolidate` runs it at the
end of each user's run, once the claim set for that run is final.

## 8. Evidence lane (docs/evidence-lane-plan.md)

**ON in production.** `docker-compose.yml` sets `EVIDENCE_LANE=${EVIDENCE_LANE:-1}`
in the `environment:` block, which overrides `env_file`. That placement is the point:
`.env.example` ships `EVIDENCE_LANE=0` (the safe default for local dev and for anyone
running the code as a library), so if the flag lived only in `.env`, re-copying the
example during a rebuild would silently ship the feature disabled. Compose can't be
re-copied — it's in git.

```bash
EVIDENCE_LANE=1        # concept membership kind, nightly stance sweep, 📎/⚡ recall
EVIDENCE_SHARE=0.20    # share of the assemble_context specifics budget
```

Verify it after every deploy — one call, no auth:

```bash
curl -s localhost:8100/health | python -m json.tool
# want: "evidence_lane": true   AND   "stance": {"ok": true, ...}
```

Both matter together. `evidence_lane: true` with a broken stance provider gives you
the lane with every verdict silently collapsed to `neutral` — sources would file as
"relates to" and never as 📎 or ⚡ (§2 is the same failure, and it ran live for three
weeks). The sweep is inert in that state.

Rollback needs no code change: set `EVIDENCE_LANE=0` in the host `.env` and restart —
`${EVIDENCE_LANE:-1}` reads it. No data fix-up, no episode is touched. To also drop
the derived `kind='evidence'` memberships, run `python -m cli rebuild`.

`STANCE_PROVIDER=haiku` + the sweep is a hard no — it bills per pair, and the
sweep refuses to start on it.

Gate evidence for turning it on is recorded in `eval/baseline_evidence_gate.json`
(verdict quality) and `eval/baseline_evidence_coverage.json` (Coverage@B). Re-run
either with `python -m eval.evidence_gate` / `python -m eval.coverage`.

## Coexistence + rollback

- v1 is untouched: its container, port, data dir, and domain stay as-is.
  The only coupling is the read-only `slate.db` mount.
- Roll back v2 anytime: `docker compose down` in `/home/ubuntu/slate-engine`
  — v1 unaffected.
- New notes saved to v2 during the transition do NOT flow back to v1; point
  Claude.ai at exactly one connector during cutover testing to avoid a
  split corpus. Once v2 is trusted, retire the v1 connector first, then the
  v1 container (PLAN.md: old repo becomes read-only archive).
- To re-migrate fresh after heavy v1 writes: stop v2, delete
  `/home/ubuntu/slate-engine-data/engine.db*`, re-run step 5.
