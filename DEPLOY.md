# Deploying slate-engine alongside the v1 Slate

Target: the existing v1 host (`ubuntu@140.245.216.42`, SSH key
`~/.ssh/slate_server`). v1 keeps running untouched on 127.0.0.1:8000 /
`myslate.duckdns.org`; v2 lands on 127.0.0.1:8100 behind its own subdomain.
Migration is replay-based and reads v1's `slate.db` read-only — v1 is never
written to and never goes down.

## 0. Pre-flight (on the server)

```bash
free -h          # v2 runs torch locally (~600-700MB RSS for the container).
                 # If total RAM < 2GB or free < 1GB, add swap first:
                 #   sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
                 #   sudo mkswap /swapfile && sudo swapon /swapfile
df -h /          # need ~4GB free: image ~2.5GB (torch) + model cache + db
docker ps        # confirm v1 'slate' container is the only thing on :8000
```

## 1. Ship the repo

Either push `slate_v2` to a private GitHub repo and clone it, or rsync:

```bash
rsync -av --exclude .venv --exclude data --exclude .git \
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

Set in `.env`: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `AUTH_USER`/`AUTH_PASS`
(reuse v1's), `SLATE_BASE_URL=https://myslate2.duckdns.org`,
`STANCE_PROVIDER=nli`. The compose file injects `DB_PATH` and `SLATE_V1_DB`.

## 3. Build + boot (the image's first real test)

```bash
docker compose up -d --build
docker compose logs -f slate-engine   # first boot downloads all-MiniLM into hf-cache
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
