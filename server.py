"""Thin FastAPI app mounting /mcp + /health + a minimal /status page.

/status is the only UI and is deliberately just a client of the same data
the API serves (PLAN.md §8b) — one server-rendered page, Basic-Auth'd with
the same credentials as the MCP OAuth login.
"""
import html
import secrets

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core import config
from mcp_server import mcp

mcp_app = mcp.http_app(path="/")
app = FastAPI(title="slate-engine", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)

_basic = HTTPBasic(auto_error=False)


def _require_auth(credentials: HTTPBasicCredentials | None = Depends(_basic)):
    if not (config.AUTH_USER and config.AUTH_PASS):
        return  # local dev: no creds configured, page open
    ok = (credentials is not None
          and secrets.compare_digest(credentials.username, config.AUTH_USER)
          and secrets.compare_digest(credentials.password, config.AUTH_PASS))
    if not ok:
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Basic realm=slate"})


def _wellknown_candidates(rest: str) -> list[str]:
    """RFC 9728/8414 clients fetch OAuth discovery docs at the DOMAIN ROOT
    with the resource path as a suffix (/.well-known/oauth-protected-resource/mcp/),
    but FastMCP serves them inside the /mcp mount. Try the path as-is first
    (the mount serves the suffixed form), then with the /mcp suffix stripped
    (the authorization-server doc has no suffix inside the mount)."""
    candidates = [rest]
    stripped = rest.rstrip("/")
    if stripped.endswith("/mcp"):
        candidates.append(stripped[: -len("/mcp")])
    return candidates


_wellknown_client = None


@app.get("/.well-known/{rest:path}")
async def well_known_forward(rest: str):
    import httpx
    from fastapi.responses import Response
    global _wellknown_client
    if _wellknown_client is None:
        _wellknown_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=mcp_app),
            base_url="http://wellknown.internal", follow_redirects=True)
    result = None
    for cand in _wellknown_candidates(rest):
        result = await _wellknown_client.get(f"/.well-known/{cand}")
        if result.status_code == 200:
            break
    return Response(content=result.content, status_code=result.status_code,
                    media_type=result.headers.get("content-type"))


@app.get("/health")
def health() -> dict:
    from core import store
    counts = store.stats(store.connect())
    return {"status": "ok", "episodes": counts["episodes"],
            "concepts": counts["concepts"]}


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>slate-engine</title>
<meta http-equiv="refresh" content="120">
<style>
body{{font-family:-apple-system,sans-serif;max-width:760px;margin:40px auto;
     padding:0 20px;background:#111;color:#ddd}}
h1{{font-size:20px}} h2{{font-size:15px;color:#8ab;margin-top:28px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap}}
.card{{background:#1c1c1e;border-radius:10px;padding:14px 20px;min-width:90px}}
.card b{{display:block;font-size:22px;color:#fff}}
.card span{{font-size:12px;color:#999}}
.run{{font-size:13px;color:#bbb}} .ok{{color:#7c5}} .failed{{color:#e66}}
pre{{background:#1c1c1e;border-radius:10px;padding:14px;white-space:pre-wrap;
    font-size:13px;line-height:1.5}}
footer{{margin-top:30px;font-size:11px;color:#666}}
</style></head><body>
<h1>🧠 slate-engine</h1>
<div class="cards">{cards}</div>
<h2>Last consolidation</h2>
<p class="run">{run}</p>
<h2>Recent digest (48h)</h2>
<pre>{digest}</pre>
<footer>auto-refreshes every 2 min · MCP endpoint: {base}/mcp</footer>
</body></html>"""


@app.get("/status", response_class=HTMLResponse, dependencies=[Depends(_require_auth)])
def status_page() -> str:
    from core import store
    from core.digest import digest as render_digest
    conn = store.connect()
    s = store.stats(conn)
    bridges = conn.execute(
        "SELECT COUNT(*) AS n FROM relations WHERE relation='bridges'").fetchone()["n"]

    cards = "".join(
        f'<div class="card"><b>{s[k]}</b><span>{k.replace("_", " ")}</span></div>'
        for k in ("episodes", "claims", "concepts", "relations"))
    cards += f'<div class="card"><b>{bridges}</b><span>bridges</span></div>'

    run = s.get("last_consolidation")
    if run:
        css = "ok" if run["status"] == "ok" else "failed"
        run_html = (f'<span class="{css}">{html.escape(str(run["status"]))}</span> · '
                    f'{run["n_episodes"]} episodes · '
                    f'${run["cost_estimate"] or 0:.2f} · '
                    f'finished {html.escape(str(run["finished_at"] or "—"))[:16]}')
    else:
        run_html = "no runs yet"

    return _PAGE.format(cards=cards, run=run_html,
                        digest=html.escape(render_digest(conn, since_hours=48)),
                        base=html.escape(config.SLATE_BASE_URL))
