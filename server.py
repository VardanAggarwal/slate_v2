"""Thin FastAPI app mounting /mcp + /health + a minimal /status page.

/status is the only UI and is deliberately just a client of the same data
the API serves (PLAN.md §8b) — one server-rendered page, Basic-Auth'd against
the same users table as the MCP OAuth login (AUTH.md §2). The page shows the
logged-in user's corpus only; /run consolidates that user's episodes (admins
run every user, mirroring the nightly cron).
"""
import html
import threading
import time

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from core import config
from mcp_server import mcp

mcp_app = mcp.http_app(path="/")
app = FastAPI(title="slate-engine", lifespan=mcp_app.lifespan)
app.mount("/mcp", mcp_app)

# P0: shout at boot if the configured stance provider cannot actually run here.
# The failure mode is silent — classify_stance() degrades to "neutral" for every
# pair, so every contradiction files as an echo and the ⚡ line simply never fires.
# That ran live 2026-07-09 → 07-30 with nothing in the logs (DEPLOY.md §2).
from core.encode import check_stance_provider  # noqa: E402

_STANCE_HEALTH = check_stance_provider()

_basic = HTTPBasic(auto_error=False)


def _current_user(credentials: HTTPBasicCredentials | None = Depends(_basic)) -> dict:
    """Resolve the Basic-Auth user against the users table (AUTH.md §2).

    Auth is enforced once either env creds are configured or any user row
    exists — removing the env pair after provisioning must not open the page.
    """
    from core import store
    from core.auth import authenticate
    conn = store.connect()
    try:
        required = bool(config.AUTH_USER and config.AUTH_PASS) or store.count_users(conn) > 0
        if not required:  # local dev: nothing configured, page open
            return {"user_id": config.DEFAULT_USER_ID, "username": "local",
                    "is_admin": True}
        user = (authenticate(conn, credentials.username, credentials.password)
                if credentials is not None else None)
        if user is None:
            raise HTTPException(status_code=401, detail="Unauthorized",
                                headers={"WWW-Authenticate": "Basic realm=slate"})
        return {"user_id": user["id"], "username": user["username"],
                "is_admin": bool(user["is_admin"])}
    finally:
        conn.close()


# --- Manual nightly trigger -------------------------------------------------
# The same work cron runs at 02:30/07:45 (DEPLOY.md §7): consolidate --all then
# digest --polish, per user (AUTH.md §4). This lets the operator kick it off
# from /status on demand: admins run every user's corpus, others only their
# own. Runs in a background thread (a full consolidation can take minutes and
# cost money) with a lock so two clicks can't run it twice concurrently.
_job_lock = threading.Lock()
_job = {"running": False, "started_at": None, "finished_at": None,
        "message": "", "ok": None}


def _run_nightly(user_id: str, run_all: bool) -> None:
    from core import store, write
    from core.consolidate import consolidate
    from core.digest import digest
    conn = store.connect()
    try:
        # Write-side catch-up first (W2–W8): refine any episodes the async trigger
        # never reached or that HF/LLM failures left unfragmented, before they are
        # consolidated. A note's refine failure leaves it for the next sweep.
        write.refine_pending(conn, None if run_all else user_id)
        total = 0
        user_ids = store.users_with_unconsolidated(conn) if run_all else [user_id]
        for uid in user_ids:
            # Mirror `cli consolidate --all`, but also stop if a round makes no
            # progress (e.g. every remaining episode keeps being skipped) so a
            # stubborn note can't spin this thread forever.
            while True:
                report = consolidate(conn, uid, max_episodes=25)
                done = report.get("episodes", 0)
                total += done
                if report["status"] == "noop" or done == 0:
                    break
        digest(conn, user_id, since_hours=48, polish=True)
        scope = f"{len(user_ids)} user(s)" if run_all else "your corpus"
        msg = f"consolidated {total} episode(s) across {scope}, digest refreshed"
        ok = True
    except Exception as e:  # never leave the flag stuck on a crash
        msg = f"failed: {e}"
        ok = False
    finally:
        try:
            conn.close()
        except Exception:
            pass
    with _job_lock:
        _job.update(running=False, message=msg, ok=ok,
                    finished_at=time.strftime("%Y-%m-%d %H:%M"))


@app.post("/run")
def run_nightly(user: dict = Depends(_current_user)):
    with _job_lock:
        if not _job["running"]:
            _job.update(running=True, ok=None, message="",
                        started_at=time.strftime("%Y-%m-%d %H:%M"),
                        finished_at=None)
            threading.Thread(target=_run_nightly,
                             args=(user["user_id"], user["is_admin"]),
                             daemon=True).start()
    # PRG: redirect back so a refresh doesn't re-POST. Relative target keeps
    # the /engine/* alias working (the proxy strips the prefix).
    return RedirectResponse(url="status", status_code=303)


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
    conn = store.connect()
    return {"status": "ok",
            "episodes": conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"],
            "concepts": conn.execute("SELECT COUNT(*) AS n FROM concepts").fetchone()["n"],
            # P0: surfaced so a broken stance provider is visible from outside the
            # container, not only in the boot log.
            "stance": _STANCE_HEALTH,
            "evidence_lane": config.EVIDENCE_LANE}


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
button{{background:#2c5;color:#062;border:0;border-radius:8px;padding:9px 16px;
    font-size:13px;font-weight:600;cursor:pointer}}
button:disabled{{background:#333;color:#888;cursor:default}}
.job{{font-size:13px;color:#bbb;margin:10px 0}}
.who{{font-size:12px;color:#888}}
footer{{margin-top:30px;font-size:11px;color:#666}}
</style></head><body>
<h1>🧠 slate-engine</h1>
<p class="who">{who}</p>
<div class="cards">{cards}</div>
<h2>Nightly jobs</h2>
<form method="post" action="run">
  <button type="submit"{run_disabled}>Run nightly jobs now</button>
</form>
<p class="job">{job}</p>
<h2>Last consolidation</h2>
<p class="run">{run}</p>
<h2>Recent digest (48h)</h2>
<pre>{digest}</pre>
<footer>auto-refreshes every 2 min · MCP endpoint: {base}/mcp</footer>
</body></html>"""


@app.get("/status", response_class=HTMLResponse)
def status_page(user: dict = Depends(_current_user)) -> str:
    from core import store
    from core.digest import digest as render_digest
    uid = user["user_id"]
    conn = store.connect()
    s = store.stats(conn, uid)
    bridges = conn.execute(
        "SELECT COUNT(*) AS n FROM relations WHERE relation='bridges' AND user_id = ?",
        (uid,)).fetchone()["n"]

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

    with _job_lock:
        job = dict(_job)
    if job["running"]:
        job_html = (f'<span class="ok">⏳ running…</span> · '
                    f'started {html.escape(str(job["started_at"]))}')
        run_disabled = " disabled"
    elif job["finished_at"]:
        css = "ok" if job["ok"] else "failed"
        job_html = (f'last manual run: <span class="{css}">'
                    f'{html.escape(job["message"])}</span> · '
                    f'{html.escape(str(job["finished_at"]))}')
        run_disabled = ""
    else:
        job_html = "no manual run this session"
        run_disabled = ""

    who = f"signed in as {html.escape(user['username'])}"
    if user["is_admin"]:
        who += " · admin (Run executes every user's corpus)"
    return _PAGE.format(who=who, cards=cards, run=run_html, job=job_html,
                        run_disabled=run_disabled,
                        digest=html.escape(render_digest(conn, uid, since_hours=48)),
                        base=html.escape(config.SLATE_BASE_URL))
