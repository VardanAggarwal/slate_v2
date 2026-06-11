"""FastMCP + OAuth 2.1. Tools: save_note, recall, assemble_context, get_note, list_recent_notes, get_concept, timeline, digest. SlateOAuthProvider ported from v1 engine/mcp_server.py. See PLAN.md §5 MCP usage pattern + Phase 4.

Two-stage retrieval: `recall` returns ~50-token headlines so the model can
call it speculatively; `assemble_context` / `get_concept` are the escalation.
Trigger conditions live in the tool descriptions — models under-reach for
tools, so the descriptions say WHEN to call, not just what they do.
"""
import json
import secrets

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from core import config, store
from core.encode import encode

# ── OAuth provider (ported from v1) ───────────────────────────────────────────
slate_auth = None
if config.AUTH_USER and config.AUTH_PASS:
    from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
    from mcp.server.auth.provider import AuthorizationParams
    from mcp.server.auth.routes import ClientRegistrationOptions
    from mcp.shared.auth import OAuthClientInformationFull

    class SlateOAuthProvider(InMemoryOAuthProvider):
        """OAuth 2.1 with a login step: /authorize redirects to a login page;
        valid AUTH_USER/AUTH_PASS issues the auth code."""

        def __init__(self, base_url: str, auth_user: str, auth_pass: str):
            super().__init__(
                base_url=base_url,
                client_registration_options=ClientRegistrationOptions(enabled=True),
            )
            self.auth_user = auth_user
            self.auth_pass = auth_pass
            self._pending: dict[str, tuple[OAuthClientInformationFull, AuthorizationParams]] = {}

        async def authorize(self, client, params) -> str:
            auth_id = secrets.token_urlsafe(32)
            self._pending[auth_id] = (client, params)
            return f"{str(self.base_url).rstrip('/')}/login?auth_id={auth_id}"

        async def approve_authorization(self, auth_id: str) -> str:
            client, params = self._pending.pop(auth_id)
            return await super().authorize(client, params)

    slate_auth = SlateOAuthProvider(
        base_url=config.SLATE_BASE_URL.rstrip("/") + "/mcp",
        auth_user=config.AUTH_USER,
        auth_pass=config.AUTH_PASS,
    )


mcp = FastMCP(
    name="Slate",
    instructions=(
        "Slate is the user's personal memory engine — their own notes and ideas, "
        "distilled into claims and concepts with full provenance. "
        "BE PROACTIVE: whenever the user shares an opinion, idea, plan, or draft on "
        "a topic they may have thought about before, call `recall` BEFORE composing "
        "your response, and weave any prior thinking into it (cite note titles/dates). "
        "Use `assemble_context` when you need the full picture on a topic. "
        "SAVE DISCIPLINE: only call save_note with text the user themselves wrote or "
        "said. Never save AI-generated summaries, analysis, or paraphrases — the "
        "corpus must stay in the user's voice. When in doubt, ask before saving. "
        "After save_note, relay the receipt to the user — echoes and contradictions "
        "with their past notes are the product, not metadata."
    ),
    auth=slate_auth,
)


def _conn():
    return store.connect()


# ── Login route (only meaningful when OAuth is enabled) ───────────────────────
if slate_auth is not None:
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, RedirectResponse, Response

    _LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Slate — Authorize</title>
<style>body{{font-family:sans-serif;max-width:400px;margin:80px auto;padding:0 20px}}
input{{width:100%;padding:8px;margin-bottom:16px;box-sizing:border-box;font-size:16px}}
button{{width:100%;padding:10px;font-size:16px;cursor:pointer}}
.error{{color:#c00;margin-bottom:16px;font-size:14px}}</style></head>
<body><h2>Authorize Slate</h2>{error}
<form method="POST" action="?auth_id={auth_id}">
<label>Username</label><input name="username" autocomplete="username" required>
<label>Password</label><input name="password" type="password" autocomplete="current-password" required>
<button type="submit">Authorize</button></form></body></html>"""

    @mcp.custom_route("/login", methods=["GET", "POST"])
    async def login_handler(request: Request) -> Response:
        auth_id = request.query_params.get("auth_id", "")
        if not auth_id or auth_id not in slate_auth._pending:
            return Response("Invalid or expired authorization request.", status_code=400)
        if request.method == "GET":
            return HTMLResponse(_LOGIN_PAGE.format(auth_id=auth_id, error=""))
        form = await request.form()
        valid = (secrets.compare_digest(str(form.get("username", "")), slate_auth.auth_user)
                 and secrets.compare_digest(str(form.get("password", "")), slate_auth.auth_pass))
        if not valid:
            return HTMLResponse(
                _LOGIN_PAGE.format(auth_id=auth_id,
                                   error='<p class="error">Invalid username or password.</p>'),
                status_code=401)
        return RedirectResponse(await slate_auth.approve_authorization(auth_id),
                                status_code=302)


# ── Receipt → renderable markdown (engineered for Claude to narrate back) ─────
def receipt_markdown(receipt: dict) -> str:
    top = config.RECEIPT_TOP_N
    lines = []
    for e in receipt.get("contradictions", [])[:top]:
        lines.append(f"⚡ **Contradicts** a stored claim: “{e['claim_text']}” "
                     f"(sim {e['similarity']}) — your new line: “{e['sentence']}”")
    for e in receipt.get("echoes", [])[:top]:
        lines.append(f"🔁 **Echoes** stored claim: “{e['claim_text']}” (sim {e['similarity']})")
    for m in receipt.get("prior_episode_matches", [])[:top]:
        title = m.get("episode_title") or "untitled note"
        date = (m.get("episode_ts") or "")[:10]
        lines.append(f"🕰️ **Resonates with** your note “{title}” ({date}): "
                     f"“{m['matched_sentence']}”")
    n_nov = receipt.get("n_novelties", 0)
    if n_nov:
        lines.append(f"✨ {n_nov} new claim(s) — nothing like them stored yet.")
    if not lines:
        lines.append("Saved. No overlaps with stored thinking detected.")
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────────
@mcp.tool
def save_note(text: str, title: str) -> dict:
    """Save a note the user wrote to Slate and get back a novelty receipt.

    Call this when the user asks to save/remember/note something, or shares a
    finished piece of their own writing they want kept. ONLY the user's own
    words — never AI-generated text, even accurate summaries.

    The response's `narrate` field is markdown written to be relayed to the
    user: echoes of past claims, contradictions with stored thinking, and
    resonant older notes. Always surface it — it is the product.

    Args:
        text: the full note body, verbatim in the user's voice.
        title: concise 3-8 word title (generate it from the text).
    """
    conn = _conn()
    try:
        receipt = encode(conn, text, title=title.strip(), source="mcp")
    except ValueError as e:
        raise ToolError(str(e))
    return {"episode_id": receipt["episode_id"], "title": title.strip(),
            "n_sentences": receipt["n_sentences"],
            "narrate": receipt_markdown(receipt)}


@mcp.tool
def recall(query: str, k: int = 8) -> list[dict]:
    """Look up the user's prior thinking on a topic. CHEAP — call speculatively.

    Call this whenever the user shares an opinion, idea, plan, or draft on a
    topic they may have thought about before — BEFORE composing your response.
    Also for: "what do I think about X?", "have I written about X?".

    Returns compact headlines (~50 tokens each): claims and concepts ranked by
    spreading activation, with why-now signals (🌉 bridged concepts, 🔁
    recurring claims, 🕰️ long-dormant thinking resurfacing, 2-hop = non-obvious
    connection). Escalate with assemble_context or get_concept when a hit
    deserves the full picture.
    """
    from core.recall import recall as _recall
    return _recall(_conn(), query, k=k)


@mcp.tool
def assemble_context(topic: str) -> str:
    """Pull the user's full prior thinking on a topic as compact markdown,
    grouped by concept with claims + provenance (which note, which date).

    Call when you're about to write something substantive on a topic the user
    has history with, or after recall() surfaces a hit worth expanding. Output
    is budgeted (~1-2K tokens, most-relevant-first) for direct injection."""
    from core.recall import assemble_context as _ac
    return _ac(_conn(), topic)


@mcp.tool
def get_note(episode_id: str) -> dict:
    """Fetch one note in full: raw text, receipt, blueprint (post-consolidation),
    and the canonical claims it supports. Use after recall/list_recent_notes."""
    from core.recall import get_episode
    result = get_episode(_conn(), episode_id)
    if not result:
        raise ToolError(f"Note not found: {episode_id}")
    return result


@mcp.tool
def list_recent_notes(limit: int = 10) -> list[dict]:
    """List the most recently saved notes (id, title, ts, essence).
    Entry point for browsing; follow up with get_note."""
    from core.recall import list_episodes
    return list_episodes(_conn(), limit=min(limit, 50))


@mcp.tool
def get_concept(concept_id: str) -> dict:
    """Fetch one concept in full: label, canonical description, state/strength,
    member claims with provenance (episodes + verbatim sentences), and its
    relations including bridges. Use after recall surfaces a concept hit."""
    from core.recall import get_concept as _gc
    result = _gc(_conn(), concept_id)
    if not result:
        raise ToolError(f"Concept not found: {concept_id}")
    return result


@mcp.tool
def timeline(concept_id: str, limit: int = 50) -> list[dict]:
    """How the user's thinking on a concept evolved: every consolidation event
    that touched it (created, claims attached, merged, split, bridged, decayed),
    oldest first. Use for "how did my thinking on X change?"."""
    conn = _conn()
    rows = conn.execute(
        """SELECT seq, ts, type, payload_json FROM events
           WHERE payload_json LIKE ? ORDER BY seq LIMIT ?""",
        (f"%{concept_id}%", limit)).fetchall()
    out = []
    for r in rows:
        p = json.loads(r["payload_json"])
        out.append({"seq": r["seq"], "ts": r["ts"], "type": r["type"],
                    "summary": {k: p[k] for k in
                                ("label", "canonical", "claim_ids", "winner_id",
                                 "loser_id", "state_to", "rationale", "a", "b")
                                if k in p}})
    return out


@mcp.tool
def digest(since_hours: int = 36) -> str:
    """What emerged from recent consolidation: new bridges, contradictions,
    strengthened claims, concepts going dormant. Call when the user asks
    "what's new in my notes?" or each morning. Markdown, ready to relay."""
    from core.digest import digest as _digest
    return _digest(_conn(), since_hours=since_hours)


@mcp.tool
def stats() -> dict:
    """Corpus size (episodes, claims, concepts, relations) and the last
    consolidation run. Use for health checks / "how big is my Slate?"."""
    return store.stats(_conn())
