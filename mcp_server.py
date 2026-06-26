"""FastMCP + OAuth 2.1. Tools: save_note, recall, assemble_context, mark_relevance, get_note, list_recent_notes, get_concept, timeline, digest. SlateOAuthProvider ported from v1 engine/mcp_server.py. See PLAN.md §5 MCP usage pattern + Phase 4, AUTH.md §2.

Two-stage retrieval: `recall` returns ~50-token headlines so the model can
call it speculatively; `assemble_context` / `get_concept` are the escalation.
Trigger conditions live in the tool descriptions — models under-reach for
tools, so the descriptions say WHEN to call, not just what they do.

Identity (AUTH.md §2): the login page validates against the users table and
binds the authenticated user_id to the issued auth code; the token exchange
stamps it into AccessToken.claims (FastMCP drops .subject in transit — claims
survive, verified against fastmcp 3.4.2). Every tool resolves user_id from
the request token via _user_id() and passes it into core explicitly.
"""
import json
import secrets

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from core import config, store
from core.encode import encode

# ── OAuth provider (ported from v1; user binding per AUTH.md §2) ──────────────
slate_auth = None
if config.AUTH_USER and config.AUTH_PASS:
    import time as _time

    from fastmcp.server.auth.providers.in_memory import InMemoryOAuthProvider
    from mcp.server.auth.provider import (AccessToken, AuthorizationParams,
                                          RefreshToken)
    from mcp.server.auth.routes import ClientRegistrationOptions
    from mcp.shared.auth import OAuthClientInformationFull

    class SlateOAuthProvider(InMemoryOAuthProvider):
        """OAuth 2.1 with a login step: /authorize redirects to a login page;
        a valid users-table login issues the auth code, bound to that user.

        DCR clients, access tokens (with their user-binding claims), and the
        refresh chain are persisted in the engine DB (oauth_* tables) and
        reloaded on boot, so connected MCP clients survive restarts — refresh
        tokens never expire, so even week-old sessions refresh straight back
        in. Auth codes and pending logins stay in-memory: 5-minute,
        mid-browser-flow state that a restart may legitimately drop.
        """

        def __init__(self, base_url: str):
            super().__init__(
                base_url=base_url,
                client_registration_options=ClientRegistrationOptions(enabled=True),
            )
            self._pending: dict[str, tuple[OAuthClientInformationFull, AuthorizationParams]] = {}
            self._code_user: dict[str, str] = {}     # auth code -> user_id
            self._refresh_user: dict[str, str] = {}  # refresh token -> user_id
            self._load_persisted()

        def _load_persisted(self) -> None:
            conn = store.connect()
            try:
                with conn:
                    store.purge_expired_oauth_tokens(conn, int(_time.time()))
                for r in store.load_oauth_clients(conn):
                    self.clients[r["client_id"]] = (
                        OAuthClientInformationFull.model_validate_json(r["client_json"]))
                for r in store.load_oauth_access_tokens(conn):
                    self.access_tokens[r["token"]] = (
                        AccessToken.model_validate_json(r["token_json"]))
                    if r["refresh_token"]:
                        self._access_to_refresh_map[r["token"]] = r["refresh_token"]
                        self._refresh_to_access_map[r["refresh_token"]] = r["token"]
                for r in store.load_oauth_refresh_tokens(conn):
                    self.refresh_tokens[r["token"]] = (
                        RefreshToken.model_validate_json(r["token_json"]))
                    if r["user_id"]:
                        self._refresh_user[r["token"]] = r["user_id"]
            finally:
                conn.close()

        async def register_client(self, client_info) -> None:
            await super().register_client(client_info)
            conn = store.connect()
            try:
                with conn:
                    store.save_oauth_client(conn, client_info.client_id,
                                            client_info.model_dump_json())
            finally:
                conn.close()

        async def authorize(self, client, params) -> str:
            auth_id = secrets.token_urlsafe(32)
            self._pending[auth_id] = (client, params)
            return f"{str(self.base_url).rstrip('/')}/login?auth_id={auth_id}"

        async def approve_authorization(self, auth_id: str, user_id: str) -> str:
            client, params = self._pending.pop(auth_id)
            redirect = await super().authorize(client, params)
            code = redirect.split("code=")[1].split("&")[0]
            self._code_user[code] = user_id
            return redirect

        def _bind(self, token, user_id: str | None):
            """Stamp user_id into the issued access token's claims and remember
            it for the refresh chain."""
            if not user_id:
                return token
            at = self.access_tokens[token.access_token]
            self.access_tokens[token.access_token] = at.model_copy(
                update={"subject": user_id, "claims": {"user_id": user_id}})
            if token.refresh_token:
                self._refresh_user[token.refresh_token] = user_id
            return token

        def _persist_tokens(self, token, user_id: str | None) -> None:
            """Write the freshly issued (post-_bind) pair through to the DB."""
            conn = store.connect()
            try:
                with conn:
                    at = self.access_tokens[token.access_token]
                    store.save_oauth_access_token(
                        conn, token.access_token, user_id, at.model_dump_json(),
                        token.refresh_token, at.expires_at)
                    if token.refresh_token:
                        rt = self.refresh_tokens[token.refresh_token]
                        store.save_oauth_refresh_token(
                            conn, token.refresh_token, user_id,
                            rt.model_dump_json(), rt.expires_at)
            finally:
                conn.close()

        def _revoke_internal(self, access_token_str=None, refresh_token_str=None):
            # capture the paired refresh before super() pops the maps
            paired = self._access_to_refresh_map.get(access_token_str)
            super()._revoke_internal(access_token_str=access_token_str,
                                     refresh_token_str=refresh_token_str)
            for k in (refresh_token_str, paired):
                if k:
                    self._refresh_user.pop(k, None)
            conn = store.connect()
            try:
                with conn:
                    store.delete_oauth_tokens(conn, access_token=access_token_str,
                                              refresh_token=refresh_token_str)
            finally:
                conn.close()

        async def exchange_authorization_code(self, client, authorization_code):
            user_id = self._code_user.pop(authorization_code.code, None)
            token = await super().exchange_authorization_code(client, authorization_code)
            self._bind(token, user_id)
            self._persist_tokens(token, user_id)
            return token

        async def exchange_refresh_token(self, client, refresh_token, scopes):
            user_id = self._refresh_user.pop(refresh_token.token, None)
            token = await super().exchange_refresh_token(client, refresh_token, scopes)
            self._bind(token, user_id)
            self._persist_tokens(token, user_id)
            return token

    slate_auth = SlateOAuthProvider(
        base_url=config.SLATE_BASE_URL.rstrip("/") + "/mcp",
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
        "with their past notes are the product, not metadata. "
        "FEEDBACK: after you use recalled items to compose a response, when some "
        "clearly helped and others were noise, call `mark_relevance` with their ids "
        "— it tunes future recall. Only when you have a clear judgment."
    ),
    auth=slate_auth,
)


def _conn():
    return store.connect()


def _user_id() -> str:
    """Resolve the authenticated user for this tool call (AUTH.md §2).

    With OAuth enabled, identity comes ONLY from the token claims bound at
    login — an unbound token gets a hard error, never a default corpus.
    Without OAuth (local dev), everything is DEFAULT_USER_ID.
    """
    if slate_auth is None:
        return config.DEFAULT_USER_ID
    from fastmcp.server.dependencies import get_access_token
    token = get_access_token()
    user_id = (token.claims or {}).get("user_id") if token else None
    if not user_id:
        raise ToolError("Unauthenticated: no user is bound to this token — re-authorize.")
    return user_id


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
        from core.auth import authenticate
        conn = _conn()
        try:
            user = authenticate(conn, str(form.get("username", "")),
                                str(form.get("password", "")))
        finally:
            conn.close()
        if user is None:
            return HTMLResponse(
                _LOGIN_PAGE.format(auth_id=auth_id,
                                   error='<p class="error">Invalid username or password.</p>'),
                status_code=401)
        return RedirectResponse(
            await slate_auth.approve_authorization(auth_id, user["id"]),
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
    user_id = _user_id()
    conn = _conn()
    try:
        receipt = encode(conn, user_id, text, title=title.strip(), source="mcp")
    except ValueError as e:
        raise ToolError(str(e))
    finally:
        conn.close()
    # W1 (raw + receipt) is committed and returned now; the predictor-driven
    # fragmentation/routing (W2–W8) runs in the background — it re-embeds and may
    # call the resolver, so it must not block the save. A failure here just leaves
    # the episode for the nightly refine_pending sweep.
    from core.write import trigger_refine_async
    trigger_refine_async(user_id, receipt["episode_id"])
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
    graph navigation, with why-now signals (🌉 bridged concepts, 🔁 recurring
    claims, 🕰️ long-dormant thinking resurfacing, 2-hop = non-obvious
    connection). Escalate with assemble_context or get_concept when a hit
    deserves the full picture. After you use the results, mark_relevance() tells
    Slate which ones helped.
    """
    from core.recall import recall as _recall
    return _recall(_conn(), _user_id(), query, k=k)


@mcp.tool
def assemble_context(topic: str) -> str:
    """Pull the user's full prior thinking on a topic as compact markdown:
    synthesis from concepts/claims PLUS verbatim source spans, with provenance
    (which note, which date).

    Call when you're about to write something substantive on a topic the user
    has history with, or after recall() surfaces a hit worth expanding. Output
    is budgeted (most-relevant-first) for direct injection."""
    # RESONANCE retrieve: navigate the consolidated graph (PE-gated, fan-out-
    # normalised spread) → materialise the brightest regions as a distilled concept
    # frame + verbatim depth/breadth spans. Best Slate variant on the SR@B eval
    # (narrow 85.7 / tail 80, paragraph 100, broad at its 33% ceiling — beats the old
    # hybrid 64/60/33). Calibration is the user's fitted profile over the in-code
    # defaults. signals=True emits R8 (fetched/dropped) for consolidation C13; commit
    # before close since append_event does NOT commit. See docs/retrieve-resonance-design.md.
    from core import resonance
    conn = _conn()
    user_id = _user_id()
    try:
        out = resonance.resonance_context(conn, user_id, topic, signals=True)
        conn.commit()
        return out
    finally:
        conn.close()


@mcp.tool
def mark_relevance(query: str, relevant: list[str] | None = None,
                   irrelevant: list[str] | None = None) -> dict:
    """Report which recalled items actually helped answer `query` and which were
    noise — explicit feedback that tunes future recall.

    Call AFTER you've used recalled context to compose a response, only when you
    have a clear judgment. `relevant` / `irrelevant` take ids straight from
    recall() headlines (claim or concept ids) or note ids (get_note /
    list_recent_notes) — pass only the ids you're confident about, omit the rest.
    Consolidation keeps the relevant ones in the foreground and demotes the
    irrelevant ones, so the next recall on a similar query ranks better."""
    rel = [i for i in (relevant or []) if i]
    irr = [i for i in (irrelevant or []) if i]
    if not rel and not irr:
        return {"status": "noop", "message": "no ids provided"}
    from core.retrieve import record_relevance_feedback
    conn = _conn()
    user_id = _user_id()
    try:
        record_relevance_feedback(conn, user_id, query, relevant=rel, irrelevant=irr)
        conn.commit()
        return {"status": "recorded", "query": query,
                "relevant": len(rel), "irrelevant": len(irr)}
    finally:
        conn.close()


@mcp.tool
def get_note(episode_id: str) -> dict:
    """Fetch one note in full: raw text, receipt, blueprint (post-consolidation),
    and the canonical claims it supports. Use after recall/list_recent_notes."""
    from core.recall import get_episode
    result = get_episode(_conn(), _user_id(), episode_id)
    if not result:
        raise ToolError(f"Note not found: {episode_id}")
    return result


@mcp.tool
def list_recent_notes(limit: int = 10) -> list[dict]:
    """List the most recently saved notes (id, title, ts, essence).
    Entry point for browsing; follow up with get_note."""
    from core.recall import list_episodes
    return list_episodes(_conn(), _user_id(), limit=min(limit, 50))


@mcp.tool
def get_concept(concept_id: str) -> dict:
    """Fetch one concept in full: label, canonical description, state/strength,
    member claims with provenance (episodes + verbatim sentences), and its
    relations including bridges. Use after recall surfaces a concept hit."""
    from core.recall import get_concept as _gc
    result = _gc(_conn(), _user_id(), concept_id)
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
           WHERE user_id = ? AND payload_json LIKE ? ORDER BY seq LIMIT ?""",
        (_user_id(), f"%{concept_id}%", limit)).fetchall()
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
    return _digest(_conn(), _user_id(), since_hours=since_hours)


@mcp.tool
def reconstruct_note(episode_id: str) -> dict:
    """Regenerate a note from its stored blueprint (essence + claims + spine +
    verbatim style samples) and score fidelity vs the original. Use to show
    how much of a note Slate's minimal storage can recreate."""
    from core.reconstruct import reconstruct
    try:
        return reconstruct(_conn(), _user_id(), episode_id)
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool
def synthesize(concept_a: str, concept_b: str) -> dict:
    """Draft a NEW short document from the intersection of two concepts —
    Slate's 'create new docs from emerging learnings'. Best on bridged pairs
    (see list_bridges); uses the stored bridge rationale automatically."""
    from core.reconstruct import synthesize as _syn
    try:
        return _syn(_conn(), _user_id(), concept_a, concept_b)
    except ValueError as e:
        raise ToolError(str(e))


@mcp.tool
def list_bridges(limit: int = 20) -> list[dict]:
    """List discovered bridges between concept pairs (newest first) — the
    non-obvious connections consolidation surfaced. Entry point for synthesize."""
    from core.reconstruct import bridges
    return bridges(_conn(), _user_id(), limit=limit)


@mcp.tool
def stats() -> dict:
    """Corpus size (episodes, claims, concepts, relations) and the last
    consolidation run. Use for health checks / "how big is my Slate?"."""
    return store.stats(_conn(), _user_id())
