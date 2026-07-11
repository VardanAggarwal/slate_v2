"""The cheap headline read: rank claims/concepts for a query with why-now signals
(~50 tok each), so callers can call speculatively before composing. 100% local.
See PLAN.md §5.

Navigation is SHARED with assemble_context: `recall()` runs `resonance.activate()`
(the denoised navigate stage — probe confluence + PE-gated, fan-out-normalised
spread) and stops at the activation field, rendering headlines. assemble_context
runs the same `activate()` then continues into verbatim materialisation. One engine,
two heads (docs/retrieve-resonance-design.md).

Also hosts the read/browse API (§5): get_episode, list_episodes, get_concept,
get_claim — plain SQL, no LLM, no network.

Every function takes an explicit user_id and every query filters on it
(AUTH.md §1/§3) — graph hops must never cross into another user's corpus.
"""
import json
from datetime import datetime, timezone

from core import config, store

SCORE_POOL = 40           # score this many brightest nodes (bounds _score_node SQL)

FREQUENT_STRENGTH = 2.0   # 🔁 claim re-encountered (1.0 + 2×SUPPORT_BUMP)
TIME_GAP_DAYS = 45        # 🕰️ resurfacing after this long
BACKGROUND_SCORE_FACTOR = 0.5  # 🌫️ C9: demote a folded-into-theme claim's standalone pull


def _days_since(iso_ts: str | None) -> int:
    if not iso_ts:
        return 999
    try:
        dt = datetime.fromisoformat(iso_ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).days
    except (ValueError, TypeError):
        return 999


# ── Headline ranking — navigation shared with assemble_context ────────────────
def recall(conn, user_id: str, query: str, k: int = 8) -> list[dict]:
    """Rank claims + concepts for a query. Returns compact headline dicts —
    ~50 tokens each — so callers (MCP `recall`) can call speculatively.

    Navigation is the resonance navigate stage (`resonance.activate`): probe
    confluence + PE-gated, fan-out-normalised spread over the consolidated graph.
    recall stops at the scored activation field and renders headlines; the heavy
    `assemble_context` runs the same `activate()` then materialises verbatim spans."""
    from core import calibration as calib, resonance, retrieve
    field = resonance.activate(conn, user_id, query)
    nodes, seeded = field["nodes"], field["seeded"]
    if not nodes:
        return []

    # Query-tagged relevance-feedback rerank — the same reweight assemble_context's
    # resonance_recall applies, so both heads rank consistently (activate() itself
    # stays pure — consolidation's query-injection must not see feedback). Positive-
    # only, claim-level, off → byte-identical. docs/relevance-feedback-findings.md.
    cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION,
                              **resonance.DEFAULT_CALIBRATION}, user_id)
    if bool(cal.get("res_fb_rerank", False)):
        retrieve.apply_feedback_rerank(
            nodes, field["q_emb"], retrieve.feedback_rerank_index(conn, user_id),
            alpha=float(cal.get("res_fb_alpha", 1.0)),
            w_pos=float(cal.get("res_fb_w_pos", 0.45)))

    # Score the brightest nodes (by navigation salience) with why-now signals.
    # Pre-trim to a bounded pool so _score_node's per-node SQL stays cheap on a wide
    # field. via = "seed" (direct hit, no tag) vs "spread" (reached by a hop → the
    # "2-hop" non-obvious signal).
    ranked = sorted(nodes.items(), key=lambda kv: -kv[1]["salience"])
    results = []
    for node, sc in ranked[:SCORE_POOL]:
        via = "seed" if node in seeded else "spread"
        entry = _score_node(conn, user_id, node, sc["salience"], via)
        if entry:
            results.append(entry)
    results.sort(key=lambda r: -r["score"])
    return results[:k]


def _score_node(conn, user_id: str, node: str, activation: float, via: str) -> dict | None:
    signals = []
    if via != "seed":
        signals.append("2-hop")

    if node.startswith("clm_"):
        c = store.get_claim(conn, user_id, node)
        if not c:
            return None
        score = activation * min(2.0, 0.5 + c["strength"] / 2.0)
        if c["strength"] >= FREQUENT_STRENGTH:
            signals.append("🔁 recurring")
        gap = _days_since(c["last_seen"])
        if gap >= TIME_GAP_DAYS:
            signals.append(f"🕰️ last seen {gap}d ago")
        n_support = conn.execute(
            "SELECT COUNT(*) AS n FROM claim_support WHERE claim_id = ? AND user_id = ?",
            (node, user_id)).fetchone()["n"]
        # C9: a background claim's theme stands for it — demote its standalone pull
        # (don't drop it; the concept surfaces instead).
        if c["background"]:
            score *= BACKGROUND_SCORE_FACTOR
            signals.append("🌫️ background")
        out = {"type": "claim", "id": node, "text": c["text"],
               "strength": round(c["strength"], 2), "n_episodes": n_support,
               "score": round(score, 4), "signals": signals,
               "status": c["status"]}
        # C8: the current view is returned, but contestation is always surfaced.
        if c["version_group"] and len(store.claim_versions(conn, user_id,
                                                            c["version_group"])) > 1:
            signals.append("⚖️ superseded" if c["status"] == "superseded"
                           else "⚖️ contested")
            if c["superseded_by"]:
                out["superseded_by"] = c["superseded_by"]
        return out

    if node.startswith("cpt_"):
        c = store.get_concept(conn, user_id, node)
        if not c:
            return None
        state_mult = config.HEALTH_SCORES.get(c["state"], 1.0)
        score = activation * state_mult
        gap = _days_since(c["last_activity"])
        if gap >= TIME_GAP_DAYS:
            signals.append(f"🕰️ dormant {gap}d")
        n_members = conn.execute(
            "SELECT COUNT(*) AS n FROM concept_members WHERE concept_id = ? AND user_id = ?",
            (node, user_id)).fetchone()["n"]
        return {"type": "concept", "id": node, "label": c["label"],
                "canonical": c["canonical"], "state": c["state"],
                "n_claims": n_members, "score": round(score, 4), "signals": signals}
    return None


# ── Read/browse API (§5) — plain SQL, $0 ──────────────────────────────────────
def get_episode(conn, user_id: str, episode_id: str) -> dict | None:
    ep = store.get_episode(conn, user_id, episode_id)
    if not ep:
        return None
    bp_row = conn.execute(
        """SELECT payload_json FROM events WHERE type = 'BLUEPRINTED'
           AND user_id = ?
           AND json_extract(payload_json, '$.episode_id') = ?
           ORDER BY seq DESC LIMIT 1""", (user_id, episode_id)).fetchone()
    claims = [dict(r) for r in conn.execute(
        """SELECT s.claim_id, c.text, s.verbatim_sentence FROM claim_support s
           JOIN claims c ON c.id = s.claim_id
           WHERE s.episode_id = ? AND s.user_id = ?""",
        (episode_id, user_id))]
    out = {"id": ep["id"], "ts": ep["ts"], "title": ep["title"],
           "source": ep["source"], "raw_text": ep["raw_text"],
           "receipt": json.loads(ep["receipt_json"] or "{}"),
           "blueprint": (json.loads(bp_row["payload_json"])["blueprint"]
                         if bp_row else None),
           "claims": claims}
    # Edit lineage: a superseded note stays fetchable (provenance) but flagged;
    # a revision points back at what it replaced.
    newer = store.episode_superseded_by(conn, user_id, episode_id)
    older = store.episode_supersedes(conn, user_id, episode_id)
    if newer:
        out["superseded_by"] = newer
    if older:
        out["edited_from"] = older
    return out


def list_episodes(conn, user_id: str, limit: int = 20,
                  before: str | None = None) -> list[dict]:
    # superseded (edited-away) versions stay out of browse — get_episode still
    # fetches them directly for provenance.
    q = ("SELECT id, ts, title, raw_text FROM episodes WHERE user_id = ? "
         "AND NOT EXISTS (SELECT 1 FROM episode_supersessions ss WHERE ss.old_id = episodes.id)")
    args: list = [user_id]
    if before:
        q += " AND ts < ?"
        args.append(before)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(q, args):
        bp = conn.execute(
            """SELECT json_extract(payload_json, '$.blueprint.essence') AS essence
               FROM events WHERE type = 'BLUEPRINTED' AND user_id = ?
               AND json_extract(payload_json, '$.episode_id') = ?
               ORDER BY seq DESC LIMIT 1""", (user_id, r["id"])).fetchone()
        essence = (bp["essence"] if bp and bp["essence"]
                   else r["raw_text"][:120].replace("\n", " "))
        out.append({"id": r["id"], "ts": r["ts"], "title": r["title"],
                    "essence": essence})
    return out


def get_concept(conn, user_id: str, concept_id: str) -> dict | None:
    c = store.get_concept(conn, user_id, concept_id)
    if not c:
        return None
    members = []
    for m in conn.execute(
            """SELECT cm.claim_id, cl.text, cl.strength FROM concept_members cm
               JOIN claims cl ON cl.id = cm.claim_id
               WHERE cm.concept_id = ? AND cm.user_id = ?
               ORDER BY cl.strength DESC""", (concept_id, user_id)):
        support = [dict(s) for s in conn.execute(
            """SELECT cs.episode_id, e.title, e.ts, cs.verbatim_sentence
               FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
               WHERE cs.claim_id = ? AND cs.user_id = ?""",
            (m["claim_id"], user_id))]
        members.append({"claim_id": m["claim_id"], "text": m["text"],
                        "strength": m["strength"], "support": support})
    relations = [dict(r) for r in conn.execute(
        """SELECT from_id, to_id, relation, weight, evidence_episode_id
           FROM relations WHERE (from_id = ? OR to_id = ?) AND user_id = ?""",
        (concept_id, concept_id, user_id))]
    return {**{k: c[k] for k in c.keys()}, "members": members,
            "relations": relations}


def get_claim(conn, user_id: str, claim_id: str) -> dict | None:
    c = store.get_claim(conn, user_id, claim_id)
    if not c:
        return None
    support = [dict(s) for s in conn.execute(
        """SELECT cs.episode_id, e.title, e.ts, cs.verbatim_sentence
           FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
           WHERE cs.claim_id = ? AND cs.user_id = ? ORDER BY e.ts""",
        (claim_id, user_id))]
    concepts = [r["concept_id"] for r in conn.execute(
        "SELECT concept_id FROM concept_members WHERE claim_id = ? AND user_id = ?",
        (claim_id, user_id))]
    return {**{k: c[k] for k in c.keys()}, "support": support, "concepts": concepts}


def assemble_context(conn, user_id: str, topic: str, max_chars: int = 6000) -> str:
    """The Claude-first read: recall(topic), group claim hits by concept, return
    compact provenance-rich markdown sized for in-conversation injection."""
    hits = recall(conn, user_id, topic, k=12)
    if not hits:
        return f"_Slate has nothing stored about “{topic}” yet._"

    concept_hits = [h for h in hits if h["type"] == "concept"]
    claim_hits = [h for h in hits if h["type"] == "claim"]

    # Group loose claim hits under their concept when it's also relevant
    grouped: dict[str, list[dict]] = {}
    loose: list[dict] = []
    for h in claim_hits:
        full = get_claim(conn, user_id, h["id"])
        cids = full["concepts"] if full else []
        (grouped.setdefault(cids[0], []) if cids else loose).append(h)

    lines = [f"## Slate context: {topic}\n"]
    seen_concepts = set()
    for ch in concept_hits:
        seen_concepts.add(ch["id"])
        sig = f" ({', '.join(ch['signals'])})" if ch["signals"] else ""
        lines.append(f"### {ch['label']}{sig}")
        if ch["canonical"]:
            lines.append(f"{ch['canonical']}")
        for h in grouped.pop(ch["id"], [])[:4]:
            lines.append(f"- {h['text']} _(seen in {h['n_episodes']} note(s))_")
        full = get_concept(conn, user_id, ch["id"])
        for m in (full["members"] if full else [])[:3]:
            src = m["support"][0] if m["support"] else None
            prov = f" — {src['title'] or 'untitled'}, {src['ts'][:10]}" if src else ""
            lines.append(f"- {m['text']}{prov}")
        lines.append("")
    for cid, hs in grouped.items():
        for h in hs:
            lines.append(f"- {h['text']} _(claim, strength {h['strength']})_")
    for h in loose:
        lines.append(f"- {h['text']} _(unattached claim)_")

    out, total = [], 0
    for line in lines:  # most-relevant-first budget cut
        total += len(line) + 1
        if total > max_chars:
            out.append("\n_(truncated — ask for a specific concept for more)_")
            break
        out.append(line)
    return "\n".join(out)
