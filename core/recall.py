"""Spreading-activation retrieval: vector seed over claims/concepts, graph hops via relations, why-now ranking. 100% local. See PLAN.md §5.

Also hosts the read/browse API (§5): get_episode, list_episodes, get_concept,
get_claim, assemble_context — plain SQL, no LLM, no network.
"""
import json
from datetime import datetime, timezone

from core import config, store
from core.encode import get_embedder

# Activation transfer per edge type (per hop)
SPREAD_CLAIM_TO_CONCEPT = 0.7
SPREAD_CONCEPT_TO_CLAIM = 0.6
SPREAD_RELATION = 0.6
HOPS = 2
SEED_CLAIMS = 12
SEED_CONCEPTS = 6
MIN_ACTIVATION = 0.05

FREQUENT_STRENGTH = 2.0   # 🔁 claim re-encountered (1.0 + 2×SUPPORT_BUMP)
TIME_GAP_DAYS = 45        # 🕰️ resurfacing after this long


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


# ── Spreading activation ──────────────────────────────────────────────────────
def recall(conn, query: str, k: int = 8) -> list[dict]:
    """Rank claims + concepts for a query. Returns compact headline dicts —
    ~50 tokens each — so callers (MCP `recall`) can call speculatively."""
    emb = get_embedder().encode([query], normalize_embeddings=True,
                                show_progress_bar=False)[0]

    activation: dict[str, float] = {}
    via: dict[str, str] = {}  # node -> how it was reached (for "non-obvious" signal)

    for hit in store.knn_claims(conn, emb, k=SEED_CLAIMS):
        if hit["similarity"] > 0:
            activation[hit["claim_id"]] = max(activation.get(hit["claim_id"], 0),
                                              hit["similarity"])
            via[hit["claim_id"]] = "seed"
    for hit in store.knn_concepts(conn, emb, k=SEED_CONCEPTS):
        if hit["similarity"] > 0:
            activation[hit["id"]] = max(activation.get(hit["id"], 0), hit["similarity"])
            via[hit["id"]] = "seed"

    frontier = dict(activation)
    for _ in range(HOPS):
        next_frontier: dict[str, float] = {}

        def push(node: str, value: float, source: str):
            if value < MIN_ACTIVATION:
                return
            if value > activation.get(node, 0):
                activation[node] = value
                via.setdefault(node, source)
                next_frontier[node] = max(next_frontier.get(node, 0), value)

        for node, act in frontier.items():
            if node.startswith("clm_"):
                for r in conn.execute(
                        "SELECT concept_id FROM concept_members WHERE claim_id = ?", (node,)):
                    push(r["concept_id"], act * SPREAD_CLAIM_TO_CONCEPT, node)
            elif node.startswith("cpt_"):
                for r in conn.execute(
                        "SELECT claim_id FROM concept_members WHERE concept_id = ?", (node,)):
                    push(r["claim_id"], act * SPREAD_CONCEPT_TO_CLAIM, node)
            for r in conn.execute(
                    """SELECT from_id, to_id, relation, weight FROM relations
                       WHERE from_id = ? OR to_id = ?""", (node, node)):
                other = r["to_id"] if r["from_id"] == node else r["from_id"]
                w = min(1.0, r["weight"] or 1.0)
                bridge_tag = f"bridge:{node}" if r["relation"] == "bridges" else node
                push(other, act * SPREAD_RELATION * w, bridge_tag)
        frontier = next_frontier
        if not frontier:
            break

    results = []
    for node, act in activation.items():
        entry = _score_node(conn, node, act, via.get(node, "seed"))
        if entry:
            results.append(entry)
    results.sort(key=lambda r: -r["score"])
    return results[:k]


def _score_node(conn, node: str, activation: float, via: str) -> dict | None:
    signals = []
    if via != "seed":
        signals.append("2-hop" if not via.startswith("bridge:") else "🌉 via bridge")

    if node.startswith("clm_"):
        c = store.get_claim(conn, node)
        if not c:
            return None
        score = activation * min(2.0, 0.5 + c["strength"] / 2.0)
        if c["strength"] >= FREQUENT_STRENGTH:
            signals.append("🔁 recurring")
        gap = _days_since(c["last_seen"])
        if gap >= TIME_GAP_DAYS:
            signals.append(f"🕰️ last seen {gap}d ago")
        n_support = conn.execute(
            "SELECT COUNT(*) AS n FROM claim_support WHERE claim_id = ?",
            (node,)).fetchone()["n"]
        return {"type": "claim", "id": node, "text": c["text"],
                "strength": round(c["strength"], 2), "n_episodes": n_support,
                "score": round(score, 4), "signals": signals}

    if node.startswith("cpt_"):
        c = store.get_concept(conn, node)
        if not c:
            return None
        state_mult = config.HEALTH_SCORES.get(c["state"], 1.0)
        score = activation * state_mult
        has_bridge = conn.execute(
            """SELECT 1 FROM relations WHERE relation = 'bridges'
               AND (from_id = ? OR to_id = ?) LIMIT 1""", (node, node)).fetchone()
        if has_bridge:
            signals.append("🌉 bridged")
        gap = _days_since(c["last_activity"])
        if gap >= TIME_GAP_DAYS:
            signals.append(f"🕰️ dormant {gap}d")
        n_members = conn.execute(
            "SELECT COUNT(*) AS n FROM concept_members WHERE concept_id = ?",
            (node,)).fetchone()["n"]
        return {"type": "concept", "id": node, "label": c["label"],
                "canonical": c["canonical"], "state": c["state"],
                "n_claims": n_members, "score": round(score, 4), "signals": signals}
    return None


# ── Read/browse API (§5) — plain SQL, $0 ──────────────────────────────────────
def get_episode(conn, episode_id: str) -> dict | None:
    ep = store.get_episode(conn, episode_id)
    if not ep:
        return None
    bp_row = conn.execute(
        """SELECT payload_json FROM events WHERE type = 'BLUEPRINTED'
           AND json_extract(payload_json, '$.episode_id') = ?
           ORDER BY seq DESC LIMIT 1""", (episode_id,)).fetchone()
    claims = [dict(r) for r in conn.execute(
        """SELECT s.claim_id, c.text, s.verbatim_sentence FROM claim_support s
           JOIN claims c ON c.id = s.claim_id WHERE s.episode_id = ?""",
        (episode_id,))]
    return {"id": ep["id"], "ts": ep["ts"], "title": ep["title"],
            "source": ep["source"], "raw_text": ep["raw_text"],
            "receipt": json.loads(ep["receipt_json"] or "{}"),
            "blueprint": (json.loads(bp_row["payload_json"])["blueprint"]
                          if bp_row else None),
            "claims": claims}


def list_episodes(conn, limit: int = 20, before: str | None = None) -> list[dict]:
    q = "SELECT id, ts, title, raw_text FROM episodes"
    args: list = []
    if before:
        q += " WHERE ts < ?"
        args.append(before)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in conn.execute(q, args):
        bp = conn.execute(
            """SELECT json_extract(payload_json, '$.blueprint.essence') AS essence
               FROM events WHERE type = 'BLUEPRINTED'
               AND json_extract(payload_json, '$.episode_id') = ?
               ORDER BY seq DESC LIMIT 1""", (r["id"],)).fetchone()
        essence = (bp["essence"] if bp and bp["essence"]
                   else r["raw_text"][:120].replace("\n", " "))
        out.append({"id": r["id"], "ts": r["ts"], "title": r["title"],
                    "essence": essence})
    return out


def get_concept(conn, concept_id: str) -> dict | None:
    c = store.get_concept(conn, concept_id)
    if not c:
        return None
    members = []
    for m in conn.execute(
            """SELECT cm.claim_id, cl.text, cl.strength FROM concept_members cm
               JOIN claims cl ON cl.id = cm.claim_id
               WHERE cm.concept_id = ? ORDER BY cl.strength DESC""", (concept_id,)):
        support = [dict(s) for s in conn.execute(
            """SELECT cs.episode_id, e.title, e.ts, cs.verbatim_sentence
               FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
               WHERE cs.claim_id = ?""", (m["claim_id"],))]
        members.append({"claim_id": m["claim_id"], "text": m["text"],
                        "strength": m["strength"], "support": support})
    relations = [dict(r) for r in conn.execute(
        """SELECT from_id, to_id, relation, weight, evidence_episode_id
           FROM relations WHERE from_id = ? OR to_id = ?""",
        (concept_id, concept_id))]
    return {**{k: c[k] for k in c.keys()}, "members": members,
            "relations": relations}


def get_claim(conn, claim_id: str) -> dict | None:
    c = store.get_claim(conn, claim_id)
    if not c:
        return None
    support = [dict(s) for s in conn.execute(
        """SELECT cs.episode_id, e.title, e.ts, cs.verbatim_sentence
           FROM claim_support cs JOIN episodes e ON e.id = cs.episode_id
           WHERE cs.claim_id = ? ORDER BY e.ts""", (claim_id,))]
    concepts = [r["concept_id"] for r in conn.execute(
        "SELECT concept_id FROM concept_members WHERE claim_id = ?", (claim_id,))]
    return {**{k: c[k] for k in c.keys()}, "support": support, "concepts": concepts}


def assemble_context(conn, topic: str, max_chars: int = 6000) -> str:
    """The Claude-first read: recall(topic), group claim hits by concept, return
    compact provenance-rich markdown sized for in-conversation injection."""
    hits = recall(conn, topic, k=12)
    if not hits:
        return f"_Slate has nothing stored about “{topic}” yet._"

    concept_hits = [h for h in hits if h["type"] == "concept"]
    claim_hits = [h for h in hits if h["type"] == "claim"]

    # Group loose claim hits under their concept when it's also relevant
    grouped: dict[str, list[dict]] = {}
    loose: list[dict] = []
    for h in claim_hits:
        full = get_claim(conn, h["id"])
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
        full = get_concept(conn, ch["id"])
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
