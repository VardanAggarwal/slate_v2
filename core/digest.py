"""Morning digest: read last night's events, one LLM call, markdown summary (bridges, contradictions, strengthened/dormant concepts). See PLAN.md §5.

The structured digest is deterministic (pure event reads); the single LLM
call only rewrites it into prose, and is skipped when unavailable — the
digest must never fail just because a provider is down.
"""
import json
from datetime import datetime, timedelta, timezone

from core import llm, store

PROMPT_DIGEST = """Rewrite this nightly knowledge-base digest as 3-6 short, warm, specific bullet lines for its author. Keep every emoji marker, concept name and date; drop nothing substantive; add nothing. Return only the rewritten markdown.

"""


def _concept_label(conn, user_id: str, concept_id: str, payloads_by_concept: dict) -> str:
    c = store.get_concept(conn, user_id, concept_id)
    if c:
        return c["label"] or concept_id
    return payloads_by_concept.get(concept_id, concept_id)  # merged/split away


def digest(conn, user_id: str, since_hours: int = 36, polish: bool = False) -> str:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=since_hours)).isoformat()
    rows = conn.execute(
        "SELECT ts, type, payload_json FROM events WHERE ts >= ? AND user_id = ? ORDER BY seq",
        (cutoff, user_id)).fetchall()
    if not rows:
        return "_Nothing consolidated recently — save some notes and the nightly run will have material._"

    # labels for concepts that may no longer exist (merged/split)
    labels: dict[str, str] = {}
    events = []
    for r in rows:
        p = json.loads(r["payload_json"])
        events.append((r["type"], p))
        if p.get("label") and p.get("concept_id"):
            labels[p["concept_id"]] = p["label"]

    lines = []
    for type_, p in events:
        if type_ == "BRIDGED":
            a = _concept_label(conn, user_id, p["a"], labels)
            b = _concept_label(conn, user_id, p["b"], labels)
            lines.append(f"🌉 New bridge: **{a}** × **{b}** — {p.get('rationale', '')}")
        elif type_ == "RELATED" and p.get("relation") == "contradicts":
            frm = store.get_claim(conn, user_id, p["from_id"])
            to = store.get_claim(conn, user_id, p["to_id"])
            if frm and to:
                lines.append(f"⚡ Contradiction: “{frm['text']}” vs earlier “{to['text']}”")
        elif type_ == "MERGED":
            w = _concept_label(conn, user_id, p["winner_id"], labels)
            loser = (p.get("loser_snapshot", {}).get("concept") or {}).get("label", p["loser_id"])
            lines.append(f"🧲 Merged: **{loser}** folded into **{w}**")
        elif type_ == "SPLIT":
            parent = (p.get("snapshot", {}).get("concept") or {}).get("label", p["concept_id"])
            children = ", ".join(f"**{c['label']}**" for c in p.get("into", []))
            lines.append(f"✂️ Split: **{parent}** → {children}")
        elif type_ == "CONCEPT_CREATED":
            lines.append(f"🌱 New concept: **{p.get('label', p['concept_id'])}** "
                         f"({len(p.get('claim_ids', []))} claims)")
        elif type_ == "DECAYED" and p.get("state_to") == "dormant":
            label = _concept_label(conn, user_id, p["concept_id"], labels)
            lines.append(f"🕰️ Going dormant: **{label}** — "
                         f"last touched {p.get('days_inactive', '?')} days ago")

    strengthened: dict[str, int] = {}
    for type_, p in events:
        if type_ == "STRENGTHENED" and p.get("claim_id"):
            strengthened[p["claim_id"]] = strengthened.get(p["claim_id"], 0) + 1
        elif type_ == "CANONICALIZED" and p.get("action") == "support":
            strengthened[p["claim_id"]] = strengthened.get(p["claim_id"], 0) + 1
    for claim_id, n in sorted(strengthened.items(), key=lambda x: -x[1])[:3]:
        c = store.get_claim(conn, user_id, claim_id)
        if c and n >= 1:
            total = conn.execute(
                "SELECT COUNT(*) AS n FROM claim_support WHERE claim_id = ? AND user_id = ?",
                (claim_id, user_id)).fetchone()["n"]
            lines.append(f"🔁 Strengthened: “{c['text']}” ({total} encounter(s))")

    if not lines:
        n_enc = sum(1 for t, _ in events if t == "ENCODED")
        lines.append(f"📝 {n_enc} note(s) encoded; nothing structural changed.")

    md = "\n".join(dict.fromkeys(lines))  # dedupe, keep order
    if polish:
        try:
            md = llm.call(PROMPT_DIGEST + md, tier="mechanical",
                          max_tokens=1024, json_out=False)["text"]
        except llm.LLMError:
            pass  # structured digest stands on its own
    return md
