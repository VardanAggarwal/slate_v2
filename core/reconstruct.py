"""reconstruct(episode_id) — regenerate doc from blueprint with fidelity score. synthesize(concept_a, concept_b) — draft a new doc from the intersection of two concepts. See PLAN.md §5.

North-star metric (PLAN.md §5): unique-claim bytes ÷ reconstructable bytes —
how few stored bytes recreate how much of the original document.
"""
import json

from core import llm, store
from core.recall import get_concept, get_episode

PROMPT_RECONSTRUCT = """Reconstruct the author's original note from its distilled structure. Write in first person, in the author's voice. The verbatim sentences show their actual style — match it. Cover every claim; follow the spine's argumentative order; do not invent content beyond the claims and assumptions. Return only the reconstructed note text.

STRUCTURE:
{structure}
"""

PROMPT_FIDELITY = """Rate how faithfully RECONSTRUCTION preserves ORIGINAL: its core argument, every distinct point, and the author's voice. Return ONLY JSON: {"fidelity": <1-10>, "missing": ["points present in the original but lost"], "invented": ["claims present in the reconstruction but not the original"]}

ORIGINAL:
{original}

RECONSTRUCTION:
{reconstruction}
"""

PROMPT_SYNTHESIZE = """The author's knowledge base found a connection between two of their concept clusters. Write a short new document (300-500 words, first person, the author's voice) that develops this connection into an actual idea — not a summary of the two concepts, but the new thought their intersection makes possible. Ground every move in the claims provided; do not invent facts.

CONNECTION RATIONALE: {rationale}

CONCEPT A: {a}

CONCEPT B: {b}

Return only the document text.
"""


def reconstruct(conn, user_id: str, episode_id: str) -> dict:
    """Regenerate a note from its blueprint; report fidelity vs raw_text."""
    ep = get_episode(conn, user_id, episode_id)
    if not ep:
        raise ValueError(f"episode not found: {episode_id}")
    if not ep["blueprint"]:
        raise ValueError(f"episode {episode_id} has no blueprint yet — "
                         "it has not been consolidated")

    bp = ep["blueprint"]
    structure = {
        "essence": bp.get("essence"),
        "clusters": [{"label": c.get("label"), "kernel": c.get("kernel"),
                      "claims": c.get("claims", []),
                      "verbatim_style_samples": c.get("representative_sentences", [])}
                     for c in bp.get("clusters", [])],
        "assumptions": bp.get("assumptions", []),
        "spine": bp.get("spine", []),
    }
    gen = llm.call(PROMPT_RECONSTRUCT.format(
        structure=json.dumps(structure, ensure_ascii=False, indent=1)),
        tier="mechanical", max_tokens=2048, json_out=False)
    text = gen["text"].strip()

    judge = llm.call(PROMPT_FIDELITY
                     .replace("{original}", ep["raw_text"])
                     .replace("{reconstruction}", text),
                     tier="mechanical", max_tokens=2048)

    stored_bytes = len(json.dumps(structure).encode())
    original_bytes = len(ep["raw_text"].encode())
    return {
        "episode_id": episode_id,
        "title": ep["title"],
        "reconstruction": text,
        "fidelity": judge["json"].get("fidelity"),
        "missing": judge["json"].get("missing", []),
        "invented": judge["json"].get("invented", []),
        "compression": {
            "stored_bytes": stored_bytes,
            "original_bytes": original_bytes,
            "ratio": round(stored_bytes / max(1, original_bytes), 3),
        },
        "cost": round(gen["cost"] + judge["cost"], 4),
    }


def _concept_block(conn, user_id: str, concept_id: str) -> str:
    c = get_concept(conn, user_id, concept_id)
    if not c:
        raise ValueError(f"concept not found: {concept_id}")
    claims = []
    for m in c["members"][:10]:
        prov = ""
        if m["support"]:
            s = m["support"][0]
            prov = f' (from "{s["title"] or "untitled"}", {s["ts"][:10]})'
        claims.append(f"- {m['text']}{prov}")
    return f"{c['label']} — {c['canonical']}\n" + "\n".join(claims)


def synthesize(conn, user_id: str, concept_a: str, concept_b: str,
               rationale: str | None = None) -> dict:
    """Draft a NEW document from the intersection of two concepts."""
    if rationale is None:
        rationale = "an unstated affinity between the two clusters"

    result = llm.call(PROMPT_SYNTHESIZE.format(
        rationale=rationale,
        a=_concept_block(conn, user_id, concept_a),
        b=_concept_block(conn, user_id, concept_b)),
        tier="judgment", max_tokens=2048, json_out=False)
    return {"concepts": [concept_a, concept_b], "rationale": rationale,
            "document": result["text"].strip(), "cost": round(result["cost"], 4)}
