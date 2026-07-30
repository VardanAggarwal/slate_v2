"""Gate A runner (docs/evidence-lane-plan.md §5): hand-feed sources, score the verdicts.

The plan's gate is "read every receipt and confirm the labels are trustworthy —
genuine backing reads 📎, a deliberately contradicting source reads ⚡". This makes
that repeatable and separates the two ways it can fail:

  * **stance** — the pair reached the classifier and it returned the wrong label.
  * **reach**  — the pair never reached the classifier at all, because the best claim
    sat below ECHO_THRESHOLD (0.72). Nothing is wrong with the stance model here; the
    threshold is simply the binding constraint. `field: "far"` rows in the gold exist
    to measure exactly this, and are reported separately rather than as failures.

Conflating the two would send you tuning the wrong knob.

ALWAYS runs on a COPY of the database — evidence saves are real episodes, and
episodes are immutable, so a mistaken run against the live corpus cannot be undone.

Usage:
    python -m eval.evidence_gate --db data/engine.prod-20260730.db \\
        --user usr_01KTXAYR20J4R6F7PT3DP10W3W
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).parent

# Receipt → the label the user actually sees (mcp_server._evidence_receipt_markdown).
BACKS, REFUTES, RELATES, ORPHAN = "backs", "refutes", "relates", "orphan"


def _load_gold(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _observed(receipt: dict) -> str:
    """The verdict the evidence receipt renders, derived from the same fields it
    renders from — so this scores the shipped narration, not a parallel reimplementation."""
    if receipt.get("contradictions"):
        return REFUTES
    echoes = receipt.get("echoes", [])
    if any(e.get("stance") == "entailment" for e in echoes):
        return BACKS
    if echoes:
        return RELATES
    return ORPHAN


def run(db: str, user_id: str, gold_path: Path, keep: bool = False) -> dict:
    from core import config, store
    from core.encode import encode, get_embedder, stance_health

    work = Path(tempfile.mkdtemp(prefix="gate_a_")) / "corpus.db"
    shutil.copy(db, work)
    for suffix in ("-wal", "-shm"):
        side = Path(str(db) + suffix)
        if side.exists():
            shutil.copy(side, str(work) + suffix)

    health = stance_health()
    gold = _load_gold(gold_path)
    conn = store.connect(work)
    embedder = get_embedder()
    rows = []

    for g in gold:
        receipt = encode(conn, user_id, g["text"], source="research",
                         title=g["source_title"],
                         citation={"url": g["source_url"], "title": g["source_title"],
                                   "retrieved_at": "2026-07-30"})
        observed = _observed(receipt)

        # Best claim by cosine REGARDLESS of threshold — this is what separates a
        # stance error from a pair the retrieval layer never surfaced.
        best = {"similarity": 0.0, "text": None}
        embs = embedder.encode(_sentences(g["text"]),
                               normalize_embeddings=True, show_progress_bar=False)
        for emb in embs:
            for hit in store.knn_claims(conn, user_id, emb, k=1):
                if hit["similarity"] > best["similarity"]:
                    best = hit

        hit_target = (g.get("target_claim") is not None
                      and best.get("text") == g["target_claim"])
        reached = best["similarity"] >= config.ECHO_THRESHOLD
        rows.append({
            "id": g["id"], "field": g["field"], "expect": g["expect"],
            "observed": observed, "ok": observed == g["expect"],
            "best_sim": round(float(best["similarity"]), 3),
            "reached_threshold": reached,
            "matched_target": hit_target,
            "best_claim": (best.get("text") or "")[:70],
        })

    conn.close()
    if not keep:
        shutil.rmtree(work.parent, ignore_errors=True)

    near = [r for r in rows if r["field"] == "near"]
    far = [r for r in rows if r["field"] == "far"]
    # The dangerous error, called out on its own: a refutation rendered as support,
    # or support rendered as a refutation.
    inverted = [r for r in rows
                if {r["expect"], r["observed"]} == {BACKS, REFUTES}]
    return {
        "stance_provider": health,
        "rows": rows,
        "summary": {
            "near_correct": f"{sum(r['ok'] for r in near)}/{len(near)}",
            "far_correct": f"{sum(r['ok'] for r in far)}/{len(far)}",
            "far_below_threshold": sum(not r["reached_threshold"] for r in far),
            "near_below_threshold": sum(not r["reached_threshold"] for r in near),
            "inverted_verdicts": len(inverted),
            "backs_recall": _recall(rows, BACKS),
            "refutes_recall": _recall(rows, REFUTES),
        },
    }


def _sentences(text: str) -> list[str]:
    from core.encode import split_sentences
    return split_sentences(text) or [text[:500]]


def _recall(rows, label) -> str:
    want = [r for r in rows if r["expect"] == label]
    return f"{sum(r['observed'] == label for r in want)}/{len(want)}" if want else "n/a"


def _print(report: dict) -> None:
    h = report["stance_provider"]
    print(f"stance provider: {h['provider']} — {'OK' if h['ok'] else 'BROKEN'}: {h['detail']}\n")
    print(f"{'id':5s} {'fld':4s} {'expect':8s} {'observed':9s} {'sim':6s} {'τ':3s} {'tgt':4s} claim")
    for r in report["rows"]:
        mark = "✓" if r["ok"] else "✗"
        print(f"{r['id']:5s} {r['field']:4s} {r['expect']:8s} {r['observed']:9s} "
              f"{r['best_sim']:<6} {'≥' if r['reached_threshold'] else '<':3s} "
              f"{'yes' if r['matched_target'] else '-':4s} {mark} {r['best_claim']}")
    print()
    for k, v in report["summary"].items():
        print(f"  {k:22s} {v}")


def main():
    ap = argparse.ArgumentParser(prog="evidence_gate")
    ap.add_argument("--db", required=True, help="corpus to copy and test against")
    ap.add_argument("--user", required=True)
    ap.add_argument("--gold", default=str(HERE / "gold_evidence.jsonl"))
    ap.add_argument("--keep", action="store_true", help="keep the working copy")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = run(args.db, args.user, Path(args.gold), keep=args.keep)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print(report)


if __name__ == "__main__":
    main()
