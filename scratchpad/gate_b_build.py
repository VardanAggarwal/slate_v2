"""Gate B step 3 (docs/evidence-lane-plan.md §7): build an evidence-BEARING corpus on a
COPY of the prod backup, so evidence actually competes for the retrieval budget.

Never touches the source DB. Steps: encode sources as source='research' → refine
(local, free) → consolidate (LLM) → sweep. Then run eval/coverage against the result.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "engine.prod-20260730.db"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["encode", "consolidate", "sweep", "all"])
    args = ap.parse_args()

    from core import config, evidence, store, write
    from core.consolidate import consolidate
    from core.encode import encode

    out = Path(args.out)
    if args.stage in ("encode", "all"):
        if out.exists():
            out.unlink()
        for s in ("-wal", "-shm"):
            Path(str(out) + s).unlink(missing_ok=True)
        shutil.copy(args.src, out)
        print(f"copied {args.src} → {out}")

    conn = store.connect(out)
    uid = args.user
    print(f"EVIDENCE_LANE={config.EVIDENCE_LANE} STANCE={config.STANCE_PROVIDER} "
          f"LLM_ORDER={config.LLM_FALLBACK_ORDER}")

    if args.stage in ("encode", "all"):
        sources = (load(ROOT / "eval" / "gold_evidence.jsonl")
                   + load(ROOT / "eval" / "evidence_corpus.jsonl"))
        for s in sources:
            r = encode(conn, uid, s["text"], source="research",
                       title=s["source_title"],
                       citation={"url": s["source_url"], "title": s["source_title"],
                                 "retrieved_at": "2026-07-30"})
            write.refine_episode(conn, uid, r["episode_id"])   # local, no LLM
        print(f"encoded + refined {len(sources)} research episodes")

    if args.stage in ("consolidate", "all"):
        n = 0
        while True:
            rep = consolidate(conn, uid, max_episodes=50)
            print(json.dumps({k: v for k, v in rep.items()
                              if k in ("status", "episodes", "skipped", "cost",
                                       "evidence")}, default=str)[:400])
            n += 1
            if rep["status"] == "noop" or n > 6:
                break

    if args.stage in ("sweep", "all"):
        with conn:
            print("sweep:", json.dumps(evidence.sweep(conn, uid)))

    print("\n--- corpus ---")
    for q, label in (("SELECT COUNT(*) FROM episodes WHERE user_id=? AND source='research'", "research episodes"),
                     ("SELECT COUNT(*) FROM claims WHERE user_id=?", "claims"),
                     ("SELECT COUNT(*) FROM concept_members WHERE user_id=? AND kind='evidence'", "evidence members"),
                     ("SELECT COUNT(*) FROM concept_members WHERE user_id=? AND kind='primary'", "primary members"),
                     ("SELECT COUNT(*) FROM evidence_attachments WHERE user_id=?", "attachments")):
        print(f"  {label:20s} {conn.execute(q, (uid,)).fetchone()[0]}")
    for r in conn.execute("SELECT stance, COUNT(*) n FROM evidence_attachments "
                          "WHERE user_id=? GROUP BY stance", (uid,)):
        print(f"    stance {r[0]:15s} {r[1]}")
    conn.close()


if __name__ == "__main__":
    main()
