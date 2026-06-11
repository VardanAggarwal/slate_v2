"""Replay driver: read old slate.db sources.raw_text chronologically, encode() each as an episode preserving original created_at. See PLAN.md §2.2, §7 Phase 1.

Idempotent: each old source id is recorded in replay_map in the same
transaction as its episode, so re-running skips everything already replayed.
"""
import argparse
import sqlite3

from core import config, store
from core.encode import encode


def replay(old_db: str | None = None, limit: int | None = None,
           db_path: str | None = None, verbose: bool = True) -> dict:
    conn = store.connect(db_path)
    old = sqlite3.connect(old_db or config.SLATE_V1_DB)
    old.row_factory = sqlite3.Row

    total = old.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    rows = old.execute(
        """SELECT id, title, created_at, raw_text FROM sources
           WHERE raw_text IS NOT NULL AND TRIM(raw_text) != ''
           ORDER BY created_at ASC"""
    ).fetchall()
    if limit:
        rows = rows[:limit]

    done = skipped = 0
    for row in rows:
        if store.replay_seen(conn, row["id"]):
            skipped += 1
            continue
        receipt = encode(conn, row["raw_text"], ts=row["created_at"],
                         title=row["title"], source="replay", replay_key=row["id"])
        done += 1
        if verbose:
            n_prior = len(receipt["prior_episode_matches"])
            print(f"[replay] {done:3d} {row['created_at'][:10]} "
                  f"{(row['title'] or '')[:48]:48s} "
                  f"sents={receipt['n_sentences']:3d} prior_matches={n_prior}")

    summary = {"replayed": done, "skipped": skipped,
               "old_sources_total": total, "old_sources_nonempty": len(rows),
               "episodes": store.count_episodes(conn)}
    if verbose:
        print(f"[replay] done: {summary}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay old slate.db into the new engine")
    ap.add_argument("--old-db", default=None, help="path to v1 slate.db")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    replay(old_db=args.old_db, limit=args.limit)
