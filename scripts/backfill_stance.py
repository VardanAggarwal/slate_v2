"""Re-derive fragment stance for the window when STANCE_PROVIDER was silently broken.

WHY THIS EXISTS
---------------
Prod ran with no STANCE_PROVIDER in .env, so it defaulted to 'nli', _get_nli()
threw on the torch-less image, and classify_stance() degraded to "neutral" for
every call. Two durable consequences, both in `fragments`:

    direction : every AMBIGUOUS fragment got 'refine'  (never 'contradict')
    strength  : every one got 1.0  (never WRITE_CONTRADICT_HOLD)

Fixed in 1148c17 + STANCE_PROVIDER=hf on the server. This script repairs the
rows written before that.

WHAT IT DOES NOT TOUCH
----------------------
* `episodes.receipt_json` — the immutable record of what the user was actually
  shown at save time. They genuinely were not shown those contradictions;
  rewriting it would falsify history. Recovered contradictions are reported
  instead, for a review pass.
* Anything a DELETE would touch. Stance is a pure function of two stored,
  immutable texts (anchor.text, fragment.text), so this is a re-derivation, not
  a reconstruction from lossy state. Prior values are copied into an
  insert-only `stance_backfill` audit table before any UPDATE, so every write
  is reversible from data this script itself records.

USAGE
    python scripts/backfill_stance.py                  # dry run, writes a report
    python scripts/backfill_stance.py --apply          # audit + update
    python scripts/backfill_stance.py --revert <run>   # undo a run from the audit table
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter

from core import config, encode, store

AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS stance_backfill (
    run_id       TEXT NOT NULL,
    fragment_id  TEXT NOT NULL,
    old_direction TEXT,
    old_strength  REAL,
    new_direction TEXT,
    new_strength  REAL,
    entail_prob   REAL,
    applied_at    TEXT NOT NULL,
    PRIMARY KEY (run_id, fragment_id)
)
"""

# Only rows the broken path could have written. NOVEL fragments have no anchor
# and never called resolve_direction, so they are correctly NULL — excluded.
SELECT_CANDIDATES = """
    SELECT f.id, f.text, f.direction, f.strength, f.user_id, f.episode_id,
           a.text AS anchor_text, e.ts, e.title
      FROM fragments f
      JOIN fragments a ON a.id = f.anchor_id
      JOIN episodes  e ON e.id = f.episode_id
     WHERE f.anchor_id IS NOT NULL
       AND f.direction = 'refine'
       AND f.id NOT IN (SELECT fragment_id FROM stance_backfill)
     ORDER BY e.ts, f.id
"""


def _candidates(conn, limit=None):
    try:
        rows = conn.execute(SELECT_CANDIDATES).fetchall()
    except sqlite3.OperationalError:          # audit table not created yet
        conn.execute(AUDIT_DDL)
        rows = conn.execute(SELECT_CANDIDATES).fetchall()
    return rows[:limit] if limit else rows


def classify_all(rows, sleep=0.0, verbose=True):
    """Re-derive stance for each row. Returns (results, failures).

    A failure is NOT silently coerced to neutral — that is the exact bug being
    repaired. Failed rows are left out so a later run retries them.
    """
    results, failures = [], []
    for i, r in enumerate(rows, 1):
        try:
            stance = encode.classify_stance(r["anchor_text"], r["text"])
        except Exception as e:                # noqa: BLE001 — recorded, then retried later
            failures.append({"id": r["id"], "error": f"{type(e).__name__}: {e}"})
            continue
        # mirror predict.resolve_direction's mapping
        if stance == "contradiction":
            direction, strength = "contradict", config.WRITE_CONTRADICT_HOLD
        elif stance == "entailment":
            direction, strength = "reinforce", 1.0
        else:
            direction, strength = "refine", 1.0
        results.append({
            "id": r["id"], "user_id": r["user_id"], "episode_id": r["episode_id"],
            "ts": r["ts"], "title": r["title"], "stance": stance,
            "old_direction": r["direction"], "old_strength": r["strength"],
            "new_direction": direction, "new_strength": strength,
            "anchor_text": r["anchor_text"], "text": r["text"],
        })
        if verbose and i % 50 == 0:
            print(f"  {i}/{len(rows)} classified", file=sys.stderr, flush=True)
        if sleep:
            time.sleep(sleep)
    return results, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--limit", type=int, help="only process the first N candidates")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls")
    ap.add_argument("--report", default="stance_backfill_report.json")
    ap.add_argument("--revert", metavar="RUN_ID", help="undo a previous --apply run")
    args = ap.parse_args()

    conn = store.connect(config.DB_PATH) if hasattr(store, "connect") else sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row

    if args.revert:
        conn.execute(AUDIT_DDL)
        rows = conn.execute(
            "SELECT fragment_id, old_direction, old_strength FROM stance_backfill WHERE run_id=?",
            (args.revert,)).fetchall()
        for r in rows:
            conn.execute("UPDATE fragments SET direction=?, strength=? WHERE id=?",
                         (r["old_direction"], r["old_strength"], r["fragment_id"]))
        conn.commit()
        print(f"reverted {len(rows)} fragments from run {args.revert}")
        return

    if config.STANCE_PROVIDER not in ("nli", "hf", "haiku"):
        sys.exit(f"STANCE_PROVIDER={config.STANCE_PROVIDER!r} — refusing to backfill with a "
                 "disabled classifier (that is what caused this).")

    # Fail loudly if the provider is broken, instead of writing 1,500 'refine's again.
    probe = encode.classify_stance("I love working in the office.",
                                   "I hate working in the office.")
    if probe != "contradiction":
        sys.exit(f"provider {config.STANCE_PROVIDER!r} self-check FAILED "
                 f"(returned {probe!r} for an obvious contradiction). Fix the provider first.")
    print(f"provider {config.STANCE_PROVIDER!r} self-check ok", file=sys.stderr)

    rows = _candidates(conn, args.limit)
    print(f"{len(rows)} candidate fragments", file=sys.stderr)
    results, failures = classify_all(rows, sleep=args.sleep)

    counts = Counter(r["stance"] for r in results)
    changed = [r for r in results if r["new_direction"] != r["old_direction"]]
    contradictions = [r for r in results if r["new_direction"] == "contradict"]

    report = {
        "provider": config.STANCE_PROVIDER,
        "candidates": len(rows), "classified": len(results), "failed": len(failures),
        "stance_counts": dict(counts), "changed": len(changed),
        "contradictions": len(contradictions),
        "contradiction_rows": contradictions,
        "failures": failures,
        "applied": bool(args.apply),
    }

    if args.apply:
        run_id = f"run_{int(time.time())}"
        conn.execute(AUDIT_DDL)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for r in changed:
            conn.execute(
                "INSERT OR IGNORE INTO stance_backfill (run_id, fragment_id, old_direction,"
                " old_strength, new_direction, new_strength, entail_prob, applied_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (run_id, r["id"], r["old_direction"], r["old_strength"],
                 r["new_direction"], r["new_strength"], None, now))
            conn.execute("UPDATE fragments SET direction=?, strength=? WHERE id=?",
                         (r["new_direction"], r["new_strength"], r["id"]))
        conn.commit()
        report["run_id"] = run_id
        print(f"applied {len(changed)} updates as {run_id} "
              f"(revert: --revert {run_id})", file=sys.stderr)

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("contradiction_rows", "failures")}, indent=2))
    print(f"\nfull report -> {args.report}")


if __name__ == "__main__":
    main()
