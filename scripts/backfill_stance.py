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

TWO STAGES, AND WHY
-------------------
Stage 1 (hf zero-shot) is a RECALL filter ONLY. Its single entailment score
cannot separate neutral from contradiction — measured on real fragment pairs it
fired on elaborations, proposed fixes, topic shifts and raw metadata blobs. It
exists here just to cheaply shortlist the "not entailed" rows.

Stage 2 (OpenRouter free-tier Nemotron) does a genuine 3-class read over that
shortlist, and only rows BOTH stages call a contradiction get written. The
fragment path needs this because — unlike the receipt path, which is gated at
similarity >= ECHO_THRESHOLD — `write.py` resolves direction against the merely
NEAREST memory row with no similarity gate, so a low entailment score there
often means "unrelated", not "opposed".

USAGE
    python scripts/backfill_stance.py                  # stage 1 only, dry run
    python scripts/backfill_stance.py --adjudicate     # both stages, dry run
    python scripts/backfill_stance.py --adjudicate --apply
    python scripts/backfill_stance.py --revert <run>   # undo a run from the audit table
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

from core import config, encode, store  # noqa: E402

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
           a.text AS anchor_text, a.episode_id AS anchor_episode_id,
           e.ts, e.title, ae.ts AS anchor_ts
      FROM fragments f
      JOIN fragments a  ON a.id  = f.anchor_id
      JOIN episodes  e  ON e.id  = f.episode_id
      JOIN episodes  ae ON ae.id = a.episode_id
     WHERE f.anchor_id IS NOT NULL
       AND f.direction = 'refine'
       AND f.id NOT IN (SELECT fragment_id FROM stance_backfill)
     ORDER BY e.ts, f.id
"""


def classify_strict(premise: str, hypothesis: str) -> str:
    """Stance that RAISES on provider failure instead of degrading to "neutral".

    encode.classify_stance() deliberately swallows failures so a save is never
    blocked. In a bulk offline run that is exactly wrong: a 503-exhausted call
    comes back "neutral", reads as "not a contradiction", and is silently dropped
    — while the failure counter stays at 0 because nothing raised. Two runs over
    identical data disagreed 13 vs 48 on the shortlist for this reason. Offline,
    a failure must be a failure.
    """
    if config.STANCE_PROVIDER == "hf":
        return encode._get_hf_stance().classify(premise, hypothesis)
    if config.STANCE_PROVIDER == "nli":
        scores = encode._get_nli().predict([(premise, hypothesis)])[0]
        return ["contradiction", "entailment", "neutral"][int(scores.argmax())]
    return encode.classify_stance(premise, hypothesis)


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
            stance = classify_strict(r["anchor_text"], r["text"])
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
            "anchor_episode_id": r["anchor_episode_id"], "anchor_ts": r["anchor_ts"],
            # An anchor inside the SAME note is usually rhetorical structure (an
            # essay arguing both sides), not a change of mind. Recorded, not
            # filtered — stage 2 still adjudicates it — so the split is visible.
            "intra_note": r["episode_id"] == r["anchor_episode_id"],
        })
        if verbose and i % 50 == 0:
            print(f"  {i}/{len(rows)} classified", file=sys.stderr, flush=True)
        if sleep:
            time.sleep(sleep)
    return results, failures


ADJUDICATE_PROMPT = """You are labelling the logical relation between two excerpts \
from one person's personal notes, written at different times.

EARLIER NOTE (premise):
{anchor}

LATER NOTE (hypothesis):
{fragment}

Label the LATER note's relation to the EARLIER one:
- "contradiction" — the later note asserts something that cannot both be true \
alongside the earlier one. A genuine reversal of position, a refuted claim, an \
incompatible fact. NOT merely a different topic.
- "entailment" — the later note restates or follows from the earlier one.
- "neutral" — anything else: a different subject, an elaboration or added detail, \
a proposed solution to a problem the earlier note raised, a narrower or broader \
case, or unrelated metadata/boilerplate.

Be strict. Most pairs are "neutral". Only call it a contradiction if you could \
point to the specific pair of incompatible assertions.

Return ONLY JSON: {{"stance": "contradiction"|"entailment"|"neutral", "why": "<12 words max>"}}"""


def adjudicate(rows, verbose=True):
    """Stage 2: real 3-class read over stage-1's shortlist, via OpenRouter only.

    LLM_FALLBACK_ORDER is pinned to openrouter for the duration: the default
    order falls through to `claude`, and a bulk run silently billing the
    Anthropic API is a documented way to drain the account. If OpenRouter is
    down the row is recorded as a failure and retried on a later run instead.
    """
    from core import llm

    saved_order = config.LLM_FALLBACK_ORDER
    config.LLM_FALLBACK_ORDER = ["openrouter"]
    kept, rejected, failures = [], [], []
    try:
        for i, r in enumerate(rows, 1):
            try:
                res = llm.call(
                    ADJUDICATE_PROMPT.format(anchor=r["anchor_text"], fragment=r["text"]),
                    tier="mechanical", max_tokens=2048)
                stance = (res["json"] or {}).get("stance", "neutral")
                why = (res["json"] or {}).get("why", "")
            except Exception as e:                # noqa: BLE001 — retried on a later run
                failures.append({"id": r["id"], "error": f"{type(e).__name__}: {e}"})
                continue
            r = {**r, "llm_stance": stance, "llm_why": why}
            (kept if stance == "contradiction" else rejected).append(r)
            if verbose and i % 20 == 0:
                print(f"  adjudicated {i}/{len(rows)} — {len(kept)} confirmed",
                      file=sys.stderr, flush=True)
    finally:
        config.LLM_FALLBACK_ORDER = saved_order
    return kept, rejected, failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    ap.add_argument("--adjudicate", action="store_true",
                    help="stage 2: LLM 3-class read over stage-1's shortlist (required for --apply)")
    ap.add_argument("--limit", type=int, help="only process the first N candidates")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between API calls")
    ap.add_argument("--report", default="stance_backfill_report.json")
    ap.add_argument("--revert", metavar="RUN_ID", help="undo a previous --apply run")
    args = ap.parse_args()

    if args.apply and not args.adjudicate:
        sys.exit("--apply requires --adjudicate: stage 1 alone has poor precision on real "
                 "fragment pairs (it fires on elaborations and metadata), so writing from it "
                 "would invent contradictions. See the module docstring.")

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
    print(f"{len(rows)} candidate fragments — stage 1 (hf shortlist)", file=sys.stderr)
    results, failures = classify_all(rows, sleep=args.sleep)
    counts = Counter(r["stance"] for r in results)

    # SCOPE: only contradictions are repaired. 'refine' vs 'reinforce' both map to
    # sign +1.0 and strength 1.0 in resolve_direction, so relabelling those would
    # rewrite ~40% of the corpus for no functional change — and stage 1's
    # entailment calls on real pairs are just as unvalidated as its contradiction
    # calls. 'refine' stays the conservative default for every non-contradiction.
    shortlist = [r for r in results if r["stance"] == "contradiction"]
    print(f"stage 1 shortlisted {len(shortlist)} of {len(results)} classified",
          file=sys.stderr)

    confirmed, rejected, llm_failures = [], [], []
    if args.adjudicate and shortlist:
        print(f"stage 2 (LLM 3-class) over {len(shortlist)} — OpenRouter only, "
              "~20 req/min paced", file=sys.stderr)
        confirmed, rejected, llm_failures = adjudicate(shortlist)

    changed = [r for r in confirmed if r["new_direction"] != r["old_direction"]]

    report = {
        "stage1_provider": config.STANCE_PROVIDER,
        "candidates": len(rows), "classified": len(results),
        "stage1_failed": len(failures), "stage1_counts": dict(counts),
        "stage1_shortlist": len(shortlist),
        "adjudicated": bool(args.adjudicate),
        "stage2_confirmed": len(confirmed), "stage2_rejected": len(rejected),
        "stage2_failed": len(llm_failures),
        "stage1_precision": (round(len(confirmed) / len(shortlist), 3)
                             if args.adjudicate and shortlist else None),
        "confirmed_intra_note": sum(1 for r in confirmed if r["intra_note"]),
        "confirmed_cross_note": sum(1 for r in confirmed if not r["intra_note"]),
        "candidate_intra_note": sum(1 for r in results if r["intra_note"]),
        "to_write": len(changed),
        "confirmed_rows": confirmed,
        "rejected_rows": rejected,
        "failures": failures + llm_failures,
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
                      if k not in ("confirmed_rows", "rejected_rows", "failures")}, indent=2))
    for r in confirmed[:25]:
        tag = "intra-note" if r["intra_note"] else f"anchor {r['anchor_ts'][:10]}"
        print(f"\n  [{r['ts'][:10]}] {(r['title'] or '')[:60]}  ({tag})"
              f"\n  ANCHOR: {r['anchor_text'][:90]}"
              f"\n  FRAG  : {r['text'][:90]}"
              f"\n  WHY   : {r['llm_why']}")
    print(f"\nfull report -> {args.report}")


if __name__ == "__main__":
    main()
