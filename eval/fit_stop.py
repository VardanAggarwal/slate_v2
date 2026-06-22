"""C12 fit — relevance-aware STOP (`value_floor`) vs frozen SR@B (plan §3 C12, §2 R2/R7).

Sweeps the assembly `value_floor` knob over the frozen gold set and reports
SR@B / tail / mean-ctx per floor, so the over-injection fix (assembly pads to
budget, hurting SR@B@B/2 — docs/retrieve-workflow-eval.md) is **fitted against the
North Star, not pre-tuned**. `value_floor=None` is the current raw-residual
baseline, always swept as the control.

The picked floor is the one that maximises frozen SR@B, tie-broken by tail then by
*lower* mean context (parsimony — the PRD credits the assembly size that maximises
the answer, not the one that fills B). It is then written into the retrieve
calibration profile by the caller; this script only MEASURES.

QUOTA — the judge+answerer is LLM-gated (CLI session limit ~8 calls/window; see
memory `slate-prod-llm-account-capped`). The driver is **RESUMABLE**: every
(floor, budget, query) result is cached to `--cache` and flushed immediately, so a
session-limit error stops cleanly and a rerun next window continues. Put `claude-cli`
first in `LLM_FALLBACK_ORDER` (stored-session auth) to bypass the env-token gate.

Usage:
    .venv/bin/python -m eval.fit_stop --user <uid> --half \
        --floors none,0.10,0.15,0.20,0.25,0.30 --cache eval/fit_stop_cache.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

# Stay on the funded Claude API, ride out RPM windows, never cascade to the
# exhausted gemini free tier. Must run BEFORE `core` imports load_dotenv.
os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")
os.environ.setdefault("LLM_BACKOFF_BASE", "2.0")

from core import assembly, config, store
from core.retrieve import MAX_ITEMS, assemble_context as fragment_context
from eval.harness import (CHARS_PER_TOKEN, DEFAULT_BUDGET_TOKENS,
                          _answer_from_context, judge, load_gold)


def _floor_key(f: float | None) -> str:
    return "none" if f is None else f"{f:g}"


def _calibration(value_floor: float | None) -> dict:
    return {"gain_floor": assembly.GAIN_FLOOR, "max_items": MAX_ITEMS,
            "value_floor": value_floor, "per_cluster": {}}


def _load_cache(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _score_query(conn, user_id: str, g: dict, value_floor: float | None,
                 budget_tok: int) -> dict:
    """One (floor, budget, query) cell: assemble with the floor, answer, judge."""
    ctx = fragment_context(conn, user_id, g["query"],
                           max_chars=budget_tok * CHARS_PER_TOKEN,
                           calibration=_calibration(value_floor))
    a = _answer_from_context(g["query"], ctx)
    j = judge(g["query"], a["answer"], g["key_facts"])
    return {"id": g.get("id", g["query"][:40]), "hard": bool(g.get("hard")),
            "pass": j["pass"], "covered": j["covered"], "answer": a["answer"],
            "context_tokens": a["context_tokens"], "notes": j["notes"],
            "cost": round(a["cost"] + j["cost"], 6)}


def sweep(gold: list[dict], conn, user_id: str, *, floors: list[float | None],
          budgets: list[int], cache_path: Path) -> list[dict]:
    """Resumable sweep. Returns one aggregate row per (floor, budget)."""
    cache = _load_cache(cache_path)
    rows: list[dict] = []
    for floor in floors:
        for b in budgets:
            per = []
            for g in gold:
                qid = g.get("id", g["query"][:40])
                ck = f"{_floor_key(floor)}|{b}|{qid}"
                if ck not in cache:
                    cache[ck] = _score_query(conn, user_id, g, floor, b)
                    cache_path.write_text(json.dumps(cache, indent=2))  # flush now
                    time.sleep(1.0)  # pace under the API RPM/ITPM tier limit
                per.append(cache[ck])
            passes = [p["pass"] for p in per]
            hard = [p["pass"] for p in per if p["hard"]]
            rows.append({
                "value_floor": floor, "budget_tokens": b, "n": len(per),
                "sr_at_b": round(statistics.fmean(passes), 4) if passes else 0.0,
                "n_hard": len(hard),
                "sr_tail": round(statistics.fmean(hard), 4) if hard else None,
                "mean_context_tokens": round(statistics.fmean(
                    [p["context_tokens"] for p in per]), 1) if per else 0,
                "total_cost": round(sum(p["cost"] for p in per), 4),
            })
    return rows


def pick_best(rows: list[dict], budget_tok: int) -> dict | None:
    """Best floor AT the primary budget: max SR@B, tie → higher tail, then lower
    mean ctx (parsimony). The frozen-set ΔSR@B is the only credit (plan risk #2)."""
    cells = [r for r in rows if r["budget_tokens"] == budget_tok]
    if not cells:
        return None
    return max(cells, key=lambda r: (r["sr_at_b"], r["sr_tail"] or 0.0,
                                     -r["mean_context_tokens"]))


def format_table(rows: list[dict]) -> str:
    out = ["value_floor sweep — SR@B per floor (None = raw-residual baseline)",
           f"  {'floor':>7}  {'B':>5}  {'SR@B':>6}  {'tail':>6}  {'ctx':>6}  {'cost':>7}"]
    for r in rows:
        tail = f"{r['sr_tail']:.0%}" if r["sr_tail"] is not None else "—"
        out.append(f"  {_floor_key(r['value_floor']):>7}  {r['budget_tokens']:>5}  "
                   f"{r['sr_at_b']:>6.1%}  {tail:>6}  "
                   f"{r['mean_context_tokens']:>6.0f}  ${r['total_cost']:>6.4f}")
    return "\n".join(out)


def _parse_floors(s: str) -> list[float | None]:
    out: list[float | None] = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        out.append(None if tok in ("none", "off", "") else float(tok))
    return out


def _main() -> None:
    ap = argparse.ArgumentParser(description="Fit the relevance-aware STOP (value_floor) vs SR@B")
    ap.add_argument("--gold", default=str(Path(__file__).parent / "gold.jsonl"))
    ap.add_argument("--user", help="user_id (corpus owner)")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS)
    ap.add_argument("--half", action="store_true", help="also sweep at B/2 (where over-injection bites)")
    ap.add_argument("--floors", default="none,0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--cache", default=str(Path(__file__).parent / "fit_stop_cache.json"),
                    help="resumable per-(floor,budget,query) cache")
    ap.add_argument("--out", help="write the aggregate sweep JSON here")
    ap.add_argument("--push", action="store_true",
                    help="C12 push-down: persist the best value_floor into the user's "
                         "calibration profile (retrieve then loads it by default)")
    args = ap.parse_args()

    conn = store.connect()
    user_id = args.user or config.DEFAULT_USER_ID
    gold = load_gold(args.gold)
    budgets = [args.budget, args.budget // 2] if args.half else [args.budget]
    floors = _parse_floors(args.floors)

    rows = sweep(gold, conn, user_id, floors=floors, budgets=budgets,
                 cache_path=Path(args.cache))
    print(format_table(rows))
    best = pick_best(rows, args.budget)
    if best:
        print(f"\nbest @ B={args.budget}: value_floor={_floor_key(best['value_floor'])}"
              f"  SR@B={best['sr_at_b']:.1%}  (baseline = floor 'none' row above)")
        if args.push:
            from core import calibration as calib
            with conn:
                prof = calib.push(conn, user_id, value_floor=best["value_floor"])
            print(f"pushed → calibration_profiles[{user_id}] = {prof}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "best": best}, indent=2))


if __name__ == "__main__":
    _main()
