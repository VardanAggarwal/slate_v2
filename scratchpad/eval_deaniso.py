"""One de-anisotropy eval arm: resonance @ a given res_deaniso_k, over one gold set.
Reuses the real harness (same answerer, same judge) so arms are directly comparable.

  python scratchpad/eval_deaniso.py --gold eval/gold_broad.jsonl --k 1 --budget 2000 \
      --user usr_01KTXAYR20J4R6F7PT3DP10W3W --out scratchpad/deaniso_broad_k1.json

Prints a compact summary; writes the full per-query JSON to --out (for compare/forgetting).
"""
import argparse, json
from pathlib import Path
from core import store, calibration as calib, retrieve, resonance
from eval import harness


def make_fn(k):
    label = "off" if k is None else f"k{k}"

    def fn(conn, user_id, query, budget_tok):
        cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION,
                                  **resonance.DEFAULT_CALIBRATION,
                                  "res_deaniso_k": k}, user_id)
        ctx = resonance.resonance_context(conn, user_id, query,
                                          max_chars=budget_tok * harness.CHARS_PER_TOKEN,
                                          calibration=cal)
        return {"system": f"res_{label}", **harness._answer_from_context(query, ctx)}
    fn.__name__ = f"res_{label}_answer"
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--k", default="off", help="off | 0 | 1 | 3 …")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    k = None if args.k == "off" else int(args.k)
    conn = store.connect()
    gold = harness.load_gold(args.gold)
    rep = harness.run_eval(gold, conn, args.user, answer_fn=make_fn(k),
                           budget_tok=args.budget)
    Path(args.out).write_text(json.dumps(rep, indent=2))

    passes = [p["id"] for p in rep["per_query"] if p["pass"]]
    fails = [p["id"] for p in rep["per_query"] if not p["pass"]]
    print(json.dumps({
        "gold": Path(args.gold).stem, "k": args.k, "budget": args.budget,
        "sr_at_b": rep["sr_at_b"], "sr_tail": rep["sr_tail"],
        "n": rep["n"], "n_hard": rep["n_hard"],
        "mean_ctx": rep["mean_context_tokens"], "cost": rep["total_cost"],
        "passes": passes, "fails": fails,
    }, indent=2))


if __name__ == "__main__":
    main()
