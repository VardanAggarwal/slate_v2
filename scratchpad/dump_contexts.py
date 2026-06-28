"""Retrieve-only: assemble the resonance context for each gold query at a given
res_deaniso_k and dump {id, query, key_facts, hard, context} to JSONL. NO LLM calls
(no answerer, no judge) — the agent layer judges the dumped contexts. $0, no provider.

  python scratchpad/dump_contexts.py --gold eval/gold_broad.jsonl --k 1 \
      --user usr_01KTXAYR20J4R6F7PT3DP10W3W --out scratchpad/ctx_broad_k1.jsonl
"""
import argparse, json
from pathlib import Path
from core import store, calibration as calib, retrieve, resonance
from eval import harness

ap = argparse.ArgumentParser()
ap.add_argument("--gold", required=True)
ap.add_argument("--user", required=True)
ap.add_argument("--k", default="off")
ap.add_argument("--budget", type=int, default=2000)
ap.add_argument("--out", required=True)
args = ap.parse_args()

k = None if args.k == "off" else int(args.k)
conn = store.connect()
gold = harness.load_gold(args.gold)

rows, tot = [], 0
for g in gold:
    cal = calib.merged(conn, {**retrieve.DEFAULT_CALIBRATION,
                              **resonance.DEFAULT_CALIBRATION,
                              "res_deaniso_k": k}, args.user)
    ctx = resonance.resonance_context(conn, args.user, g["query"],
                                      max_chars=args.budget * harness.CHARS_PER_TOKEN,
                                      calibration=cal)
    rows.append({"id": g.get("id", g["query"][:40]), "query": g["query"],
                 "key_facts": g["key_facts"], "hard": bool(g.get("hard")),
                 "context_tokens": len(ctx) // harness.CHARS_PER_TOKEN, "context": ctx})
    tot += len(ctx) // harness.CHARS_PER_TOKEN

Path(args.out).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
print(json.dumps({"gold": Path(args.gold).stem, "k": args.k, "n": len(rows),
                  "mean_ctx_tok": round(tot / max(len(rows), 1), 1),
                  "out": args.out}))
