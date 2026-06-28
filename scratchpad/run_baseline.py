"""Clean SR@B baseline over the frozen gold set, via the (now-funded) Claude API.

Runs the 4 answerers × {B, B/2} over all gold queries on a COPY of engine.db
(fragments materialized). Resumable: each (system,budget,query) verdict is cached
to a JSON and skipped on re-run, so a mid-run failure costs nothing to resume.
Replaces the noisy CLI-judged numbers in docs/retrieve-workflow-eval.md.

Usage: .venv/bin/python scratchpad/run_baseline.py [--db PATH] [--systems a,b]
"""
import argparse
import json
import os
import time
from pathlib import Path

# Eval pacing/robustness: stay on the (funded) Claude API, ride out RPM windows,
# and never cascade to the exhausted gemini free tier. Set BEFORE importing core.
os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")
os.environ.setdefault("LLM_BACKOFF_BASE", "2.0")

from core import store, llm
import eval.harness as H

UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
DB = "/tmp/slate_eval.db"
CACHE = Path("/tmp/slate_baseline_results.json")
BUDGETS = [2000, 1000]


def _meter():
    orig = llm.call
    t = {"calls": 0, "cost": 0.0, "providers": {}}

    def metered(*a, **k):
        r = orig(*a, **k)
        t["calls"] += 1
        t["cost"] += r.get("cost", 0.0) or 0.0
        t["providers"][r["provider"]] = t["providers"].get(r["provider"], 0) + 1
        return r
    llm.call = metered
    H.llm.call = metered
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB)
    ap.add_argument("--systems", default="slate,grep,frag,hybrid")
    args = ap.parse_args()

    systems = args.systems.split(",")
    conn = store.connect(args.db)
    gold = H.load_gold("eval/gold.jsonl")
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    meter = _meter()
    t0 = time.time()

    for system in systems:
        fn = H._ANSWERERS[system]
        for b in BUDGETS:
            for g in gold:
                key = f"{system}|{b}|{g['id']}"
                if key in cache:
                    continue
                a = fn(conn, UID, g["query"], b)
                v = H.judge(g["query"], a["answer"], g["key_facts"])
                cache[key] = {"pass": bool(v["pass"]), "n_have": v.get("n_have"),
                              "n_facts": len(g["key_facts"]),
                              "ctx_tok": a.get("context_tokens") or a.get("ctx_tokens"),
                              "hard": bool(g.get("hard"))}
                CACHE.write_text(json.dumps(cache, indent=1))  # flush each verdict
                done = meter["calls"] // 2
                print(f"[{done:>3}] {key:32} pass={int(bool(v['pass']))} "
                      f"running=${meter['cost']:.4f}  (${meter['cost']/max(1,done):.4f}/cell)",
                      flush=True)
                time.sleep(1.0)  # pace under the API RPM/ITPM tier limit

    # report matrix from cache
    print(f"\nSR@B baseline  (n_gold={len(gold)})   cost=${meter['cost']:.3f}  "
          f"calls={meter['calls']}  providers={meter['providers']}  {time.time()-t0:.0f}s\n")
    print(f"{'system':8} {'B':>6} {'SR@B':>7} {'tail':>7} {'meanctx':>8}  n")
    for system in systems:
        for b in BUDGETS:
            rows = [v for k, v in cache.items() if k.startswith(f"{system}|{b}|")]
            if not rows:
                continue
            n = len(rows)
            sr = sum(r["pass"] for r in rows) / n
            hard = [r for r in rows if r["hard"]]
            tail = (sum(r["pass"] for r in hard) / len(hard)) if hard else None
            mc = sum(r.get("ctx_tok") or 0 for r in rows) / n
            tail_s = f"{tail:6.0%}" if tail is not None else "    — "
            print(f"{system:8} {b:>6} {sr:6.0%} {tail_s:>7} {mc:8.0f}  {n}")


if __name__ == "__main__":
    main()
