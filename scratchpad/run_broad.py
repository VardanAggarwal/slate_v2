"""SR@B baseline on the BROAD probe gold (synthesis-across-notes regime).
4 answerers × {B=2000} over eval/gold_broad.jsonl. Resumable per-cell cache."""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")
os.environ.setdefault("LLM_BACKOFF_BASE", "2.0")

from core import store, llm
import eval.harness as H

UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
DB = os.environ.get("DB_PATH", "/tmp/slate_eval.db")
CACHE = Path("/tmp/slate_broad_results.json")
GOLD = "eval/gold_broad.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="grep,slate,frag,hybrid")
    ap.add_argument("--budgets", default="2000")
    args = ap.parse_args()

    systems = args.systems.split(",")
    budgets = [int(b) for b in args.budgets.split(",")]
    conn = store.connect(DB)
    gold = H.load_gold(GOLD)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    orig = llm.call
    meter = {"calls": 0, "cost": 0.0}
    def metered(*a, **k):
        r = orig(*a, **k); meter["calls"] += 1; meter["cost"] += r.get("cost", 0.0) or 0.0
        return r
    llm.call = metered; H.llm.call = metered
    t0 = time.time()

    for system in systems:
        fn = H._ANSWERERS[system]
        for b in budgets:
            for g in gold:
                key = f"{system}|{b}|{g['id']}"
                if key in cache:
                    continue
                a = fn(conn, UID, g["query"], b)
                v = H.judge(g["query"], a["answer"], g["key_facts"])
                cache[key] = {"pass": bool(v["pass"]), "covered": v["covered"],
                              "n_facts": len(g["key_facts"]),
                              "ctx_tok": a.get("context_tokens"),
                              "answer": a["answer"][:400], "hard": bool(g.get("hard"))}
                CACHE.write_text(json.dumps(cache, indent=1))
                print(f"[{meter['calls']//2:>3}] {key:24} pass={int(bool(v['pass']))} "
                      f"({sum(v['covered'])}/{len(g['key_facts'])}) ${meter['cost']:.4f}", flush=True)
                time.sleep(1.0)

    print(f"\nBROAD SR@B  cost=${meter['cost']:.3f}  {time.time()-t0:.0f}s\n")
    print(f"{'system':8}{'B':>6}{'SR@B':>8}{'meanctx':>9}  n")
    for system in systems:
        for b in budgets:
            rows = [v for k, v in cache.items() if k.startswith(f"{system}|{b}|")]
            if not rows: continue
            n = len(rows); sr = sum(r["pass"] for r in rows)/n
            mc = sum((r.get("ctx_tok") or 0) for r in rows)/n
            print(f"{system:8}{b:>6}{sr:>7.0%}{mc:>9.0f}  {n}")


if __name__ == "__main__":
    main()
