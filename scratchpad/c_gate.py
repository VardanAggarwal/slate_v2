"""C-gate (P5 EXIT) — small-batch version.

Measures the consolidation safety core end-to-end on the frozen gold set:
  before  = SR@B over gold (slate answerer = claims/concepts path)
  rollback the most-recent run -> frees its N episodes (re-derivable from raw)
  consolidate those N from raw -> a new run (blueprint/merge/reconcile/prune/bg)
  after   = SR@B over the SAME gold
  compare -> ΔSR@B, catastrophic-forgetting (passed-before-now-fail)
  reversibility = rollback the new run; assert state returns to pre-run.

EXIT (PRD §3 stage gate C): ΔSR@B >= 0 AND forgetting_rate == 0 AND run reversible.

Operates on a COPY (/tmp/slate_eval.db); live engine.db untouched. Backup taken
by the caller. One-shot (the DB mutation isn't resumable) — on failure, restore
the backup and re-run.
"""
import os
import time

os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")
os.environ.setdefault("LLM_BACKOFF_BASE", "2.0")

from core import store, llm, consolidate as C
import eval.harness as H

DB = "/tmp/slate_eval.db"
UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
BUDGET = 2000
SYSTEM = "slate"          # claims/concepts path — the layer consolidation restructures
ROLLBACK_RUN = "run_01KTXJDNW0EM2Z2TDNMYR9SWK4"   # most-recent ok run, 15 episodes
BATCH = 15


def meter():
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
    C.llm.call = metered
    return t


def counts(conn):
    return {t: conn.execute(f"SELECT count(*) FROM {t} WHERE user_id=?", (UID,)).fetchone()[0]
            for t in ("claims", "concepts", "relations")}


def sr(conn, gold, fn, tag):
    rep = H.run_eval(gold, conn, UID, answer_fn=fn, budget_tok=BUDGET)
    print(f"  [{tag}] SR@B={rep['sr_at_b']:.1%}  tail="
          f"{('%.0f%%' % (100*rep['sr_tail'])) if rep['sr_tail'] is not None else '—'}"
          f"  meanctx={rep['mean_context_tokens']:.0f}tok  cost=${rep['total_cost']:.4f}", flush=True)
    return rep


def main():
    t0 = time.time()
    m = meter()
    conn = store.connect(DB)
    gold = H.load_gold("eval/gold.jsonl")
    fn = H._ANSWERERS[SYSTEM]
    print(f"C-GATE small-batch | system={SYSTEM} B={BUDGET} gold={len(gold)} "
          f"batch={BATCH}ep run={ROLLBACK_RUN}\n", flush=True)

    c0 = counts(conn)
    print("state S0:", c0, flush=True)

    # ── before ──────────────────────────────────────────────────────────────
    print("\n[1/5] before SR@B …", flush=True)
    before = sr(conn, gold, fn, "before")

    # ── roll back the recent run -> its episodes become pending (S_minus) ─────
    print(f"\n[2/5] rolling back {ROLLBACK_RUN} (frees {BATCH} episodes) …", flush=True)
    rb = C.rollback_run(conn, ROLLBACK_RUN)
    c_minus = counts(conn)
    pending = len(store.unconsolidated_episodes(conn, UID))
    print(f"  freed={rb['episodes_freed']} events_applied={rb['events_applied']} "
          f"pending={pending}  state S_minus: {c_minus}", flush=True)

    # ── re-consolidate the freed batch FROM RAW -> new run (S1) ───────────────
    print(f"\n[3/5] consolidate(max_episodes={BATCH}) — re-derive from raw …", flush=True)
    rep = C.consolidate(conn, UID, max_episodes=BATCH)
    new_run = rep.get("run_id")
    print(f"  {rep}", flush=True)
    c1 = counts(conn)
    print(f"  state S1: {c1}", flush=True)

    # ── after ─────────────────────────────────────────────────────────────────
    print("\n[4/5] after SR@B …", flush=True)
    after = sr(conn, gold, fn, "after")

    cmp = H.compare(before, after)
    print(f"\n  ΔSR@B = {cmp['delta_sr_at_b']:+.1%}   forgetting_rate = "
          f"{cmp['forgetting_rate']:.1%}", flush=True)
    print(f"  regressions (passed→fail): {cmp['regressions'] or 'none'}", flush=True)
    print(f"  gains (fail→pass):         {cmp['gains'] or 'none'}", flush=True)

    # ── reversibility: roll back the NEW run -> should return to S_minus ──────
    print(f"\n[5/5] reversibility: rollback {new_run} …", flush=True)
    rev_ok = None
    if new_run:
        C.rollback_run(conn, new_run)
        c_rev = counts(conn)
        rev_ok = (c_rev == c_minus)
        print(f"  state after rollback: {c_rev}   == S_minus? {rev_ok}", flush=True)
    else:
        print("  no new run (noop) — skipped", flush=True)

    # ── verdict ──────────────────────────────────────────────────────────────
    g1 = cmp["delta_sr_at_b"] >= 0
    g2 = cmp["forgetting_rate"] == 0.0
    g3 = bool(rev_ok)
    print(f"\n{'='*60}\nC-GATE VERDICT (small batch)")
    print(f"  ΔSR@B >= 0           : {'PASS' if g1 else 'FAIL'} ({cmp['delta_sr_at_b']:+.1%})")
    print(f"  forgetting == 0      : {'PASS' if g2 else 'FAIL'} ({cmp['forgetting_rate']:.1%})")
    print(f"  run reversible       : {'PASS' if g3 else 'FAIL'}")
    print(f"  -> {'GATE PASS' if (g1 and g2 and g3) else 'GATE FAIL'}")
    print(f"{'='*60}")
    print(f"\ncost=${m['cost']:.4f}  calls={m['calls']}  providers={m['providers']}  "
          f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
