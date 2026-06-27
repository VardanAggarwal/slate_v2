"""Does the query-claim gain survive a REAL LLM answerer+judge (not the cosine proxy)?
And does the leaked question text change the LLM's answer?

Runs eval/harness.run_eval (LLM answer -> LLM judge, binary all-facts-covered) with the
resonance answerer, on the BROAD gold (the prize), for three DBs:
  baseline | treatment(leaky) | treatment(text-suppressed)
All LLM via the local Claude subscription (query_bridges sets claude-cli).

Run:  PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/query_claims_llm.py
"""
from __future__ import annotations

from pathlib import Path

from core import config, store, resonance
# claude-cli session limit hit; queries are cached so we no longer need Claude. Run the
# LLM answerer+judge on gemini (free, directional — weaker judge than sonnet).
config.LLM_FALLBACK_ORDER = ["gemini"]
from eval import harness
from eval.coverage import _load_gold, HERE
from query_bridges import U, build_qstream, fresh_conn
from query_claims import inject_query_claims, patch_no_query_text

import sys
_GOLD_FILE = sys.argv[1] if len(sys.argv) > 1 else "gold_broad.jsonl"
GOLD_BROAD = HERE / _GOLD_FILE


def run(conn, tag, gold):
    rep = harness.run_eval(gold, conn, U, answer_fn=harness.resonance_answer,
                           budget_tok=harness.DEFAULT_BUDGET_TOKENS)
    print(f"  {tag:24} SR@B={rep['sr_at_b']:.3f}  tail={rep['sr_tail']}  "
          f"cost=${rep['total_cost']:.3f}")
    return rep


def main():
    gold = _load_gold(GOLD_BROAD)
    print(f"{_GOLD_FILE}: {len(gold)} queries — real LLM answerer + judge\n")

    base_conn, _ = fresh_conn("llm_base")
    rep_base = run(base_conn, "baseline", gold)

    tr_conn, _ = fresh_conn("llm_treat")
    queries = build_qstream(tr_conn)
    inject_query_claims(tr_conn, queries)
    rep_leak = run(tr_conn, "treatment (leaky)", gold)

    patch_no_query_text()                 # suppress query-claim text in the frame
    rep_supp = run(tr_conn, "treatment (suppressed)", gold)

    print("\n— compare (baseline → suppressed) —")
    cmp = harness.compare(rep_base, rep_supp)
    print(f"  ΔSR@B={cmp['delta_sr_at_b']:+.3f}  forgetting={cmp['forgetting_rate']:.3f}")
    print(f"  gains={cmp['gains']}\n  regressions={cmp['regressions']}")
    print("\n— leaky vs suppressed (does hiding the text change the LLM answer?) —")
    cmp2 = harness.compare(rep_leak, rep_supp)
    print(f"  ΔSR@B={cmp2['delta_sr_at_b']:+.3f}  "
          f"flips={sorted(set(cmp2['gains']) | set(cmp2['regressions']))}")


if __name__ == "__main__":
    main()
