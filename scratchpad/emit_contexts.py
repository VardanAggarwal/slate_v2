"""Emit retrieval CONTEXTS (no LLM) for baseline vs treatment, for in-session judging.

Retrieval is fully local — only the answerer+judge need an LLM. So this builds the
assembled context for each broad-gold query under both conditions and dumps them to JSON;
the Opus-4.8 session then answers from each context and judges against key_facts (the
harness protocol), serving as the sonnet-or-better judge the gate needs.

Run: PYTHONPATH=.:scratchpad .venv/bin/python scratchpad/emit_contexts.py [gold_file]
"""
import json
import sys

from core import resonance
from eval.coverage import _load_gold, HERE
from query_bridges import U, fresh_conn, build_qstream
from query_claims import inject_query_claims

GOLD = sys.argv[1] if len(sys.argv) > 1 else "gold_broad.jsonl"
OUT = HERE.parent / "scratchpad" / f"contexts_{GOLD.replace('.jsonl','')}.json"


def main():
    gold = _load_gold(HERE / GOLD)

    base_conn, _ = fresh_conn("emit_base")
    tr_conn, _ = fresh_conn("emit_treat")
    queries = build_qstream(tr_conn)
    inject_query_claims(tr_conn, queries)   # query_claims.RECOMPUTE controls re-anchor

    rows = []
    for g in gold:
        rows.append({
            "id": g.get("id", g["query"][:40]),
            "query": g["query"],
            "key_facts": g["key_facts"],
            "context_base": resonance.resonance_context(base_conn, U, g["query"]),
            "context_treat": resonance.resonance_context(tr_conn, U, g["query"]),
        })
    OUT.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} query contexts to {OUT}")


if __name__ == "__main__":
    main()
