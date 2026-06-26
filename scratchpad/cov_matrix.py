"""Coverage@B matrix — all systems × all gold sets, one process (model loads once)."""
import json
from pathlib import Path
from core import store
from eval.coverage import coverage_eval, _load_gold, CONTEXT_FNS, HERE

DB = "/tmp/slate_eval.db"
U = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
TAU = 0.54
SYS = ["grep", "slate", "frag", "hybrid", "hier", "resonance"]
GOLD = {"narrow": "gold.jsonl", "broad": "gold_broad.jsonl", "paragraph": "gold_paragraph.jsonl"}

conn = store.connect(DB)
results = {}
for gname, gfile in GOLD.items():
    gold = _load_gold(HERE / gfile)
    for sys in SYS:
        rep = coverage_eval(gold, conn, U, CONTEXT_FNS[sys], TAU)
        tail = rep["sr_tail"]
        results[(gname, sys)] = (rep["sr_at_b"], tail, rep["mean_context_tokens"])
        print(f"{gname:10} {sys:10} SR@B={rep['sr_at_b']:.1%} "
              f"tail={('%.0f%%'%(100*tail)) if tail is not None else '—':>4} "
              f"ctx={rep['mean_context_tokens']:.0f}t", flush=True)

print("\n=== TABLE (SR@B / tail) τ=0.54 ===")
print(f"{'system':10} {'narrow':>14} {'broad':>8} {'paragraph':>10}")
for sys in SYS:
    def cell(g):
        if (g, sys) not in results: return "—"
        sr, tl, _ = results[(g, sys)]
        return f"{sr:.0%}" + (f"/{tl:.0%}" if tl is not None else "")
    print(f"{sys:10} {cell('narrow'):>14} {cell('broad'):>8} {cell('paragraph'):>10}")
