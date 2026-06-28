"""Assemble (no judge, no answerer) the 5 disagreement queries through both paths.
Shows WHAT each path surfaced at B=2000 — diagnostic for concept-vs-fragment wins."""
import os
os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")

from core import store
from core.recall import assemble_context as concept_ctx
from core.retrieve import assemble_context as frag_ctx

UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
DB = "/tmp/slate_eval.db"
B = 2000 * 4  # chars

# (id, winner, query)
QS = [
    ("g02", "frag",    "What's my take on mercenaries vs missionaries in product?"),
    ("g05", "frag",    "Why do I argue modern employment is a form of slavery?"),
    ("g08", "frag",    "Why do I think debugging is the real test of AI coding?"),
    ("g04", "concept", "How do I think Seed Savers Club actually adds value?"),
    ("g11", "concept", "Why have I grown from atheist to anti-theist about religion?"),
]

conn = store.connect(DB)
for gid, winner, q in QS:
    print("=" * 90)
    print(f"{gid}  [winner: {winner}]  {q}")
    for name, fn in [("CONCEPT", concept_ctx), ("FRAGMENT", frag_ctx)]:
        try:
            ctx = fn(conn, UID, q, max_chars=B)
        except Exception as e:
            ctx = f"<ERROR: {type(e).__name__}: {e}>"
        print(f"\n--- {name} ({len(ctx)} chars) ---")
        print(ctx[:1600])
        if len(ctx) > 1600:
            print(f"... [+{len(ctx)-1600} more chars]")
    print()
