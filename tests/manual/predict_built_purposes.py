"""Ad-hoc test: can core/predict.py fulfill the built + wrapper purposes in PRD v2 §How?

Built (claimed validated on the live corpus): Write-side measure()/decide() —
  segment, match, label disposition, centre.
Wrappers (claimed designed, not yet measured): sequential scan (segmentation /
  query decomposition), iterative residual-against-the-assembly loop (retrieval
  depth + stopping), reconstruction guard (safe merge / safe forget).

We load the REAL local corpus (568 claims, 63 concepts) and probe each.
"""
import sqlite3, sqlite_vec, numpy as np
from core import predict

DB = "data/engine.db"

def load_corpus():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
    # claim_id -> concept_id (cluster). A claim may belong to >1 concept; take first.
    cl = {}
    for r in c.execute("SELECT claim_id, concept_id FROM concept_members"):
        cl.setdefault(r["claim_id"], r["concept_id"])
    corpus = []
    for r in c.execute("SELECT c.id, c.text, v.embedding FROM claims c "
                       "JOIN vec_claims v ON v.claim_id=c.id"):
        emb = np.frombuffer(r["embedding"], dtype=np.float32).astype(float)
        corpus.append({"id": r["id"], "text": r["text"], "embedding": emb,
                       "cluster": cl.get(r["id"])})
    c.close()
    return corpus

def embed(texts):
    from sentence_transformers import SentenceTransformer
    global _m
    try: _m
    except NameError: _m = SentenceTransformer("all-MiniLM-L6-v2")
    a = _m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    return np.atleast_2d(np.asarray(a, dtype=float))

def banner(s): print("\n" + "="*78 + "\n" + s + "\n" + "="*78)

corpus = load_corpus()
n_clustered = sum(1 for c in corpus if c["cluster"])
print(f"corpus: {len(corpus)} claims, {n_clustered} clustered, "
      f"{len(set(c['cluster'] for c in corpus if c['cluster']))} clusters")
base = predict.compute_baselines(corpus)
print("prior (mu,sd):", tuple(round(x,3) for x in base["prior"]),
      "| n_clusters w/ cohesion:", len(base["clusters"]))

# ─────────────────────────────────────────────────────────────────────────────
banner("BUILT #1  MATCH + LABEL: re-pushing an EXACT existing claim → PREDICTED?")
# Take a few real claims verbatim; pushing them again should reinforce, not store.
samples = [c for c in corpus if c["cluster"]][:5]
for s in samples:
    r = predict.prediction_error(s["text"], corpus, embed=embed, baselines=base)
    f = r["fragments"][0]
    print(f"  z={f['z']:+.2f} near_sim={f['nearest_sim']:.3f} route={f['route']:9} "
          f"| {s['text'][:55]!r}")
print("  EXPECT: route PREDICTED (z<=0.5). An exact dup that stores is a miss.")

banner("BUILT #2  NOVEL: content from a totally unrelated domain → NOVEL?")
novel_texts = [
    "The recipe calls for two cups of basmati rice and a pinch of saffron.",
    "My bicycle's rear derailleur needs adjusting before the weekend ride.",
    "Quarterly SaaS churn dropped to 2.1% after the onboarding redesign.",
]
for t in novel_texts:
    r = predict.prediction_error(t, corpus, embed=embed, baselines=base)
    f = r["fragments"][0]
    print(f"  z={f['z']:+.2f} near_sim={f['nearest_sim']:.3f} route={f['route']:9} "
          f"| {t[:55]!r}")
print("  EXPECT: NOVEL (high z, no anchor present).")

banner("BUILT #3  AMBIGUOUS: a paraphrase/refinement of an existing claim")
# Hand-built refinements + contradictions of real claims, to see if they land
# AMBIGUOUS (store + resolve) rather than silently PREDICTED or NOVEL.
pairs = [
    ("Democracy's equality principle is fundamentally false",
     "The equality premise of democracy is, in truth, completely false."),     # paraphrase
    ("Democracy's equality principle is fundamentally false",
     "Democracy's equality principle is actually fundamentally sound."),       # contradiction
    ("Modern humans represent negatively evolved beings",
     "Contemporary humans are, in evolutionary terms, a regression."),         # paraphrase
]
for anchor, probe in pairs:
    r = predict.prediction_error(probe, corpus, embed=embed, baselines=base)
    f = r["fragments"][0]
    print(f"  z={f['z']:+.2f} near_sim={f['nearest_sim']:.3f} route={f['route']:9} "
          f"resolve={f['resolve']} | {probe[:50]!r}")
print("  EXPECT: AMBIGUOUS for close paraphrase/contradiction (store + resolve).")

banner("BUILT #4  SEGMENT: does measure() actually cut a multi-claim note?")
multi = ("Democracy's equality principle is fundamentally false. "
         "Also, spacetime curvature is what we perceive as gravity. "
         "And my bicycle derailleur needs adjusting.")
r = predict.prediction_error(multi, corpus, embed=embed, baselines=base)
print(f"  fragments returned: {len(r['fragments'])} (3 sentences in)")
for f in r["fragments"]:
    print(f"    z={f['z']:+.2f} route={f['route']:9} | {f['text'][:50]!r}")
print("  NOTE: 'segmentation' here = sentence split only. PRD asks for boundaries")
print("  'where content stops being predictable' + resolution that FOLLOWS surprise.")

# ─────────────────────────────────────────────────────────────────────────────
banner("WRAPPER CHECK: are the three claimed wrappers present in predict.py?")
import inspect
names = [n for n,_ in inspect.getmembers(predict, inspect.isfunction)]
print("  public/module functions:", [n for n in names if not n.startswith("__")])
expected = {
    "segment/decompose (sequential scan)": ["segment", "decompose", "scan"],
    "retrieval residual-against-assembly loop + stop": ["assemble", "retrieve", "stop", "route_query", "borrow"],
    "reconstruction guard (safe merge/forget)": ["reconstruct", "guard", "safe_merge", "safe_forget", "can_reconstruct"],
}
for purpose, needles in expected.items():
    hit = [n for n in names if any(k in n.lower() for k in needles)]
    print(f"  [{'PRESENT' if hit else 'MISSING'}] {purpose}: {hit or '—'}")

banner("WRAPPER #1  RETRIEVE — residual-of-memory-against-query (the asymmetry test)")
# PRD §How flags retrieval as 'least proven': does a question embed near the
# claim that answers it? Use measure() with X=query, Y=corpus and inspect anchor.
queries = [
    "Why does the author think democracy is flawed?",
    "What is gravity, according to these notes?",
]
for q in queries:
    r = predict.measure(q, corpus, embed=embed, baselines=base)
    f = r[0]
    print(f"  q={q!r}")
    print(f"    near_sim={f['nearest_sim']:.3f} anchor={f['anchor_text'][:60]!r}")
print("  EXPECT (if retrieval were viable): anchor is the ANSWER claim, high sim.")
