"""Full eval of core/predict.py against the real corpus, in three parts:

  A. WRITE — all built cases (segment, match, label, centre) + the routing FIX.
  B. RETRIEVE — prototype the wrappers (decompose, route, assemble + STOP) on the
     same primitive (_residual_against) and test on real queries.
  C. CONSOLIDATE — prototype the reconstruction guard (safe forget / safe merge),
     dedupe, and split detection on the same primitive.

Run: .venv/bin/python tests/manual/predict_full_eval.py
"""
import sqlite3, sqlite_vec, numpy as np
from core import predict
from sentence_transformers import SentenceTransformer

# ── load real corpus ──────────────────────────────────────────────────────────
c = sqlite3.connect("data/engine.db"); c.row_factory = sqlite3.Row
c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False)
cl, lbl = {}, {}
for r in c.execute("SELECT claim_id, concept_id FROM concept_members"):
    cl.setdefault(r["claim_id"], r["concept_id"])
for r in c.execute("SELECT id,label FROM concepts"):
    lbl[r["id"]] = r["label"]
corpus = []
for r in c.execute("SELECT c.id,c.text,v.embedding FROM claims c JOIN vec_claims v ON v.claim_id=c.id"):
    corpus.append({"id": r["id"], "text": r["text"], "cluster": cl.get(r["id"]),
                   "embedding": np.frombuffer(r["embedding"], dtype=np.float32).astype(float)})
C = np.vstack([x["embedding"] for x in corpus])
_m = SentenceTransformer("all-MiniLM-L6-v2")
def embed(t): return np.atleast_2d(np.asarray(_m.encode(list(t), normalize_embeddings=True, show_progress_bar=False), dtype=float))
def B(s): print("\n"+"="*80+"\n"+s+"\n"+"="*80)
print(f"corpus {len(corpus)} claims / {len(set(v for v in cl.values()))} concepts")

# ════════════════════════════════════════════════════════════════════════════
# THE FIX — two parts.
#  (1) distribution-consistent baseline: score members with the SAME operator a
#      probe gets (global STAT_K leave-one-out), not within-cluster only.
#  (2) identity-gated routing: REINFORCE only on near-identity (dedup gate);
#      anything that sits ON an anchor but is not identical → AMBIGUOUS so the
#      resolver can sign it. (A polarity-blind encoder cannot tell a contradiction
#      from a paraphrase geometrically — so it must NOT be silently reinforced.)
# ════════════════════════════════════════════════════════════════════════════
# Part 1 (distribution-consistent global-LOO baseline) is now IN the sensor:
# compute_baselines routes members through predict._loo_residual. Part 2 (the
# spread-relative z_echo reinforce floor) is now IN decide(). So this harness
# calls the SHIPPED code paths — no inline prototype.
STAT_K, SPAN_K = predict.STAT_K, predict.SPAN_K
loo_residual = lambda i: predict._loo_residual(i, C)
base_new = predict.compute_baselines(corpus, C)
print(f"prior μ,σ (global-LOO, shipped)={tuple(round(x,3) for x in base_new['prior'])}  "
      f"z_echo={predict.Z_ECHO}")

def route_fixed(text, base):
    m = predict._measure_one(text, embed([text])[0], corpus, C, base)
    return predict.decide([m], None)["fragments"][0]["route"], m

# ── A. WRITE built cases ───────────────────────────────────────────────────────
B("A. WRITE — built cases")

print("\n[A1 MATCH] verbatim re-push of 568 real claims → should all reinforce")
from collections import Counter
fix = Counter()
for x in corpus:
    m = predict._measure_one(x["text"], x["embedding"], corpus, C, base_new)
    fix[predict.decide([m], None)["fragments"][0]["route"]] += 1
print("   shipped routing:", dict(fix), "(expect all PREDICTED)")

print("\n[A2 LABEL] paraphrase / contradiction / novel — shipped routing")
cases = [
 ("paraphrase",   "The equality premise of democracy is, in truth, completely false."),
 ("contradiction","Democracy's equality principle is actually fundamentally sound and just."),
 ("contradiction","Modern humans represent positively evolved, advanced beings."),
 ("contradiction","Spacetime is perfectly flat and unaffected by mass."),
 ("refine",       "Democracy's equality principle is false specifically in economic outcomes."),
 ("novel",        "My bicycle's rear derailleur needs adjusting before the weekend."),
]
for kind, t in cases:
    rf, m = route_fixed(t, base_new)
    print(f"   {kind:13} route={rf:9} sim={m['nearest_sim']:.3f} z={m['z']:+.2f} | {t[:42]!r}")

print("\n[A3 RESOLVER] direction over the AMBIGUOUS ones the fix now flags (local NLI)")
from core.encode import classify_stance
for kind, t in cases:
    rf, m = route_fixed(t, base_new)
    if rf == "AMBIGUOUS":
        d = predict.resolve_direction(t, m["anchor_text"], classify=classify_stance)
        print(f"   {kind:13} -> {d['direction']:11} (sign {d['sign']:+.0f}) anchor={m['anchor_text'][:40]!r}")

print("\n[A4 SEGMENT] cut a 3-topic note — current = sentence split only")
multi = ("Democracy's equality principle is fundamentally false. "
         "Spacetime curvature is what we perceive as gravity. "
         "My bicycle derailleur needs adjusting.")
for f in predict.prediction_error(multi, corpus, embed=embed, baselines=base_new)["fragments"]:
    print(f"   z={f['z']:+.2f} {f['route']:9} | {f['text'][:48]!r}")

print("\n[A5 CENTRE] note core (medoid) + most-novel fragment — NOT in predict.py; prototype")
note = ("Democracy claims everyone is equal. That equality premise is false. "
        "It shifted responsibility from the individual to society. "
        "The real foundation of society is the risk-taking system-creator.")
frags = predict._split(note); fe = embed(frags)
mean_sim = (fe @ fe.T).mean(axis=1)          # core = most central fragment
res = [predict._residual_against(fe[i], np.delete(fe,i,0)) for i in range(len(fe))]
print(f"   CORE (max centrality): {frags[int(mean_sim.argmax())][:55]!r}")
print(f"   MOST NOVEL (max residual): {frags[int(np.argmax(res))][:55]!r}")

# ── B. RETRIEVE prototype ──────────────────────────────────────────────────────
B("B. RETRIEVE — wrappers prototyped on _residual_against (assemble + STOP)")
def assemble(query, k_budget=6, relevance_floor=0.25, gain_floor=0.35):
    """Decompose query → pool relevant memory → greedily add the item that adds
    the most NEW info (max residual vs query+assembly) while staying relevant;
    STOP when marginal gain < gain_floor. Pure _residual_against."""
    qf = predict._split(query); qe = embed(qf)
    rel = (C @ qe.T).max(axis=1)                       # each item's best query-sim
    cand = [i for i in np.argsort(-rel) if rel[i] >= relevance_floor][:40]
    if not cand: return []
    chosen, ctx = [], list(qe)                          # assembly seeded with query
    log = []
    while cand and len(chosen) < k_budget:
        ctxM = np.vstack(ctx)
        gains = {i: predict._residual_against(C[i], ctxM) for i in cand}
        i = max(gains, key=lambda j: gains[j]*rel[j])    # info-per-relevance
        if gains[i] < gain_floor:
            log.append(("STOP", gains[i])); break
        chosen.append(i); ctx.append(C[i]); cand.remove(i)
        log.append((corpus[i]["text"][:50], round(gains[i],3), round(float(rel[i]),3)))
    return chosen, log
for q in ["Why is democracy flawed and who really builds society?",
          "What is gravity and how does spacetime relate to it?"]:
    chosen, log = assemble(q)
    print(f"\n   Q: {q}")
    for row in log:
        if row[0]=="STOP": print(f"      ⏹ STOP (next gain {row[1]:.2f} < floor)")
        else: print(f"      + gain={row[1]:.2f} rel={row[2]:.2f} | {row[0]!r}")
    print(f"      assembled {len(chosen)} items (budget 6) — stopped on marginal gain")

# ── C. CONSOLIDATE prototype ────────────────────────────────────────────────────
B("C. CONSOLIDATE — reconstruction guard (safe forget/merge), dedupe, split")
def reconstruct_z(i, base):
    """How well the REST of memory rebuilds item i, as z vs region spread.
    Low z (well reconstructed) = safe to forget/merge; high z = irreplaceable."""
    r = loo_residual(i)
    k = corpus[i]["cluster"]
    mu, sd = base["clusters"].get(k, base["prior"])
    return (r - mu)/sd, r

print("\n[C1 SAFE-FORGET] lowest- vs highest-reconstruction-z claims")
zs = [(reconstruct_z(i, base_new)[0], i) for i in range(len(corpus)) if corpus[i]["cluster"]]
zs.sort()
print("   SAFE to forget (rebuildable from rest, z≪0):")
for z,i in zs[:3]: print(f"      z={z:+.2f} | {corpus[i]['text'][:55]!r}")
print("   PROTECT (irreplaceable, z≫0):")
for z,i in zs[-3:]: print(f"      z={z:+.2f} | {corpus[i]['text'][:55]!r}")

print("\n[C2 DEDUPE] near-identical claim pairs (sim ≥ 0.90)")
S = C @ C.T; np.fill_diagonal(S, 0)
seen=set(); n=0
for i in range(len(corpus)):
    j = int(S[i].argmax())
    if S[i,j] >= 0.90 and (j,i) not in seen:
        seen.add((i,j)); n+=1
        if n<=4: print(f"      sim={S[i,j]:.3f} | {corpus[i]['text'][:34]!r} ≈ {corpus[j]['text'][:34]!r}")
print(f"      total candidate-dup pairs ≥0.90: {n}")

print("\n[C3 SPLIT] clusters with widest internal spread (split candidates)")
spread = sorted(base_new["clusters"].items(), key=lambda kv:-kv[1][1])[:3]
for k,(mu,sd) in spread:
    print(f"      σ={sd:.3f} μ={mu:.3f} | {lbl.get(k,k)!r}")
