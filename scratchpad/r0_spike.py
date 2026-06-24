"""R0 spike — query->answer-note rank on the broad gold. Measures the symmetric-
encoder asymmetry directly: for each query, where do its fact-bearing notes rank in
the fragment knn? If they sit beyond SEED_K (40), seeding is the ceiling. No LLM."""
import numpy as np
from core import store, retrieve

UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
conn = store.connect("/tmp/slate_eval.db")

# query -> substrings of the episode TITLES that carry the key facts (from authoring)
ANSWER_NOTES = {
    "Summarise my professional work experience across the companies and ventures I have been part of.":
        ["Seed savers club", "Thrive -", "Almabase -", "My role at goSTOPS"],
    "How has my thinking on religion evolved over the years?":
        ["Religion", "Why am I an anti-theist", "Evolution of Religious Rituals",
         "Hinduism as a system"],
    "What are the various product or startup ideas I have explored?":
        ["Seed savers", "gifting industry", "web3 architecture", "Memory as the Next AI"],
    "Summarise my views on AI and memory systems.":
        ["Memory as the Next AI", "RAG Limitations", "Debugging as the real test",
         "Hierarchical Context"],
    "What is my critique of capitalism and crony capitalism?":
        ["Capitalism, or not", "Cronyism, Inequality", "Crony Capital as Foreign"],
    "How do I connect employment, gig work, and slavery as forms of coercion?":
        ["thin line between slavery", "Economic Coercion of Gig"],
}


def ranked_titles(emb):
    """All fragments ranked by similarity to emb; return list of (title) in order."""
    cand = store.fragment_candidates(conn, UID, emb, k=300)
    cand.sort(key=lambda c: -c["similarity"])
    return [(c.get("title") or "") for c in cand], [c["similarity"] for c in cand]


def best_rank(titles, needle):
    for i, t in enumerate(titles):
        if needle.lower() in t.lower():
            return i
    return None


print(f"{'query':10} {'answer note':32} {'rank':>6} {'sim':>6}")
worst = []
for q, needles in ANSWER_NOTES.items():
    emb = retrieve._embed_query(q)
    titles, sims = ranked_titles(emb)
    qid = q[:8]
    for nd in needles:
        r = best_rank(titles, nd)
        rs = f"{r}" if r is not None else "MISS"
        ss = f"{sims[r]:.3f}" if r is not None else "—"
        flag = "  <-- beyond SEED_K=40" if (r is None or r >= 40) else ""
        print(f"{qid:10} {nd:32} {rs:>6} {ss:>6}{flag}")
        worst.append(r if r is not None else 9999)
    print()

beyond = sum(1 for r in worst if r >= 40)
print(f"answer-notes total={len(worst)}  beyond SEED_K(40)={beyond}  "
      f"median rank={int(np.median([r for r in worst if r < 9999]))}")
