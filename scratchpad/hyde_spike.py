"""HyDE spike — does a hypothetical-answer embedding pull the 5 unreachable answer-
notes into the top-40 seed? Generate a hypothetical answer (LLM), embed its
sentences, union with the raw-query seed, re-rank. Compare answer-note ranks."""
import os
os.environ.setdefault("LLM_FALLBACK_ORDER", "claude,local")
os.environ.setdefault("LLM_MAX_ATTEMPTS", "6")

import numpy as np
from core import store, retrieve, llm
from core.encode import get_embedder, split_sentences

UID = "usr_01KTXAYR20J4R6F7PT3DP10W3W"
conn = store.connect("/tmp/slate_eval.db")

PROBES = {  # query -> the notes that were beyond SEED_K=40 in r0_spike
    "What is my critique of capitalism and crony capitalism?": ["Cronyism, Inequality"],
    "Summarise my views on AI and memory systems.": ["Debugging as the real test"],
    "How has my thinking on religion evolved over the years?": ["Hinduism as a system"],
    "What are the various product or startup ideas I have explored?":
        ["web3 architecture", "Memory as the Next AI"],
}

HYDE_SYS = ("You are the author of a personal notebook. Given a question, write a "
            "short hypothetical passage (3-5 sentences) in first person that the "
            "answer might contain — the kind of specific points and vocabulary you'd "
            "have written. Do not hedge; just write the plausible note content.")


def ranks_for(emb_list, needles):
    """Union the seeds of each embedding (max sim per fragment), rank, find needle ranks."""
    merged = {}
    for emb in emb_list:
        for c in store.fragment_candidates(conn, UID, emb, k=120):
            cur = merged.get(c["id"])
            if cur is None or c["similarity"] > cur["similarity"]:
                merged[c["id"]] = c
    ranked = sorted(merged.values(), key=lambda c: -c["similarity"])
    titles = [(c.get("title") or "") for c in ranked]
    out = {}
    for nd in needles:
        r = next((i for i, t in enumerate(titles) if nd.lower() in t.lower()), None)
        out[nd] = r
    return out


emb = get_embedder()
print(f"{'query':9} {'note':28} {'baseline':>9} {'HyDE':>6}")
for q, needles in PROBES.items():
    q_emb = retrieve._embed_query(q)
    base = ranks_for([q_emb], needles)
    # HyDE: generate hypothetical answer, embed its sentences + the query
    res = llm.call(f"Question: {q}\n\nHypothetical note passage:", tier="judgment",
                   max_tokens=256, system=HYDE_SYS, json_out=False)
    hyp = res["text"].strip()
    sents = split_sentences(hyp, min_chars=1) or [hyp]
    hyp_embs = list(emb.encode(sents, normalize_embeddings=True, show_progress_bar=False))
    hyde = ranks_for([q_emb] + hyp_embs, needles)
    for nd in needles:
        b = base[nd]; h = hyde[nd]
        bs = str(b) if b is not None else "MISS"
        hs = str(h) if h is not None else "MISS"
        win = " ✓ now in top-40" if (h is not None and h < 40 and (b is None or b >= 40)) else ""
        print(f"{q[:9]:9} {nd:28} {bs:>9} {hs:>6}{win}")
