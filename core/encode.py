"""Awake-time encoding: embed locally, kNN vs canonical claims, novelty receipt (echoes/novelties/contradictions), episode write. No LLM graph decisions. See PLAN.md §5.

Synchronous and cheap (<1s once the embedder is warm). The only optional model
call is local NLI stance classification for contradiction detection.

Before the first consolidation the claims table is empty, so the receipt also
reports prior-episode sentence matches ("echoes your March note on X") — claims
remain the canonical layer once consolidate() has run.
"""
import logging
import re
from datetime import datetime, timezone

from core import config, store

log = logging.getLogger("slate.encode")

# ── Embedder singleton (ported from v1 engine/db.py) ──────────────────────────
# Production embeds via the HF Inference API (HF_TOKEN set) — no torch on the
# server. Local SentenceTransformer is the dev/test path. Both produce
# identical 384-dim L2-normalized all-MiniLM-L6-v2 vectors.
_embedder = None


class HFEmbedder:
    """Thin wrapper around HF Inference API feature-extraction (v1's HFEmbedModel)."""

    def __init__(self, token: str, model: str | None = None):
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(api_key=token)
        self._model = model or f"sentence-transformers/{config.EMBED_MODEL_NAME}"

    def encode(self, sentences, **kwargs):  # accepts/ignores ST kwargs
        import numpy as np
        single = isinstance(sentences, str)
        inputs = [sentences] if single else list(sentences)
        result = self._client.feature_extraction(inputs, model=self._model,
                                                 normalize=True)
        arr = np.array(result)
        if arr.ndim == 3:  # token-level embeddings returned — mean-pool
            arr = arr.mean(axis=1)
        return arr[0] if single else arr


def get_embedder():
    global _embedder
    if _embedder is None:
        if config.HF_TOKEN:
            _embedder = HFEmbedder(config.HF_TOKEN)
        else:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _embedder


# ── Stance classifier (PLAN.md §9.1: local NLI first, one Haiku call fallback) ─
# The W6 resolver answers ONE bit at write: contradiction or not. The 'nli'
# CrossEncoder needs torch — absent on the 1GB HF-only prod host, where it throws
# and silently returns "neutral", turning every contradiction into a refine. The
# 'hf' provider mirrors HFEmbedder: same HF_TOKEN, server-side MNLI via
# InferenceClient.zero_shot_classification — no torch, so contradictions survive.
_nli = None
_hf_stance = None


def _get_nli():
    global _nli
    if _nli is None:
        from sentence_transformers import CrossEncoder
        _nli = CrossEncoder(config.NLI_MODEL)
    return _nli


STANCE_HF_URL = "https://router.huggingface.co/hf-inference/models/{model}"


class HFStance:
    """Zero-shot MNLI over the HF Inference API (no torch). P(premise ⊨ hypothesis)
    is read with the hypothesis as the single candidate label and a pass-through
    template, then bucketed high→entail / low→contradict / mid→neutral.

    Deliberately posts to the router directly instead of using the typed
    InferenceClient.zero_shot_classification helper: huggingface_hub >=1.x
    validates the response against a list-shaped schema, but the router returns
    a bare object for some models (a dict with parallel labels/scores) and a
    list for others. The mismatch raises *inside* the client, which
    classify_stance() then swallows into a silent "neutral" — i.e. the helper
    turns every contradiction into an echo with no error surfaced.

    LIMITATION (known): a single entailment score separates entail from not-entail,
    but "not entailed" spans BOTH neutral and contradiction — this collapsed read
    cannot distinguish them. STANCE_CONTRADICT_MAX is therefore set LOW so only a
    confident non-entailment is called a contradiction (favouring false-neutrals
    over false-contradictions, since a missed contradiction is reconciled later at
    consolidation but a false one is over-held). A faithful 3-class read (P_contra,
    P_neutral, P_entail) for the 'hf' provider — matching the local 'nli' path — is
    a follow-up; the 'nli' and 'haiku' providers already read all three."""

    def __init__(self, token: str, model: str | None = None):
        self._token = token
        self._model = model or config.STANCE_HF_MODEL

    def _post(self, premise: str, hypothesis: str):
        import requests
        res = requests.post(
            STANCE_HF_URL.format(model=self._model),
            headers={"Authorization": f"Bearer {self._token}"},
            json={"inputs": premise,
                  "parameters": {"candidate_labels": [hypothesis],
                                 "multi_label": True,
                                 "hypothesis_template": "{}"}},
            timeout=config.STANCE_HF_TIMEOUT)
        res.raise_for_status()
        return res.json()

    def _entail_prob(self, premise: str, hypothesis: str) -> float:
        res = self._post(premise, hypothesis)
        # Two live response shapes, both seen on the router:
        #   {"sequence":…, "labels":[…], "scores":[…]}   (DeBERTa MNLI)
        #   [{"label":…, "score":…}]                     (bart-large-mnli)
        if isinstance(res, dict):
            if "error" in res:
                raise RuntimeError(f"HF stance error: {res['error']}")
            return float(res["scores"][0] if "scores" in res else res["score"])
        return float(res[0]["score"])

    def classify(self, premise: str, hypothesis: str) -> str:
        p = self._entail_prob(premise, hypothesis)
        if p >= config.STANCE_ENTAIL_MIN:
            return "entailment"
        if p <= config.STANCE_CONTRADICT_MAX:
            return "contradiction"
        return "neutral"


def _get_hf_stance():
    global _hf_stance
    if _hf_stance is None:
        _hf_stance = HFStance(config.HF_TOKEN)
    return _hf_stance


def classify_stance(premise: str, hypothesis: str) -> str:
    """Return 'contradiction' | 'entailment' | 'neutral' per STANCE_PROVIDER."""
    # Every failure below degrades to "neutral" so a save is never blocked — but
    # it MUST be logged: a silent degrade is indistinguishable from "no
    # contradictions found", which is exactly how a torch-less prod box ran for
    # two weeks filing every contradiction as an echo.
    if config.STANCE_PROVIDER == "nli":
        try:
            scores = _get_nli().predict([(premise, hypothesis)])[0]
            labels = ["contradiction", "entailment", "neutral"]  # nli-deberta-v3 label order
            return labels[int(scores.argmax())]
        except Exception as e:
            log.warning("stance 'nli' unavailable (%s: %s) — degrading to neutral; "
                        "on a server without torch set STANCE_PROVIDER=hf",
                        type(e).__name__, e)
            return "neutral"  # model unavailable/offline — don't block the save
    if config.STANCE_PROVIDER == "hf":
        try:
            return _get_hf_stance().classify(premise, hypothesis)
        except Exception as e:
            log.warning("stance 'hf' failed (%s: %s) — degrading to neutral",
                        type(e).__name__, e)
            return "neutral"  # transient HF failure — degrade, don't block
    if config.STANCE_PROVIDER == "haiku":
        from core import llm
        try:
            result = llm.call(
                f'Premise: "{premise}"\nHypothesis: "{hypothesis}"\n'
                'Does the hypothesis contradict, entail, or stay neutral to the premise? '
                'Return ONLY JSON: {"stance": "contradiction"|"entailment"|"neutral"}',
                tier="mechanical", max_tokens=2048)
            stance = result["json"].get("stance", "neutral")
            return stance if stance in ("contradiction", "entailment", "neutral") else "neutral"
        except Exception as e:
            log.warning("stance 'haiku' failed (%s: %s) — degrading to neutral",
                        type(e).__name__, e)
            return "neutral"
    return "neutral"


# ── Sentence splitter (ported from v1 engine/extract.py) ──────────────────────
def split_sentences(text: str, min_chars: int | None = None) -> list[str]:
    min_chars = config.SENT_MIN_CHARS if min_chars is None else min_chars
    try:
        import nltk
        try:
            sents = nltk.sent_tokenize(text)
        except LookupError:
            import ssl
            import certifi
            ssl._create_default_https_context = (
                lambda: ssl.create_default_context(cafile=certifi.where()))
            nltk.download("punkt_tab", quiet=True)
            sents = nltk.sent_tokenize(text)
        return [s.strip() for s in sents if len(s.strip()) >= min_chars]
    except Exception:
        pass  # no nltk / no punkt and offline → regex fallback below

    protected = re.sub(r'\b(Mr|Mrs|Dr|Prof|etc|e\.g|i\.e)\.\s', r'\1__DOT__ ', text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    return [p.replace('__DOT__', '.').strip()
            for p in parts if len(p.strip()) >= min_chars]


# ── Receipt ───────────────────────────────────────────────────────────────────
def _build_receipt(conn, user_id: str, sentences: list[str], embeddings,
                   exclude_episode_ids: frozenset | set = frozenset()) -> dict:
    """Classify each sentence against canonical claims (echo/novelty/contradiction)
    and against prior episode sentences (pre-consolidation echo signal).

    exclude_episode_ids: episodes whose sentences must not count as prior matches —
    edit_note passes the note being replaced, else every edit echoes itself."""
    echoes, contradictions, novelties, prior_matches = [], [], [], []

    for i, sent in enumerate(sentences):
        emb = embeddings[i]

        claim_hits = store.knn_claims(conn, user_id, emb, k=3)
        best = claim_hits[0] if claim_hits else None
        if best and best["similarity"] >= config.ECHO_THRESHOLD:
            stance = classify_stance(best["text"], sent) if best["text"] else "neutral"
            entry = {"sentence": sent, "claim_id": best["claim_id"],
                     "claim_text": best["text"],
                     "similarity": round(best["similarity"], 3)}
            if stance == "contradiction":
                contradictions.append(entry)
            else:
                echoes.append(entry)
        elif not best or best["similarity"] < config.NOVELTY_THRESHOLD:
            novelties.append(sent)

        sent_hits = store.knn_sentences(conn, user_id, emb, k=2)
        for hit in sent_hits:
            if hit["episode_id"] in exclude_episode_ids:
                continue
            if hit["similarity"] >= config.ECHO_THRESHOLD:
                prior_matches.append({
                    "sentence": sent,
                    "episode_id": hit["episode_id"],
                    "episode_title": hit["episode_title"],
                    "episode_ts": hit["episode_ts"],
                    "matched_sentence": hit["text"],
                    "similarity": round(hit["similarity"], 3),
                })

    # Full lists, no truncation: the receipt is a consolidation input persisted
    # on an immutable row. Display truncation (RECEIPT_TOP_N) is the MCP layer's job.
    prior_matches.sort(key=lambda m: -m["similarity"])
    return {
        "n_sentences": len(sentences),
        "echoes": sorted(echoes, key=lambda e: -e["similarity"]),
        "contradictions": sorted(contradictions, key=lambda e: -e["similarity"]),
        "novelties": novelties,
        "n_novelties": len(novelties),
        "prior_episode_matches": prior_matches,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def encode(conn, user_id: str, text: str, ts: str | None = None,
           title: str | None = None, source: str = "mcp",
           replay_key: str | None = None,
           exclude_episode_ids: frozenset | set = frozenset()) -> dict:
    """Append an episode for one user, classify novelty, return the receipt.

    ts: ISO timestamp; replayed notes pass their original created_at.
    replay_key: old source id — recorded in replay_map inside the same
    transaction as the episode, so replay is idempotent even across crashes.
    exclude_episode_ids: keep these episodes out of the receipt's prior-episode
    matches (edit_note passes the note being replaced).
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("encode() requires non-empty text")

    ts = ts or store.now_iso()
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:  # naive timestamps are treated as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        ts_unix = dt.timestamp()
    except ValueError:
        ts_unix = None

    sentences = split_sentences(text)
    if not sentences:
        sentences = [text[:500]]

    embeddings = get_embedder().encode(sentences, batch_size=32,
                                       normalize_embeddings=True,
                                       show_progress_bar=False)

    receipt = _build_receipt(conn, user_id, sentences, embeddings,
                             exclude_episode_ids=exclude_episode_ids)
    episode_id = store.new_episode_id(ts_unix)
    receipt["episode_id"] = episode_id

    with conn:
        store.insert_episode(conn, user_id, episode_id, ts, text, title, source,
                             receipt, sentences, embeddings)
        if replay_key is not None:
            store.mark_replayed(conn, user_id, replay_key, episode_id)
        store.append_event(conn, user_id, "ENCODED", {
            "episode_id": episode_id,
            "ts": ts,
            "source": source,
            "n_sentences": len(sentences),
            "n_novelties": receipt["n_novelties"],
            "echo_claim_ids": [e["claim_id"] for e in receipt["echoes"]],
            "contradiction_claim_ids": [c["claim_id"] for c in receipt["contradictions"]],
            "prior_episode_ids": sorted({m["episode_id"] for m in receipt["prior_episode_matches"]}),
        })

    return receipt
