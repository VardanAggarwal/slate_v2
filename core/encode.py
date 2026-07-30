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
        # The router 503s on cold-start//throttle. Without a retry those become
        # degrade-to-neutral, i.e. silently dropped contradictions — the same
        # failure this class exists to fix, just intermittent instead of total.
        import time

        import requests
        last = None
        for attempt in range(config.STANCE_HF_RETRIES):
            try:
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
            except Exception as e:  # noqa: BLE001 — re-raised below if all attempts fail
                last = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise  # permanent (bad model/task/auth) — retrying just wastes time
                if attempt < config.STANCE_HF_RETRIES - 1:
                    time.sleep(config.STANCE_HF_BACKOFF * (2 ** attempt))
        raise last

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


def stance_health(probe: bool = True) -> dict:
    """Is the configured STANCE_PROVIDER actually runnable here? (P0)

    probe=True issues ONE real classification. Called once at startup and cached
    (server.py), so the cost is per-boot, not per-request. Set probe=False only
    where a network call is unacceptable — an unprobed 'ok' means "configured",
    not "working".

    `nli` needs torch, which the 1GB prod image does not ship. When the import
    fails classify_stance() degrades to "neutral" for EVERY pair — _build_receipt
    then buckets every contradiction into `echoes` and the ⚡ line never fires,
    with nothing raised and nothing logged at boot. That ran live 2026-07-09 → 07-30.
    Local dev has torch in .venv, so it never reproduces in tests: hence a startup
    check rather than a test. Returns {provider, ok, detail}."""
    p = config.STANCE_PROVIDER
    if p == "nli":
        try:
            import sentence_transformers  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return {"provider": p, "ok": False,
                    "detail": f"STANCE_PROVIDER=nli but sentence_transformers is "
                              f"unimportable ({type(e).__name__}: {e}) — every stance "
                              f"call will silently return 'neutral' and no contradiction "
                              f"will ever fire. Set STANCE_PROVIDER=hf + HF_TOKEN."}
        return {"provider": p, "ok": True, "detail": "local CrossEncoder available"}
    if p == "hf":
        if not config.HF_TOKEN:
            return {"provider": p, "ok": False,
                    "detail": "STANCE_PROVIDER=hf but HF_TOKEN is empty — every stance "
                              "call will degrade to 'neutral'."}
        if not probe:
            return {"provider": p, "ok": True, "detail": "HF Inference API MNLI (unprobed)"}
        return _probe(p, "HF Inference API MNLI")
    if p == "openrouter":
        if not config.OPENROUTER_KEY:
            return {"provider": p, "ok": False,
                    "detail": "STANCE_PROVIDER=openrouter but OPENROUTER_API_KEY is "
                              "empty — every stance call will degrade to 'neutral'."}
        if not probe:
            return {"provider": p, "ok": True, "detail": "OpenRouter (unprobed)"}
        return _probe(p, f"OpenRouter {config.OPENROUTER_MODEL_MECHANICAL}")
    if p == "haiku":
        return {"provider": p, "ok": True,
                "detail": "billed per pair — never use with the evidence sweep"}
    return {"provider": p, "ok": True, "detail": "stance disabled"}


def _probe(provider: str, detail: str) -> dict:
    """One real call, because a reachable-looking config is not a working provider.

    Checking only that HF_TOKEN is non-empty reported ok=True while every call
    returned 402 Payment Required (exhausted credits) and classify_stance degraded
    every pair to "neutral" — the original silent failure, reached by a different
    route and shown as green on /health. A credential that exists but cannot buy a
    call has to read as broken.
    """
    try:
        verdict = classify_stance("I love working in the office.",
                                  "I hate working in the office.")
    except Exception as e:  # noqa: BLE001 — classify_stance shouldn't raise, but never trust that
        return {"provider": provider, "ok": False,
                "detail": f"{detail} — probe raised {type(e).__name__}: {e}"}
    if verdict != "contradiction":
        return {"provider": provider, "ok": False,
                "detail": f"{detail} — probe returned {verdict!r} for a blatant "
                          f"contradiction, so the provider is unreachable or out of "
                          f"quota and every stance call is silently 'neutral'. "
                          f"Check the server log for the degrade warning."}
    return {"provider": provider, "ok": True, "detail": detail}


def check_stance_provider() -> dict:
    """stance_health(), logged loudly when broken. Called at server/CLI startup."""
    h = stance_health()
    if not h["ok"]:
        log.error("STANCE PROVIDER BROKEN — %s", h["detail"])
    else:
        log.info("stance provider %s ok (%s)", h["provider"], h["detail"])
    return h


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
    if config.STANCE_PROVIDER in ("haiku", "openrouter"):
        # 'openrouter' is 'haiku' with the fallback chain PINNED. The chain's whole
        # point elsewhere is to survive a provider outage, but for stance at sweep
        # volume it is a billing trap: OpenRouter's free tier fails, the next rung
        # is the paid Anthropic API, and a few thousand pairs quietly bill. Pinned,
        # a provider failure is a failure — which is the safe direction here.
        from core import llm
        pinned = config.STANCE_PROVIDER == "openrouter"
        saved = config.LLM_FALLBACK_ORDER
        if pinned:
            config.LLM_FALLBACK_ORDER = ["openrouter"]
        try:
            result = llm.call(
                f'Premise: "{premise}"\nHypothesis: "{hypothesis}"\n'
                'Does the hypothesis contradict, entail, or stay neutral to the premise? '
                'Return ONLY JSON: {"stance": "contradiction"|"entailment"|"neutral"}',
                tier="mechanical", max_tokens=2048)
            stance = result["json"].get("stance", "neutral")
            return stance if stance in ("contradiction", "entailment", "neutral") else "neutral"
        except Exception as e:
            log.warning("stance %r failed (%s: %s) — degrading to neutral",
                        config.STANCE_PROVIDER, type(e).__name__, e)
            return "neutral"
        finally:
            if pinned:
                config.LLM_FALLBACK_ORDER = saved
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
def stance_for(claim_text: str, sentence: str, *, is_evidence: bool) -> str:
    """Stance of one (stored claim, incoming sentence) pair, with the premise chosen
    by which side is the warrant (E2).

    A NOTE is the user thinking again: the stored claim is the premise, the new line
    the hypothesis — "does what I now say follow from what I believed?".

    EVIDENCE is the reverse. The source is the warrant and the claim is what is on
    trial: evidence ⊨ claim. Entailment is directional — a specific finding entails a
    general claim, not the other way round — so run unflipped, genuine backing scores
    `neutral` and lands in `echoes`, indistinguishable from mere topical adjacency."""
    if not claim_text:
        return "neutral"
    return (classify_stance(sentence, claim_text) if is_evidence
            else classify_stance(claim_text, sentence))


def _build_receipt(conn, user_id: str, sentences: list[str], embeddings,
                   exclude_episode_ids: frozenset | set = frozenset(),
                   source: str = "mcp") -> dict:
    """Classify each sentence against canonical claims (echo/novelty/contradiction)
    and against prior episode sentences (pre-consolidation echo signal).

    exclude_episode_ids: episodes whose sentences must not count as prior matches —
    edit_note passes the note being replaced, else every edit echoes itself.
    source: 'research' flips the stance direction (E2) and splits `echoes` into
    entailment (backs a claim) vs neutral (merely relates to it) — the split the
    evidence receipt narrates and the sweep persists."""
    is_evidence = source == store.EVIDENCE_SOURCE
    echoes, contradictions, novelties, prior_matches = [], [], [], []
    # [NOVELTY_THRESHOLD, ECHO_THRESHOLD) used to hit NEITHER branch below, so a
    # sentence there produced no receipt line at all — an on-topic-but-unconfident
    # match was invisible rather than hedged, which is why evidence saves read as
    # "backs nothing you've written yet" when they did touch something.
    weak = []

    for i, sent in enumerate(sentences):
        emb = embeddings[i]

        # k=3 is kept deliberately: one evidence sentence legitimately attaches to
        # several claims with different verdicts (the sweep walks all of them).
        claim_hits = store.knn_claims(conn, user_id, emb, k=3)
        best = claim_hits[0] if claim_hits else None
        if best and best["similarity"] >= config.ECHO_THRESHOLD:
            stance = stance_for(best["text"], sent, is_evidence=is_evidence)
            entry = {"sentence": sent, "claim_id": best["claim_id"],
                     "claim_text": best["text"],
                     "similarity": round(best["similarity"], 3)}
            if is_evidence:
                entry["stance"] = stance
            if stance == "contradiction":
                contradictions.append(entry)
            else:
                echoes.append(entry)
        elif not best or best["similarity"] < config.NOVELTY_THRESHOLD:
            novelties.append(sent)
        else:  # NOVELTY <= sim < ECHO — too close to call novel, too far to assert
            weak.append({"sentence": sent, "claim_id": best["claim_id"],
                         "claim_text": best["text"],
                         "similarity": round(best["similarity"], 3)})

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
                    # E7: whose words the match is. Needed on the NOTE path too —
                    # knn_sentences spans every episode, so once research episodes
                    # exist a note receipt would otherwise render a paper as
                    # "resonates with YOUR note".
                    "episode_source": hit.get("episode_source"),
                    "matched_sentence": hit["text"],
                    "similarity": round(hit["similarity"], 3),
                })

    # Full lists, no truncation: the receipt is a consolidation input persisted
    # on an immutable row. Display truncation (RECEIPT_TOP_N) is the MCP layer's job.
    prior_matches.sort(key=lambda m: -m["similarity"])
    return {
        "n_sentences": len(sentences),
        "source": source,
        "echoes": sorted(echoes, key=lambda e: -e["similarity"]),
        "contradictions": sorted(contradictions, key=lambda e: -e["similarity"]),
        "novelties": novelties,
        "n_novelties": len(novelties),
        # Sub-threshold near-misses. NOT consolidation input — no 'contradicts'
        # edge is ever minted from these (that needs a stance call, which the gate
        # skipped). Display only, so the receipt can hedge instead of going silent.
        "weak_matches": sorted(weak, key=lambda e: -e["similarity"]),
        "prior_episode_matches": prior_matches,
    }


# ── Entry point ───────────────────────────────────────────────────────────────
def encode(conn, user_id: str, text: str, ts: str | None = None,
           title: str | None = None, source: str = "mcp",
           replay_key: str | None = None,
           exclude_episode_ids: frozenset | set = frozenset(),
           citation: dict | None = None) -> dict:
    """Append an episode for one user, classify novelty, return the receipt.

    ts: ISO timestamp; replayed notes pass their original created_at.
    replay_key: old source id — recorded in replay_map inside the same
    transaction as the episode, so replay is idempotent even across crashes.
    exclude_episode_ids: keep these episodes out of the receipt's prior-episode
    matches (edit_note passes the note being replaced).
    citation: E1 provenance for source='research' evidence ({url, title,
    retrieved_at}). Stored in its own column, never folded into the text.
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
                             exclude_episode_ids=exclude_episode_ids,
                             source=source)
    episode_id = store.new_episode_id(ts_unix)
    receipt["episode_id"] = episode_id

    with conn:
        store.insert_episode(conn, user_id, episode_id, ts, text, title, source,
                             receipt, sentences, embeddings, citation=citation)
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
