"""Env vars. Port pattern from old engine/config.py. See .env.example."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "engine.db"))

# Old v1 database, used only by migrate.py replay.
SLATE_V1_DB = os.getenv("SLATE_V1_DB", str(Path.home() / "slate" / "data" / "slate.db"))

# ── Embeddings (PLAN.md §2.10: HF Inference API in production via HF_TOKEN;
# local SentenceTransformer only for dev/tests — identical 384-dim vectors) ───
HF_TOKEN         = os.getenv("HF_TOKEN", "")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM        = 384

# ── Encode-time receipt thresholds (cosine similarity) ────────────────────────
# 0.72 → 0.60 (2026-07-30). One shared gate for the receipt AND the evidence
# stance sweep, deliberately not split. 0.72 was the binding constraint, not
# stance quality: a corpus-wide sweep at 0.72 admitted 37 sentence/claim pairs
# and confirmed ZERO contradictions; at 0.60 it admitted 633 and confirmed real
# ones (organic-yield reversal, the ₹647-vs-₹0 pricing tension, the staff-locker
# claim), with stage-1 precision going UP (0.0 → ~0.19), not down. Matches the
# evidence-lane Gate A finding that every pair reaching 0.72 was labelled
# correctly and every miss was a REACH failure at cos .44–.64.
# Cost of the move: ~17x more sentences bucket as echoes, so save receipts are
# chattier. That was accepted as the price of the gate actually reaching.
ECHO_THRESHOLD    = float(os.getenv("ECHO_THRESHOLD", "0.60"))     # >= : echo / contradiction candidate
NOVELTY_THRESHOLD = float(os.getenv("NOVELTY_THRESHOLD", "0.55"))  # <  : novelty
# NOTE: [NOVELTY_THRESHOLD, ECHO_THRESHOLD) is a reporting dead zone — a sentence
# there is neither an echo nor a novelty and produces NO receipt line at all
# (encode.py _build_receipt). Lowering ECHO shrank it from [0.55,0.72) to
# [0.55,0.60); it is not yet closed.
SENT_MIN_CHARS    = int(os.getenv("SENT_MIN_CHARS", "40"))
RECEIPT_TOP_N     = int(os.getenv("RECEIPT_TOP_N", "5"))

# ── Contradiction detection (PLAN.md §9.1: NLI local first, Haiku fallback) ───
# Providers: 'nli'  — local CrossEncoder (needs torch; dev fast-path)
#            'hf'   — HF Inference API zero-shot MNLI (NO torch; the prod path on
#                     the 1GB host, where 'nli' would throw and silently return
#                     "neutral", turning every contradiction into a refine)
#            'haiku'— one Haiku call   'off' — disabled (tests)
STANCE_PROVIDER = os.getenv("STANCE_PROVIDER", "nli")
NLI_MODEL       = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
# Server-side MNLI model for the 'hf' provider (zero-shot via InferenceClient).
STANCE_HF_MODEL = os.getenv("STANCE_HF_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
STANCE_HF_TIMEOUT = float(os.getenv("STANCE_HF_TIMEOUT", "30"))
STANCE_HF_RETRIES = int(os.getenv("STANCE_HF_RETRIES", "3"))    # router 503s on cold-start
STANCE_HF_BACKOFF = float(os.getenv("STANCE_HF_BACKOFF", "1.5"))  # seconds, doubled per retry
# Bucketing of P(anchor ⊨ fragment) into entail/contradict/neutral. Calibratable
# (fitted at consolidation later); these are the profile defaults. NOTE: a single
# zero-shot entailment score separates entail from not-entail, but not-entailed
# spans BOTH neutral and contradiction — so CONTRADICT_MAX is deliberately LOW
# (only a confident non-entailment is called a contradiction) to limit false
# contradictions. The 'nli'/'haiku' providers read the full 3-way distribution;
# a proper 3-class read for 'hf' is a follow-up (see HFStance).
STANCE_ENTAIL_MIN     = float(os.getenv("STANCE_ENTAIL_MIN", "0.60"))   # >= : entailment
STANCE_CONTRADICT_MAX = float(os.getenv("STANCE_CONTRADICT_MAX", "0.10"))  # <= : contradiction

# ── Evidence lane (docs/evidence-lane-plan.md) ────────────────────────────────
# External research/facts saved as episodes with source='research'. The master
# flag gates the GRAPH + RETRIEVAL half (E3 membership kind, E4 sweep, E5 recall
# labels + budget partition). The save path (E1/E2/E6/E7) is inert without a
# research episode existing, so it carries no flag.
EVIDENCE_LANE = os.getenv("EVIDENCE_LANE", "0") == "1"
# Share of the assemble_context specifics budget reserved for evidence. Partitioned
# like res_depth_share: evidence gets its own slice instead of bidding freely, so
# self-lane Coverage@B stays measurable against the existing gold sets.
EVIDENCE_SHARE = float(os.getenv("EVIDENCE_SHARE", "0.20"))
# Cap on stance calls per nightly sweep. A consolidation run that re-canonicalises
# many claims makes that night's sweep proportionally large; what the cap drops is
# logged, never silently truncated (it is picked up next run).
EVIDENCE_SWEEP_MAX_PAIRS = int(os.getenv("EVIDENCE_SWEEP_MAX_PAIRS", "2000"))

# ── Write refine pass (W2–W8; core/write.py) ──────────────────────────────────
# W1 (persist raw + cheap receipt) is sync. The predictor-driven fragmentation /
# routing is async + retryable: save_note kicks it off best-effort in a thread,
# and refine_pending() sweeps anything HF/LLM failures left behind.
WRITE_REFINE_ASYNC = os.getenv("WRITE_REFINE_ASYNC", "1") == "1"
WRITE_REINFORCE_BUMP = float(os.getenv("WRITE_REINFORCE_BUMP", "0.25"))  # strength bump on a confirmed prediction
# Initial hold for a fragment the resolver flags as a CONTRADICTION — born held
# stronger than a refine/novel (PRD W6: "store, held strongest, flag").
WRITE_CONTRADICT_HOLD = float(os.getenv("WRITE_CONTRADICT_HOLD", "2.0"))

# ── API keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_KEY     = os.getenv("GEMINI_API_KEY", "")
# OpenRouter — primary rung (single key routes to many underlying models,
# incl. Anthropic/Gemini/OSS, with its own failover). Falls through to the
# direct provider rungs below when unset or erroring.
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Claude Code subscription token (`claude setup-token`) — enables the
# claude-cli provider: nightly LLM calls ride the Max/Pro subscription and
# fall through to API keys per-call when the session is limited.
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN", "")

# ── Model tiering (PLAN.md §2.11) ─────────────────────────────────────────────
CLAUDE_MODEL_MECHANICAL = os.getenv("CLAUDE_MODEL_MECHANICAL", "claude-haiku-4-5")
CLAUDE_MODEL_JUDGMENT   = os.getenv("CLAUDE_MODEL_JUDGMENT", "claude-sonnet-4-6")

# OpenRouter model ids (provider-prefixed slugs, e.g. "anthropic/claude-...").
# Defaults are free-tier (":free" suffix) — this is the primary rung, billed
# to nothing until a paid model is deliberately chosen.
OPENROUTER_MODEL_MECHANICAL = os.getenv("OPENROUTER_MODEL_MECHANICAL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_MODEL_JUDGMENT   = os.getenv("OPENROUTER_MODEL_JUDGMENT", "nvidia/nemotron-3-super-120b-a12b:free")


def _csv(var: str, default: list[str]) -> list[str]:
    raw = os.getenv(var, "")
    return [m.strip() for m in raw.split(",") if m.strip()] if raw else default


# openrouter tried first: one key, many underlying models + its own failover.
# Direct rungs (claude-cli/claude/gemini) stay as the fallback chain below it.
LLM_FALLBACK_ORDER = _csv("LLM_FALLBACK_ORDER", ["openrouter", "claude", "gemini", "local"])
GEMINI_MODELS      = _csv("GEMINI_MODELS", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])

# ── LLM retry/backoff (absorb transient 503/429/overload within a run) ────────
# Each configured provider gets up to LLM_MAX_ATTEMPTS tries with exponential
# backoff (LLM_BACKOFF_BASE * 2**attempt seconds) before falling through to the
# next provider. Keeps a single nightly run alive across a brief provider spike.
LLM_MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
LLM_BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "2.0"))  # seconds

# ── OpenRouter rate-limit handling (free tier ≈ 20 req/min) ───────────────────
# Client-side pacing: min seconds between openrouter dispatches so bulk loops
# stay under the per-minute cap instead of firing bursts that 429. 3s ⇒ 20/min.
# Set to 0 to disable pacing.
OPENROUTER_MIN_INTERVAL_S = float(os.getenv("OPENROUTER_MIN_INTERVAL_S", "3.0"))
# On a 429 we wait out the window on openrouter rather than falling through to a
# paid rung (the billing trap). But a *daily*-cap 429 resets hours out — never
# block on that; if the reported wait exceeds MAX_WAIT, bail to the next
# provider. DEFAULT_WAIT is used when the response carries no reset header.
OPENROUTER_RATELIMIT_MAX_WAIT     = float(os.getenv("OPENROUTER_RATELIMIT_MAX_WAIT", "90"))
OPENROUTER_RATELIMIT_MAX_RETRIES  = int(os.getenv("OPENROUTER_RATELIMIT_MAX_RETRIES", "6"))
OPENROUTER_RATELIMIT_DEFAULT_WAIT = float(os.getenv("OPENROUTER_RATELIMIT_DEFAULT_WAIT", "6.0"))

# ── Concept health (ported from v1 health.py state model) ─────────────────────
HEALTH_SCORES = {
    "grounded": 1.3,
    "active":   1.0,
    "stale":    0.4,
    "dormant":  0.2,
}
HEALTH_ACTIVE_DAYS = int(os.getenv("HEALTH_ACTIVE_DAYS", "30"))
HEALTH_STALE_DAYS  = int(os.getenv("HEALTH_STALE_DAYS", "60"))

# ── MCP server ────────────────────────────────────────────────────────────────
AUTH_USER      = os.getenv("AUTH_USER", "")
AUTH_PASS      = os.getenv("AUTH_PASS", "")
SLATE_BASE_URL = os.getenv("SLATE_BASE_URL", "http://localhost:8000")

# Corpus owner used when no auth is configured at all (local dev / CLI on a
# fresh box). With OAuth enabled, identity always comes from the token —
# never from this (AUTH.md §2).
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "local")
