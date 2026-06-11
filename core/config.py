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
ECHO_THRESHOLD    = float(os.getenv("ECHO_THRESHOLD", "0.72"))     # >= : echo / contradiction candidate
NOVELTY_THRESHOLD = float(os.getenv("NOVELTY_THRESHOLD", "0.55"))  # <  : novelty
SENT_MIN_CHARS    = int(os.getenv("SENT_MIN_CHARS", "40"))
RECEIPT_TOP_N     = int(os.getenv("RECEIPT_TOP_N", "5"))

# ── Contradiction detection (PLAN.md §9.1: NLI local first, Haiku fallback) ───
STANCE_PROVIDER = os.getenv("STANCE_PROVIDER", "nli")  # 'nli' | 'haiku' | 'off'
NLI_MODEL       = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-small")

# ── API keys ──────────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_KEY    = os.getenv("GEMINI_API_KEY", "")

# ── Model tiering (PLAN.md §2.11) ─────────────────────────────────────────────
CLAUDE_MODEL_MECHANICAL = os.getenv("CLAUDE_MODEL_MECHANICAL", "claude-haiku-4-5")
CLAUDE_MODEL_JUDGMENT   = os.getenv("CLAUDE_MODEL_JUDGMENT", "claude-sonnet-4-6")


def _csv(var: str, default: list[str]) -> list[str]:
    raw = os.getenv(var, "")
    return [m.strip() for m in raw.split(",") if m.strip()] if raw else default


LLM_FALLBACK_ORDER = _csv("LLM_FALLBACK_ORDER", ["claude", "gemini", "local"])
GEMINI_MODELS      = _csv("GEMINI_MODELS", ["gemini-2.5-flash", "gemini-2.5-flash-lite"])

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
