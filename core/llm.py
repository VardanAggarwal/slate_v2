"""Provider fallback chain (default: openrouter→claude→gemini→local; claude-cli
slots in when LLM_FALLBACK_ORDER is overridden with a subscription token) +
Batch API helpers. openrouter is the primary rung — one key, many underlying
models, its own internal failover — with the direct provider rungs as fallback
below it. Port from /Users/vardanaggarwal/slate engine/extract.py + engine/concepts.py. See PLAN.md §6.

call() is the sync path (Phase 2 dev iteration); submit_batch()/poll_batch()
are the nightly Batch API path (50% off). Callers pick the tier; this module
picks the provider/model and survives rate limits by walking the chain.

OpenRouter's free tier caps at ~20 req/min. We stay under it two ways: a
client-side pacer (_openrouter_pace) spaces dispatches, and a 429 is caught as
RateLimitError so call() waits out the window on the openrouter rung instead of
falling through to the paid claude rung (the billing trap).
"""
import json
import re
import threading
import time

from core import config

# $ per MTok (input, output) — PLAN.md §8 pricing refs
_PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
BATCH_DISCOUNT = 0.5


class LLMError(Exception):
    pass


class RateLimitError(LLMError):
    """A provider hit its request-rate cap (HTTP 429). Carries retry_after
    (seconds) so call() can wait out the window on the same rung instead of
    falling through to a paid one."""
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


# Serializes openrouter dispatches to keep them ≥ OPENROUTER_MIN_INTERVAL_S
# apart (client-side pacing under the free-tier req/min cap). Holding the lock
# across the sleep is intentional: it spaces concurrent callers, not just this
# thread. Uses a monotonic clock so it is immune to wall-clock jumps.
_openrouter_lock = threading.Lock()
_openrouter_last = [0.0]


def _openrouter_pace() -> None:
    interval = config.OPENROUTER_MIN_INTERVAL_S
    if interval <= 0:
        return
    with _openrouter_lock:
        wait = interval - (time.monotonic() - _openrouter_last[0])
        if wait > 0:
            time.sleep(wait)
        _openrouter_last[0] = time.monotonic()


def _retry_after_seconds(resp) -> float:
    """Seconds to wait per a 429's headers. Retry-After is seconds (or an
    HTTP-date, which we ignore); OpenRouter's X-RateLimit-Reset is a Unix epoch
    in milliseconds. Falls back to a fixed default when neither is present."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            pass  # HTTP-date form — fall through to reset header / default
    reset = resp.headers.get("X-RateLimit-Reset")
    if reset:
        try:
            return max(0.0, float(reset) / 1000.0 - time.time())
        except ValueError:
            pass
    return config.OPENROUTER_RATELIMIT_DEFAULT_WAIT


def _model_for_tier(tier: str) -> str:
    return config.CLAUDE_MODEL_JUDGMENT if tier == "judgment" else config.CLAUDE_MODEL_MECHANICAL


def _openrouter_model_for_tier(tier: str) -> str:
    return (config.OPENROUTER_MODEL_JUDGMENT if tier == "judgment"
            else config.OPENROUTER_MODEL_MECHANICAL)


def parse_json(raw: str) -> dict:
    """Strip code fences / stray prose and parse the first JSON object."""
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  batch: bool = False) -> float:
    inp, out = _PRICES.get(model, (3.0, 15.0))
    cost = (input_tokens * inp + output_tokens * out) / 1_000_000
    return cost * (BATCH_DISCOUNT if batch else 1.0)


# ── Sync path ─────────────────────────────────────────────────────────────────
def _call_openrouter(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    """OpenRouter — OpenAI-compatible chat/completions, one key routes to many
    underlying models. Primary rung: tried first, before any direct provider."""
    import requests
    _openrouter_pace()  # stay under the free-tier req/min cap
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {config.OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": messages, "max_tokens": max_tokens,
              # Ask OpenRouter to report real dollar cost in usage.cost —
              # omitted, it silently reads as 0.0 even on paid models.
              "usage": {"include": True}},
        timeout=120,
    )
    if resp.status_code == 429:
        raise RateLimitError(f"openrouter 429: {resp.text[:200]}",
                             _retry_after_seconds(resp))
    if resp.status_code != 200:
        raise LLMError(f"openrouter {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content", "")
    truncated = choice.get("finish_reason") == "length"
    # Reasoning models (GPT-OSS, Nemotron) can burn the whole max_tokens
    # budget on hidden chain-of-thought and return EMPTY visible content with
    # finish_reason="length" — that must flow through as truncated=True so
    # call() escalates the budget, not raise (which would retry at the same
    # budget forever via the backoff path instead).
    if not text and not truncated:
        raise LLMError(f"openrouter empty response: {str(data)[:200]}")
    usage = data.get("usage") or {}
    in_tok, out_tok = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    return {
        "text": text,
        "provider": "openrouter",
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        # OpenRouter reports the real dollar cost per call (0.0 on :free
        # models) — use it directly instead of our own per-model price table,
        # which doesn't know OpenRouter's (possibly discounted) rates.
        "cost": usage.get("cost", 0.0),
        "truncated": truncated,
    }


def _call_claude_cli(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    """Claude Code CLI in print mode — bills the Max/Pro subscription via
    CLAUDE_CODE_OAUTH_TOKEN instead of API keys. Raises LLMError on any
    failure (missing CLI, session limit, timeout) so the chain falls through
    to the API per-call."""
    import os
    import shutil
    import subprocess
    if not shutil.which("claude"):
        raise LLMError("claude CLI not installed")
    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    if system:
        cmd += ["--system-prompt", system]
    # The CLI prefers ANTHROPIC_API_KEY over the subscription OAuth token, so
    # an inherited key makes this rung silently bill the API — the very thing
    # the chain's next step is for. Strip it: this rung is subscription-only.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True,
                              text=True, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        raise LLMError("claude CLI timed out")
    if proc.returncode != 0:
        # newer CLIs exit 1 with the error JSON on stdout and an empty stderr
        detail = (proc.stderr or "").strip()
        if not detail:
            try:
                detail = str(json.loads(proc.stdout).get("result", ""))[:200]
            except (json.JSONDecodeError, AttributeError):
                detail = proc.stdout[:200]
        raise LLMError(f"claude CLI exit {proc.returncode}: {detail}")
    data = json.loads(proc.stdout)
    if data.get("is_error"):
        raise LLMError(f"claude CLI error: {str(data.get('result'))[:200]}")
    usage = data.get("usage") or {}
    return {
        "text": data.get("result", ""),
        "provider": "claude-cli",
        "model": model,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cost": 0.0,  # subscription — no marginal spend
        # Claude Code CLI's --output-format json has no stop_reason field
        # (only type/subtype/is_error/result/usage/total_cost_usd/session_id)
        # — no signal to detect truncation here, so always False.
        "truncated": False,
    }


def _call_claude(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    kwargs = {"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
    if system:
        kwargs["system"] = system
    resp = client.messages.create(**kwargs)
    return {
        "text": resp.content[0].text,
        "provider": "claude",
        "model": model,
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "cost": estimate_cost(model, resp.usage.input_tokens, resp.usage.output_tokens),
        "truncated": resp.stop_reason == "max_tokens",
    }


def _call_gemini(prompt: str, model: str, max_tokens: int, system: str | None) -> dict:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=config.GEMINI_KEY)
    cfg = types.GenerateContentConfig(
        temperature=0.1,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
        **({"system_instruction": system} if system else {}),
    )
    resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
    usage = getattr(resp, "usage_metadata", None)
    candidates = getattr(resp, "candidates", None) or []
    finish_reason = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
    return {
        "text": resp.text,
        "provider": "gemini",
        "model": model,
        "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "cost": 0.0,  # gemini free tier; only claude costs are tracked
        "truncated": "MAX_TOKENS" in finish_reason,
    }


MAX_TOKENS_CEILING = 16384  # cap for the truncation-retry escalation below


def call(prompt: str, tier: str = "mechanical", max_tokens: int = 2048,
         system: str | None = None, json_out: bool = True,
         providers: list[str] | None = None) -> dict:
    """Walk LLM_FALLBACK_ORDER; within claude, use the tier's model.

    `providers` restricts this ONE call to a subset of the chain. Per-call rather
    than by assigning config.LLM_FALLBACK_ORDER: the write path runs its refine
    pass in a thread (write.py), so mutating the module global races with any
    concurrent call — and a save/restore pair that interleaves can capture an
    already-narrowed order and leave the chain permanently narrowed, silently
    costing every other caller its fallback.

    Returns {json?, text, provider, model, input_tokens, output_tokens, cost}.
    Raises LLMError when every remote provider fails — callers that have a
    local fallback (e.g. KMeans blueprint) catch it; others let it bubble so
    the nightly run records a failed status and retries next night.
    """
    last_err: Exception | None = None
    attempts = max(1, config.LLM_MAX_ATTEMPTS)
    for provider in (providers if providers is not None else config.LLM_FALLBACK_ORDER):
        configured = (
            (provider == "openrouter" and config.OPENROUTER_KEY)
            or (provider == "claude-cli" and config.CLAUDE_CODE_OAUTH_TOKEN)
            or (provider == "claude" and config.ANTHROPIC_KEY)
            or (provider == "gemini" and config.GEMINI_KEY))
        if not configured:
            continue  # provider not set up; next provider
        # Up to `attempts` tries with exponential backoff: a transient 503 /
        # 429 / overload (or a malformed-JSON sampling blip) usually clears in
        # seconds, so wait before falling through to the next provider. A
        # response cut off at max_tokens (reasoning models especially burn
        # hidden CoT tokens before the visible answer) also consumes a retry,
        # but escalates the budget instead of just waiting — same provider,
        # doubled max_tokens, up to MAX_TOKENS_CEILING.
        cur_max_tokens = max_tokens
        attempt = 0
        rl_waits = 0  # rate-limit waits are free — they don't consume attempts
        while attempt < attempts:
            try:
                if provider == "openrouter":
                    result = _call_openrouter(prompt, _openrouter_model_for_tier(tier),
                                              cur_max_tokens, system)
                elif provider == "claude-cli":
                    result = _call_claude_cli(prompt, _model_for_tier(tier),
                                              cur_max_tokens, system)
                elif provider == "claude":
                    result = _call_claude(prompt, _model_for_tier(tier), cur_max_tokens, system)
                else:  # gemini
                    result = None
                    g_err: Exception | None = None
                    for m in config.GEMINI_MODELS:
                        try:
                            result = _call_gemini(prompt, m, cur_max_tokens, system)
                            break
                        except Exception as e:  # try next gemini model
                            g_err = e
                    if result is None:  # all gemini models failed — retryable
                        raise g_err or LLMError("no gemini model configured")
                if result.get("truncated") and cur_max_tokens < MAX_TOKENS_CEILING \
                        and attempt < attempts - 1:
                    last_err = LLMError(
                        f"{provider} truncated at max_tokens={cur_max_tokens}")
                    cur_max_tokens = min(cur_max_tokens * 2, MAX_TOKENS_CEILING)
                    attempt += 1
                    continue  # retry same provider immediately, no backoff sleep
                if json_out:
                    result["json"] = parse_json(result["text"])
                return result
            except RateLimitError as e:
                # Wait out the rate-limit window on THIS rung rather than
                # falling through to a paid provider. Cap the wait (a daily-cap
                # 429 resets hours out — don't block on it) and the retry count.
                last_err = e
                if e.retry_after <= config.OPENROUTER_RATELIMIT_MAX_WAIT \
                        and rl_waits < config.OPENROUTER_RATELIMIT_MAX_RETRIES:
                    rl_waits += 1
                    time.sleep(e.retry_after)
                    continue  # free retry: same attempt index, same budget
                break  # give up on this provider; fall through to the next
            except Exception as e:
                last_err = e
                attempt += 1
                if attempt < attempts:
                    time.sleep(config.LLM_BACKOFF_BASE * (2 ** (attempt - 1)))
    raise LLMError(f"all providers failed: {last_err}")


# ── Batch path (Anthropic Message Batches API, 50% off) ───────────────────────
def submit_batch(requests: list[dict]) -> str:
    """requests: [{custom_id, prompt, tier?, max_tokens?, system?}, ...] → batch_id."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    entries = []
    for r in requests:
        params = {
            "model": _model_for_tier(r.get("tier", "mechanical")),
            "max_tokens": r.get("max_tokens", 2048),
            "messages": [{"role": "user", "content": r["prompt"]}],
        }
        if r.get("system"):
            params["system"] = r["system"]
        entries.append({"custom_id": r["custom_id"], "params": params})
    batch = client.messages.batches.create(requests=entries)
    return batch.id


def poll_batch(batch_id: str, interval: int = 30, timeout: int = 24 * 3600) -> dict:
    """Block until the batch ends; return {custom_id: result_dict} for succeeded
    entries; errored entries map to {"error": ...}."""
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_KEY)
    deadline = time.time() + timeout
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        if time.time() > deadline:
            raise LLMError(f"batch {batch_id} timed out")
        time.sleep(interval)

    out = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            msg = entry.result.message
            out[entry.custom_id] = {
                "text": msg.content[0].text,
                "provider": "claude",
                "model": msg.model,
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "cost": estimate_cost(msg.model, msg.usage.input_tokens,
                                      msg.usage.output_tokens, batch=True),
            }
        else:
            out[entry.custom_id] = {"error": entry.result.type}
    return out
