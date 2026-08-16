"""Provider-agnostic LLM client via litellm.

All model calls in raglab go through this module.  To switch provider, set
two env vars — that's it, no code to change::

    CHAT_MODEL=anthropic/claude-haiku-3
    EMBED_MODEL=ollama/nomic-embed-text

litellm reads the corresponding API key automatically:
  OpenAI / compatible  → OPENAI_API_KEY (+ optional OPENAI_BASE_URL)
  Anthropic            → ANTHROPIC_API_KEY
  Google Gemini        → GEMINI_API_KEY
  Groq                 → GROQ_API_KEY
  Ollama (local)       → no key needed, OLLAMA_API_BASE if non-default

Preflight check
---------------
Call :func:`check_provider_ready` early (before any embedding or generation)
to get a clear error message if a required env var is missing, rather than a
cryptic HTTP 401 later.
"""

from __future__ import annotations

import functools
import importlib
import os
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TypeVar

from raglab.core.text import token_count
from raglab.providers.env import env_float, load_dotenv

# ─── Defaults (readable from env, overridable per pipeline) ──────────────────

CHAT_MODEL_DEFAULT = "gpt-4.1-mini"
EMBED_MODEL_DEFAULT = "text-embedding-3-small"
_ACTIVE_LEDGER: ContextVar[ProviderUsageLedger | None] = ContextVar("raglab_provider_ledger", default=None)


def default_chat_model() -> str:
    """Return the configured chat model (``CHAT_MODEL`` env, else gpt-4.1-mini)."""
    load_dotenv()
    return os.getenv("CHAT_MODEL", CHAT_MODEL_DEFAULT)


def default_embed_model() -> str:
    """Return the configured embedding model (``EMBED_MODEL`` env, else text-embedding-3-small)."""
    load_dotenv()
    return os.getenv("EMBED_MODEL", EMBED_MODEL_DEFAULT)


# ─── Preflight check ─────────────────────────────────────────────────────────

# Map model-name prefixes → required env var.
_KEY_REQUIREMENTS: list[tuple[tuple[str, ...], str | None]] = [
    (("gpt-", "o1", "o3", "o4", "text-embedding-3", "text-embedding-ada"), "OPENAI_API_KEY"),
    (("claude-", "anthropic/"), "ANTHROPIC_API_KEY"),
    (("gemini/", "gemini-"), "GEMINI_API_KEY"),
    (("groq/",), "GROQ_API_KEY"),
    (("together/",), "TOGETHER_API_KEY"),
    (("ollama/", "ollama_chat/"), None),  # local, no key
    (("openrouter/",), "OPENROUTER_API_KEY"),
]

# Renamed for provider-neutral pricing (litellm supports many providers, not
# just OpenAI). Old names are rejected rather than silently aliased so a stale
# .env cannot make ``cost_estimate`` quietly report the wrong number.
_LEGACY_PRICING_ENV: dict[str, str] = {
    "OPENAI_EMBEDDING_INPUT_COST_PER_1K": "LLM_EMBEDDING_INPUT_COST_PER_1K",
    "OPENAI_CHAT_INPUT_COST_PER_1K": "LLM_CHAT_INPUT_COST_PER_1K",
    "OPENAI_CHAT_OUTPUT_COST_PER_1K": "LLM_CHAT_OUTPUT_COST_PER_1K",
}


def validate_pricing_env() -> None:
    """Raise ``RuntimeError`` if a legacy ``OPENAI_*_COST_PER_1K`` var is set.

    These were renamed to ``LLM_*_COST_PER_1K`` (litellm is multi-provider, not
    OpenAI-only). Rejecting loudly instead of aliasing prevents a stale ``.env``
    from silently under-reporting cost.
    """
    load_dotenv()
    present = sorted(name for name in _LEGACY_PRICING_ENV if os.getenv(name) is not None)
    if present:
        migrations = "; ".join(f"{name} -> {_LEGACY_PRICING_ENV[name]}" for name in present)
        raise RuntimeError(
            f"Legacy pricing environment variable(s) are no longer read: {migrations}. "
            "Rename them in your .env / shell environment (see .env.example)."
        )


def check_provider_ready(model: str) -> None:
    """Raise ``RuntimeError`` with a helpful message if the API key for *model* is missing.

    Call this before the first LLM call in a pipeline stage (ingest / query).
    Silent if the model is local (Ollama) or if the key is already present.

    Example::

        check_provider_ready(self.embedding_model)   # in ingest()
        check_provider_ready(self.generator_model)   # in query()
    """
    load_dotenv()
    validate_pricing_env()
    name = model.strip().lower()
    for prefixes, env_var in _KEY_REQUIREMENTS:
        if any(name.startswith(p) for p in prefixes):
            if env_var is None:
                return  # local provider — no key needed
            if os.getenv(env_var):
                return  # key is present
            raise RuntimeError(
                f"Model '{model}' requires {env_var} but it is not set.\n"
                f"  • Add '{env_var}=<your-key>' to .env in the project root, or\n"
                f"  • export {env_var}=<your-key> in your shell.\n"
                f"  • To use a different provider set CHAT_MODEL / EMBED_MODEL accordingly."
            )
    # Unknown prefix — let litellm raise its own error at call time.


# ─── Retry / backoff ─────────────────────────────────────────────────────────

# Names, not classes: litellm re-raises provider errors as its own exception
# hierarchy (mirroring the openai SDK's), but importing that hierarchy here
# would force litellm to load even when every call is mocked out in tests.
# Transient: worth retrying. Anything else (auth, bad request, not found,
# model/config errors) fails fast — retrying those only burns time and money.
_RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APIConnectionRateLimitError",
    "Timeout",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIError",
}

_T = TypeVar("_T")


def _is_retryable(exc: Exception) -> bool:
    return type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES


def _call_with_retry(
    func: Callable[[], _T],
    *,
    max_attempts: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> tuple[_T, int]:
    """Call *func*, retrying transient failures with exponential backoff + jitter.

    Returns ``(result, retry_count)``. Non-transient errors (auth, bad
    request, unknown model) and the final attempt's transient error both
    propagate immediately — a confused caller should fail fast, not spin.
    """
    attempt = 0
    while True:
        try:
            return func(), attempt
        except Exception as exc:
            if attempt + 1 >= max_attempts or not _is_retryable(exc):
                raise
            delay = min(max_delay, base_delay * (2**attempt)) * (0.5 + random.random())
            time.sleep(min(delay, max_delay))
            attempt += 1


# ─── Result type ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ChatCompletionResult:
    text: str
    usage: dict[str, int]
    latency_ms: float
    estimated_cost: float
    retries: int = 0


@dataclass(slots=True)
class ProviderUsageLedger:
    chat_calls: int = 0
    embedding_calls: int = 0
    chat_usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )
    embedding_tokens: int = 0
    estimated_cost: float = 0.0
    pricing_configured: bool = False
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_calls": self.chat_calls,
            "embedding_calls": self.embedding_calls,
            "chat_usage": self.chat_usage,
            "embedding_tokens": self.embedding_tokens,
            "estimated_cost": round(self.estimated_cost, 8),
            "retries": self.retries,
            "cost_status": (
                "estimated" if self.pricing_configured or self.chat_calls + self.embedding_calls == 0 else "unknown"
            ),
        }


@contextmanager
def capture_provider_usage():
    ledger = ProviderUsageLedger()
    token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_LEDGER.reset(token)


# ─── Client ──────────────────────────────────────────────────────────────────


class LLMClient:
    """Thin wrapper around litellm that exposes the two calls raglab needs.

    Parameters
    ----------
    timeout:
        HTTP timeout in seconds for each individual request.
    """

    def __init__(self, timeout: float = 60.0) -> None:
        load_dotenv()
        self.timeout = timeout

    # ── Embeddings ───────────────────────────────────────────────────────────

    def create_embeddings(self, model: str, inputs: list[str], batch_size: int = 64) -> list[list[float]]:
        """Return one float vector per input string.

        Batches the inputs to stay within provider limits.
        """
        vectors: list[list[float]] = []
        litellm = _litellm()
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            response, retries = _call_with_retry(
                functools.partial(litellm.embedding, model=model, input=batch, timeout=self.timeout)
            )
            # litellm returns EmbeddingResponse; .data is a list of Embedding objects
            sorted_items = sorted(response.data, key=lambda item: item["index"])
            vectors.extend(item["embedding"] for item in sorted_items)
            ledger = _ACTIVE_LEDGER.get()
            if ledger is not None:
                tokens = sum(token_count(text) for text in batch)
                rate = env_float("LLM_EMBEDDING_INPUT_COST_PER_1K", 0.0)
                ledger.embedding_calls += 1
                ledger.embedding_tokens += tokens
                ledger.estimated_cost += tokens / 1000 * rate
                ledger.pricing_configured = ledger.pricing_configured or rate > 0
                ledger.retries += retries
        return vectors

    # ── Chat completions ─────────────────────────────────────────────────────

    def create_chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> ChatCompletionResult:
        started = time.perf_counter()
        litellm = _litellm()
        response, retries = _call_with_retry(
            lambda: litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.timeout,
            )
        )
        latency_ms = (time.perf_counter() - started) * 1000
        text: str = response.choices[0].message.content or ""
        usage = _extract_usage(response, messages, text)
        result = ChatCompletionResult(
            text=text.strip(),
            usage=usage,
            latency_ms=round(latency_ms, 3),
            estimated_cost=_estimate_chat_cost(usage),
            retries=retries,
        )
        ledger = _ACTIVE_LEDGER.get()
        if ledger is not None:
            ledger.chat_calls += 1
            for key, value in usage.items():
                ledger.chat_usage[key] += int(value)
            ledger.estimated_cost += result.estimated_cost
            ledger.pricing_configured = ledger.pricing_configured or any(
                env_float(name, 0.0) > 0 for name in ("LLM_CHAT_INPUT_COST_PER_1K", "LLM_CHAT_OUTPUT_COST_PER_1K")
            )
            ledger.retries += retries
        return result


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _extract_usage(response: Any, messages: list[dict[str, str]], output: str) -> dict[str, int]:
    raw = getattr(response, "usage", None) or {}
    if hasattr(raw, "__dict__"):
        raw = raw.__dict__
    prompt_tokens = int(raw.get("prompt_tokens") or raw.get("input_tokens") or 0)
    completion_tokens = int(raw.get("completion_tokens") or raw.get("output_tokens") or 0)
    if prompt_tokens == 0:
        prompt_tokens = sum(token_count(m.get("content", "")) for m in messages)
    if completion_tokens == 0:
        completion_tokens = token_count(output)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": int(raw.get("total_tokens") or prompt_tokens + completion_tokens),
    }


def _estimate_chat_cost(usage: dict[str, int]) -> float:
    input_cost = env_float("LLM_CHAT_INPUT_COST_PER_1K", 0.0)
    output_cost = env_float("LLM_CHAT_OUTPUT_COST_PER_1K", 0.0)
    return round(
        usage["prompt_tokens"] / 1000 * input_cost + usage["completion_tokens"] / 1000 * output_cost,
        8,
    )


def _litellm() -> Any:
    """Import LiteLLM only when a paid/local model call is actually requested."""
    module: Any = importlib.import_module("litellm")
    module.suppress_debug_info = True
    return module
