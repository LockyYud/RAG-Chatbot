from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from ragbench.core.base import load_pipeline
from ragbench.core.doctor import diagnose_technique
from ragbench.providers.llm_client import probe_provider


class AuthenticationError(Exception):
    """Stand-in for litellm/openai-SDK's AuthenticationError — probe_provider
    only inspects ``type(exc).__name__``, so a plain local class with that
    name is enough to exercise the auth-vs-missing-key distinction without
    importing litellm."""


def _fake_litellm(*, calls: list[str], raise_exc: Exception | None = None, chat_response: Any = None) -> Any:
    def _record(name: str):
        def _call(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            if raise_exc is not None:
                raise raise_exc
            return chat_response

        return _call

    return SimpleNamespace(embedding=_record("embedding"), completion=_record("chat"), rerank=_record("rerank"))


def test_missing_key_never_attempts_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    # ANTHROPIC_API_KEY is never set in this repo's .env, so deleting it here
    # (and never calling _litellm) really does simulate "no key at all".
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    calls: list[str] = []
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", lambda: _fake_litellm(calls=calls))

    result = probe_provider("claude-haiku-3", "chat")

    assert calls == []  # no network call attempted
    assert result["configured"] is False
    assert result["reachable"] is False
    assert result["error_type"] == "RuntimeError"  # check_provider_ready's own error
    assert "ANTHROPIC_API_KEY" in result["error_detail"]


def test_present_but_revoked_key_is_distinct_from_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "revoked-key")
    calls: list[str] = []
    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm",
        lambda: _fake_litellm(calls=calls, raise_exc=AuthenticationError("key revoked")),
    )

    result = probe_provider("claude-haiku-3", "chat")

    assert calls == ["chat"]  # a live call WAS attempted, unlike the missing-key case
    assert result["configured"] is True
    assert result["reachable"] is False
    assert result["error_type"] == "AuthenticationError"


def test_live_probe_makes_exactly_one_attempt_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "revoked-key")
    calls: list[str] = []
    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm",
        lambda: _fake_litellm(calls=calls, raise_exc=AuthenticationError("key revoked")),
    )

    probe_provider("claude-haiku-3", "chat")

    assert len(calls) == 1


def test_embedding_probe_bypasses_the_embedding_cache(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))

    from ragbench.providers.embedding_cache import EmbeddingCache

    # Pre-seed the cache with a vector for exactly the probe's text — if the
    # probe went through the cache, this stale hit would make a revoked key
    # look alive.
    cache = EmbeddingCache(tmp_path / "embeddings.sqlite")
    cache.put("text-embedding-3-small", "ragbench provider health check", [1.0, 2.0, 3.0])
    cache.close()

    cache_get_calls: list[str] = []
    from ragbench.providers import embedding_cache as embedding_cache_module

    original_get = embedding_cache_module.EmbeddingCache.get

    def _tracking_get(self, model: str, text: str):
        cache_get_calls.append(text)
        return original_get(self, model, text)

    monkeypatch.setattr(embedding_cache_module.EmbeddingCache, "get", _tracking_get)

    calls: list[str] = []
    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm",
        lambda: _fake_litellm(calls=calls, raise_exc=AuthenticationError("key revoked")),
    )

    result = probe_provider("text-embedding-3-small", "embedding")

    assert calls == ["embedding"]  # the real call happened...
    assert cache_get_calls == []  # ...and never went through the cache's get()
    assert result["reachable"] is False
    assert result["error_type"] == "AuthenticationError"


def test_local_reranker_backend_never_triggers_a_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """parent_child has no reranker at all; bm25_hybrid_rerank defaults to a
    local reranker_backend — either way, a local backend must never probe
    the network (rerank operation) for reranker_model, even though other
    attributes (embedding_model) are still live-probed."""
    calls: list[str] = []
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", lambda: _fake_litellm(calls=calls))
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")

    pipeline = load_pipeline("bm25_hybrid_rerank")
    assert pipeline is not None
    assert pipeline.reranker_backend == "local"  # type: ignore[attr-defined]

    diagnose_technique(pipeline, mode="retrieval_only")

    assert "rerank" not in calls


def test_api_reranker_backend_is_probed_via_rerank_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("COHERE_API_KEY", "fake-key")
    calls: list[str] = []
    fake_rerank_response = SimpleNamespace(results=[{"index": 0, "relevance_score": 0.9}])
    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm",
        lambda: _fake_litellm(calls=calls, chat_response=fake_rerank_response),
    )

    pipeline = load_pipeline(
        "bm25_hybrid_rerank",
        params={"reranker_backend": "api", "reranker_model": "cohere/rerank-english-v3.0"},
    )
    assert pipeline is not None

    report = diagnose_technique(pipeline, mode="retrieval_only")

    assert "rerank" in calls
    cross_encoder_check = next(c for c in report["checks"] if c["name"] == "cross_encoder")
    assert cross_encoder_check["status"] == "ok"


def test_contextual_retrieval_retrieval_only_probes_embedding_and_context_not_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    calls: list[tuple[str, str]] = []

    def _tracking_probe(model: str, operation: str) -> dict[str, Any]:
        calls.append((operation, model))
        return {"configured": True, "reachable": True, "latency_ms": 1.0, "error_type": None, "error_detail": None}

    monkeypatch.setattr("ragbench.providers.llm_client.probe_provider", _tracking_probe)

    pipeline = load_pipeline(
        "contextual_retrieval_2024",
        params={"context_model": "claude-haiku-3", "generator_model": "gpt-4.1-mini"},
    )
    assert pipeline is not None

    report = diagnose_technique(pipeline, mode="retrieval_only")

    assert report["ready"] is True
    probed_models = {model for _op, model in calls}
    assert "claude-haiku-3" in probed_models  # context_model: always checked
    assert "text-embedding-3-small" in probed_models  # embedding_model: always checked
    assert "gpt-4.1-mini" not in probed_models  # generator_model: full_rag-only, not needed here


def test_diagnose_technique_dedupes_live_probes_for_shared_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """contextual_retrieval_2024 in full_rag mode has context_model and
    generator_model both defaulting to the same chat model — that must be
    exactly one live probe, not two."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    calls: list[tuple[str, str]] = []

    def _tracking_probe(model: str, operation: str) -> dict[str, Any]:
        calls.append((operation, model))
        return {"configured": True, "reachable": True, "latency_ms": 1.0, "error_type": None, "error_detail": None}

    monkeypatch.setattr("ragbench.providers.llm_client.probe_provider", _tracking_probe)

    pipeline = load_pipeline("contextual_retrieval_2024")
    assert pipeline is not None
    assert pipeline.context_model == pipeline.generator_model == "gpt-4.1-mini"  # type: ignore[attr-defined]

    diagnose_technique(pipeline, mode="full_rag")

    chat_probes_for_shared_model = [c for c in calls if c == ("chat", "gpt-4.1-mini")]
    assert len(chat_probes_for_shared_model) == 1


def test_offline_mode_skips_live_probes_entirely(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    calls: list[str] = []
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", lambda: _fake_litellm(calls=calls))

    pipeline = load_pipeline("naive_rag")
    assert pipeline is not None

    report = diagnose_technique(pipeline, mode="retrieval_only", offline=True)

    assert calls == []
    assert report["ready"] is True
