from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from raglab.providers.embedding_cache import EmbeddingCache, get_embedding_cache
from raglab.providers.llm_client import LLMClient, capture_provider_usage


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]


def _counting_fake_litellm(call_log: list[list[str]]) -> Any:
    def embedding(model: str, input: list[str], timeout: float) -> _FakeEmbeddingResponse:
        call_log.append(list(input))
        # Deterministic, distinct-per-text vector so ordering/correctness is checkable.
        return _FakeEmbeddingResponse([[float(len(text)), float(sum(map(ord, text)) % 97)] for text in input])

    return SimpleNamespace(embedding=embedding)


def test_embedding_cache_roundtrip(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    assert cache.get("model-a", "hello") is None
    cache.put("model-a", "hello", [1.0, 2.0, 3.0])
    assert cache.get("model-a", "hello") == pytest.approx([1.0, 2.0, 3.0])
    cache.close()


def test_embedding_cache_keys_by_model_and_normalized_text(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "cache.sqlite")
    cache.put("model-a", "hello  world", [1.0])
    # Different model -> different key, even for the identical text.
    assert cache.get("model-b", "hello  world") is None
    # Whitespace differences collapse under normalize_text -> same key.
    assert cache.get("model-a", "hello world") == pytest.approx([1.0])
    cache.close()


def test_get_embedding_cache_disabled_via_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE", "0")
    assert get_embedding_cache() is None
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE", "1")
    cache = get_embedding_cache()
    assert cache is not None
    cache.close()


def test_create_embeddings_skips_api_call_on_cache_hit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    call_log: list[list[str]] = []
    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _counting_fake_litellm(call_log))
    client = LLMClient()

    first = client.create_embeddings("fake-embed", ["alpha", "beta"])
    assert call_log == [["alpha", "beta"]]

    second = client.create_embeddings("fake-embed", ["alpha", "beta"])
    assert call_log == [["alpha", "beta"]]  # no new call — both were cache hits
    assert second == first


def test_create_embeddings_only_calls_api_for_cache_misses_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    call_log: list[list[str]] = []
    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _counting_fake_litellm(call_log))
    client = LLMClient()

    client.create_embeddings("fake-embed", ["alpha"])
    call_log.clear()

    mixed = client.create_embeddings("fake-embed", ["new", "alpha", "also-new"])

    assert call_log == [["new", "also-new"]]  # only the two misses were sent, "alpha" was a hit
    reference = client.create_embeddings("fake-embed", ["new", "alpha", "also-new"])
    assert mixed == reference  # order matches original input order regardless of hit/miss split


def test_create_embeddings_ledger_tracks_hits_and_misses_and_cache_hits_are_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "1.0")
    call_log: list[list[str]] = []
    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _counting_fake_litellm(call_log))
    client = LLMClient()

    with capture_provider_usage() as first_ledger:
        client.create_embeddings("fake-embed", ["alpha", "beta"])
    first = first_ledger.to_dict()
    assert first["embedding_cache_hits"] == 0
    assert first["embedding_cache_misses"] == 2
    assert first["embedding_calls"] == 1
    assert first["embedding_cost"] > 0

    with capture_provider_usage() as second_ledger:
        client.create_embeddings("fake-embed", ["alpha", "beta"])
    second = second_ledger.to_dict()
    assert second["embedding_cache_hits"] == 2
    assert second["embedding_cache_misses"] == 0
    assert second["embedding_calls"] == 0  # no real API call happened
    assert second["embedding_cost"] == 0.0  # cache hits are free, not double-counted


def test_no_embedding_cache_env_disables_reuse_across_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE", "0")
    call_log: list[list[str]] = []
    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _counting_fake_litellm(call_log))
    client = LLMClient()

    client.create_embeddings("fake-embed", ["alpha"])
    client.create_embeddings("fake-embed", ["alpha"])

    assert call_log == [["alpha"], ["alpha"]]  # cache disabled -> every call hits the "API"


def test_concurrent_writers_to_the_same_cache_file_do_not_raise_database_locked(tmp_path: Path) -> None:
    """raglab eval/bench --concurrency has every worker thread open its own
    EmbeddingCache connection to the same sqlite file. Without WAL + a busy
    timeout, concurrent writers race into "database is locked" instead of
    just waiting their turn."""
    path = tmp_path / "cache.sqlite"

    def write_one(index: int) -> None:
        cache = EmbeddingCache(path)
        try:
            cache.put("model-a", f"text-{index}", [float(index)])
        finally:
            cache.close()

    with ThreadPoolExecutor(max_workers=16) as executor:
        # list(...) forces every future to resolve, re-raising any exception
        # (e.g. sqlite3.OperationalError: database is locked) here.
        list(executor.map(write_one, range(64)))

    verify_cache = EmbeddingCache(path)
    try:
        for index in range(64):
            assert verify_cache.get("model-a", f"text-{index}") == pytest.approx([float(index)])
    finally:
        verify_cache.close()


def test_concurrent_create_embeddings_across_threads_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Simulates raglab eval --concurrency: many threads, each with their own
    LLMClient, embedding different (guaranteed cache-miss) texts against the
    same on-disk cache at the same time."""
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path))
    call_log: list[list[str]] = []
    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _counting_fake_litellm(call_log))

    def embed_one(index: int) -> list[list[float]]:
        return LLMClient().create_embeddings("fake-embed", [f"concurrent-text-{index}"])

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(embed_one, range(64)))

    assert len(results) == 64
    # Every distinct text was a genuine miss — none lost, none duplicated.
    assert sum(len(batch) for batch in call_log) == 64
