from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_embedding_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give every test its own embedding cache directory.

    Without this, the default ``.raglab_cache/`` would live inside the real
    repo checkout while running tests, and — since the cache is keyed only by
    (model, normalized text) — an unrelated test embedding the same fixture
    string with the same fake model name could get a cache hit seeded by a
    completely different test earlier in the same pytest run.
    """
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE_DIR", str(tmp_path / "embedding_cache"))
