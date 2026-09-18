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


@pytest.fixture(autouse=True)
def _isolate_provider_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the real repo ``.env`` from leaking model defaults into tests.

    ``load_dotenv()`` uses ``os.environ.setdefault`` against the repo's own
    ``.env`` file on every call, so merely deleting these vars isn't enough —
    the next ``default_embed_model()``/``default_chat_model()`` call reloads
    a dev's real values (e.g. ``EMBED_MODEL=bge-m3``) right back, silently
    overriding the neutral defaults (``text-embedding-3-small`` /
    ``gpt-4.1-mini``) that tests assert on — even though CI, with no ``.env``
    file, would never see them. Pin them to the neutral defaults instead so
    ``setdefault`` is a no-op.
    """
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("CHAT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "gpt-4.1-mini")
