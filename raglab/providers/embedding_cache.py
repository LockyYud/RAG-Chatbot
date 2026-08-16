"""Persistent embedding cache keyed by ``(model, normalized_text)``.

Avoids re-paying for an embedding call whose exact (model, text) pair was
already computed — across ingest re-runs of the same corpus, and across eval
queries repeated by ``--resume`` or by multiple techniques evaluated on the
same dataset. Backed by sqlite (stdlib) rather than JSON: parsing floats out
of text is the exact cost artifact v4 already removed from ``nodes.json``,
so a JSON-based cache would reintroduce the same problem for embeddings that
happen to be cached instead of freshly computed.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from raglab.core.text import normalize_text

DEFAULT_CACHE_DIR = ".raglab_cache"
CACHE_FILENAME = "embeddings.sqlite"


class EmbeddingCache:
    """One sqlite file, one table, keyed by sha256(model + normalized text)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # timeout=30 makes sqlite3 itself retry internally (rather than raise
        # "database is locked" immediately) when a writer collides with
        # another connection — expected under raglab eval/bench --concurrency,
        # where every worker thread opens its own connection to the same file.
        self._conn = sqlite3.connect(str(self._path), timeout=30.0)
        # WAL lets readers proceed without blocking on a concurrent writer (and
        # vice versa), which is the actual source of most lock contention here
        # — one writer at a time is still serialized, but busy_timeout below
        # covers that remaining case by waiting instead of failing outright.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, vector BLOB NOT NULL, dimension INTEGER NOT NULL)"
        )

    @staticmethod
    def cache_key(model: str, text: str) -> str:
        payload = f"{model}\n{normalize_text(text)}".encode()
        return hashlib.sha256(payload).hexdigest()

    def get(self, model: str, text: str) -> list[float] | None:
        key = self.cache_key(model, text)
        row = self._conn.execute("SELECT vector, dimension FROM embeddings WHERE cache_key = ?", (key,)).fetchone()
        if row is None:
            return None
        blob, dimension = row
        return np.frombuffer(blob, dtype="float32").reshape(dimension).tolist()

    def put(self, model: str, text: str, vector: list[float]) -> None:
        key = self.cache_key(model, text)
        array = np.asarray(vector, dtype="float32")
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (cache_key, model, vector, dimension) VALUES (?, ?, ?, ?)",
            (key, model, array.tobytes(), array.shape[0]),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def _cache_enabled() -> bool:
    value = os.getenv("RAGLAB_EMBEDDING_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no"}


def get_embedding_cache() -> EmbeddingCache | None:
    """Return a fresh cache connection, or ``None`` if the cache is disabled.

    Deliberately not a process-level singleton: ``create_embeddings()`` opens
    one of these per *batch* call, not per embedded text, so connection
    overhead is negligible — and a fresh connection per call means no shared
    mutable state to reset between tests or between differently-configured
    runs in the same process (see ``RAGLAB_EMBEDDING_CACHE``/
    ``RAGLAB_EMBEDDING_CACHE_DIR``, both read fresh on every call).
    """
    if not _cache_enabled():
        return None
    directory = os.getenv("RAGLAB_EMBEDDING_CACHE_DIR", DEFAULT_CACHE_DIR)
    return EmbeddingCache(Path(directory) / CACHE_FILENAME)
