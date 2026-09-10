"""Cross-encoder reranking — the precision tier of a hybrid retrieval pipeline.

Retrieval (BM25, dense, or their RRF fusion) is tuned for *recall*: pull back a
generous candidate pool.  A cross-encoder is tuned for *precision*: it scores
each ``(query, passage)`` pair *jointly* through full attention, capturing
token-level interaction that bi-encoders (which embed query and passage
independently) cannot.  Running it over a small candidate pool (top-20 to
top-50) reorders the survivors far more accurately than the first-stage scores.

Two backends are available, chosen with ``backend=``:

- ``"local"`` (default): a `sentence-transformers` CrossEncoder loaded once in
  ``load()``, when the optional ``rerank`` extra is installed.
- ``"api"``: a hosted rerank endpoint via litellm (``model`` is e.g.
  ``cohere/rerank-english-v3.0`` or
  ``jina_ai/jina-reranker-v2-base-multilingual``) — no local model
  download/GPU needed, at the cost of a network call per query.

Benchmark mode is strict by default: a missing dependency/model (local) or a
failed API call (api) is an error. Demo queries may explicitly set
``strict=False`` to fall back to :class:`LexicalOverlapReranker`; the
effective implementation is then recorded in result metadata.

Install the local model::

    pip install ".[rerank]"
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

from ragbench.core.interfaces import BaseReranker
from ragbench.core.schema import RetrievalResult
from ragbench.inference.rerankers.lexical_overlap import LexicalOverlapReranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Re-score candidates with a cross-encoder; fallback requires ``strict=False``.

    Parameters
    ----------
    model:
        A `sentence-transformers` CrossEncoder model id (``backend="local"``,
        the default is a small, fast MS MARCO model that runs on CPU) or a
        litellm rerank model id (``backend="api"``).
    backend:
        ``"local"`` loads *model* once here via `sentence-transformers`.
        ``"api"`` calls a hosted rerank endpoint per ``rerank()`` call instead
        — nothing is loaded at construction time, so construction never fails
        for a bad API model id; only the first real call can.
    fallback_weight:
        ``weight`` passed to :class:`LexicalOverlapReranker` when the
        cross-encoder model cannot be loaded/called.
    timeout:
        HTTP timeout in seconds for each ``backend="api"`` call.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fallback_weight: float = 0.25,
        strict: bool = True,
        backend: Literal["local", "api"] = "local",
        timeout: float = 60.0,
        **_: object,
    ) -> None:
        self.model = model
        self.fallback_weight = fallback_weight
        self.strict = strict
        self.backend = backend
        self.timeout = timeout
        self._load_error: Exception | None = None
        if backend == "api":
            # Nothing to load: readiness is a runtime property of each call,
            # not something known at construction time (unlike a local model,
            # which either loads now or never will).
            self._encoder = None
            return
        try:
            self._encoder = self._load_encoder(model)
        except Exception as exc:  # model/dependency errors are surfaced in strict mode
            self._encoder = None
            self._load_error = exc
        if self._encoder is None and self.strict:
            detail = f": {self._load_error}" if self._load_error else ""
            raise RuntimeError(
                f"Cross-encoder '{model}' is unavailable in strict mode{detail}. "
                "Install the rerank extra and ensure the model is available."
            ) from self._load_error

    @staticmethod
    def _load_encoder(model: str) -> Any:
        """Return a loaded CrossEncoder, or ``None`` if the extra is unavailable."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        return CrossEncoder(model)

    @property
    def available(self) -> bool:
        """True when a real cross-encoder is expected to run (not lexical fallback).

        For ``backend="local"`` this reflects whether the model actually
        loaded. For ``backend="api"`` there is nothing to have failed yet, so
        this is always ``True``; a call that fails at runtime either raises
        (``strict=True``) or is tagged ``lexical_overlap_fallback`` per-result
        (``strict=False``) — both are visible without this flag.
        """
        return self.backend == "api" or self._encoder is not None

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if not results:
            return []
        if self.backend == "api":
            return self._rerank_api(query, results, top_k)
        if self._encoder is None:
            reranked = LexicalOverlapReranker(weight=self.fallback_weight).rerank(query, results, top_k)
            return [_tag(result, "lexical_overlap_fallback") for result in reranked]

        pairs = [(query, result.text) for result in results]
        scores = self._encoder.predict(pairs)
        rescored = [
            _tag(replace(result, score=float(score), metadata=dict(result.metadata)), self.model)
            for result, score in zip(results, scores, strict=True)
        ]
        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        return [replace(result, rank=index) for index, result in enumerate(ranked, start=1)]

    def _rerank_api(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        from ragbench.providers.llm_client import LLMClient

        try:
            scored = LLMClient(timeout=self.timeout).create_rerank(
                model=self.model,
                query=query,
                documents=[result.text for result in results],
                top_n=len(results),
            )
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"API cross-encoder '{self.model}' call failed: {exc}") from exc
            reranked = LexicalOverlapReranker(weight=self.fallback_weight).rerank(query, results, top_k)
            return [_tag(result, "lexical_overlap_fallback") for result in reranked]

        rescored = [
            _tag(replace(results[index], score=score, metadata=dict(results[index].metadata)), self.model)
            for index, score in scored
        ]
        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        return [replace(result, rank=index) for index, result in enumerate(ranked, start=1)]


def _tag(result: RetrievalResult, reranker: str) -> RetrievalResult:
    metadata = dict(result.metadata)
    metadata["reranker"] = reranker
    return replace(result, metadata=metadata)


def effective_reranker_name(reranker: CrossEncoderReranker, reranked: list[RetrievalResult]) -> str:
    """Which reranker actually ran for *this* ``rerank()`` call.

    Read from the returned results' own ``"reranker"`` tag, never from
    ``reranker.available`` — that flag is static (set once, at construction)
    and cannot reflect a ``backend="api"`` call that failed *this* time and
    fell back to lexical overlap, while other calls on the same shared
    instance succeed. It also deliberately avoids stashing the last outcome
    on ``self``: under ``--concurrency`` multiple queries share one reranker
    instance (see the HyDE/RAG-Fusion ``last_metadata`` race in
    ``docs/adding_techniques.md``), so the only race-free source of "what
    happened this call" is the call's own return value.
    """
    if reranked:
        return str(reranked[0].metadata.get("reranker", reranker.model))
    return reranker.model if reranker.available else "lexical_overlap_fallback"
