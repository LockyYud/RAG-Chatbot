"""Cross-encoder reranking — the precision tier of a hybrid retrieval pipeline.

Retrieval (BM25, dense, or their RRF fusion) is tuned for *recall*: pull back a
generous candidate pool.  A cross-encoder is tuned for *precision*: it scores
each ``(query, passage)`` pair *jointly* through full attention, capturing
token-level interaction that bi-encoders (which embed query and passage
independently) cannot.  Running it over a small candidate pool (top-20 to
top-50) reorders the survivors far more accurately than the first-stage scores.

This reranker uses a `sentence-transformers` CrossEncoder when the optional
``rerank`` extra is installed. Benchmark mode is strict by default: a missing
dependency or model is an error. Demo queries may explicitly set
``strict=False`` to use :class:`LexicalOverlapReranker`; the effective
implementation is then recorded in result metadata.

Install the real model::

    pip install ".[rerank]"
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from raglab.core.interfaces import BaseReranker
from raglab.core.schema import RetrievalResult
from raglab.inference.rerankers.lexical_overlap import LexicalOverlapReranker

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker(BaseReranker):
    """Re-score candidates with a cross-encoder; fallback requires ``strict=False``.

    Parameters
    ----------
    model:
        sentence-transformers CrossEncoder model id.  The default is a small,
        fast MS MARCO model that runs on CPU.
    fallback_weight:
        ``weight`` passed to :class:`LexicalOverlapReranker` when the
        cross-encoder model cannot be loaded.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fallback_weight: float = 0.25,
        strict: bool = True,
        **_: object,
    ) -> None:
        self.model = model
        self.fallback_weight = fallback_weight
        self.strict = strict
        self._load_error: Exception | None = None
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
        """True when the real cross-encoder model loaded successfully."""
        return self._encoder is not None

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        if self._encoder is None:
            reranked = LexicalOverlapReranker(weight=self.fallback_weight).rerank(query, results, top_k)
            return [_tag(result, "lexical_overlap_fallback") for result in reranked]

        if not results:
            return []

        pairs = [(query, result.text) for result in results]
        scores = self._encoder.predict(pairs)
        rescored = [
            _tag(replace(result, score=float(score), metadata=dict(result.metadata)), self.model)
            for result, score in zip(results, scores, strict=True)
        ]
        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        return [replace(result, rank=index) for index, result in enumerate(ranked, start=1)]


def _tag(result: RetrievalResult, reranker: str) -> RetrievalResult:
    metadata = dict(result.metadata)
    metadata["reranker"] = reranker
    return replace(result, metadata=metadata)
