from __future__ import annotations

from dataclasses import replace

from raglab.core.interfaces import BaseReranker
from raglab.core.schema import RetrievalResult
from raglab.core.text import tokenize


class LexicalOverlapReranker(BaseReranker):
    def __init__(self, weight: float = 0.35, **_: object) -> None:
        self.weight = weight

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        query_terms = set(tokenize(query))
        rescored: list[RetrievalResult] = []
        for result in results:
            text_terms = set(tokenize(result.text))
            overlap = len(query_terms & text_terms) / max(1, len(query_terms))
            score = (1 - self.weight) * result.score + self.weight * overlap
            rescored.append(replace(result, score=score, metadata=dict(result.metadata)))

        reranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        return [replace(result, rank=index) for index, result in enumerate(reranked, start=1)]
