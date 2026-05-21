from __future__ import annotations

from raglab.core.interfaces import BaseReranker
from raglab.core.schema import RetrievalResult


class NoReranker(BaseReranker):
    def __init__(self, **_: object) -> None:
        pass

    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        del query
        reranked = results[:top_k]
        for index, result in enumerate(reranked, start=1):
            result.rank = index
        return reranked
