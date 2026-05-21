from __future__ import annotations

from raglab.core.interfaces import BaseContextBuilder
from raglab.core.schema import BuiltContext, RetrievalResult
from raglab.core.text import token_count


class TopKContextBuilder(BaseContextBuilder):
    def __init__(self, max_tokens: int = 1500, **_: object) -> None:
        self.max_tokens = max_tokens

    def build_context(self, query: str, results: list[RetrievalResult]) -> BuiltContext:
        del query
        selected: list[RetrievalResult] = []
        parts: list[str] = []
        used_tokens = 0
        for result in results:
            count = token_count(result.text)
            if parts and used_tokens + count > self.max_tokens:
                break
            selected.append(result)
            parts.append(result.text)
            used_tokens += count
        return BuiltContext(text="\n\n".join(parts), results=selected, citation_map={}, token_count=used_tokens)
