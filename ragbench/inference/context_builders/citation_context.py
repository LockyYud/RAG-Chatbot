from __future__ import annotations

from dataclasses import replace

from ragbench.core.interfaces import BaseContextBuilder
from ragbench.core.schema import BuiltContext, RetrievalResult
from ragbench.core.text import token_count


class CitationContextBuilder(BaseContextBuilder):
    def __init__(self, max_tokens: int = 1800, include_metadata: bool = True, **_: object) -> None:
        self.max_tokens = max_tokens
        self.include_metadata = include_metadata

    def build_context(self, query: str, results: list[RetrievalResult]) -> BuiltContext:
        del query
        selected: list[RetrievalResult] = []
        parts: list[str] = []
        citation_map: dict[str, RetrievalResult] = {}
        used_tokens = 0
        for index, result in enumerate(results, start=1):
            citation_id = f"C{index}"
            citation = _source_label(result)
            header = f"[{citation_id}] {citation}"
            if self.include_metadata and result.metadata.get("section_title"):
                header = f"{header} | {result.metadata['section_title']}"
            text = f"{header}\n{result.text}"
            count = token_count(text)
            if parts and used_tokens + count > self.max_tokens:
                break
            metadata = dict(result.metadata)
            metadata["citation_id"] = citation_id
            metadata["citation"] = citation
            cited_result = replace(result, metadata=metadata)
            selected.append(cited_result)
            citation_map[citation_id] = cited_result
            parts.append(text)
            used_tokens += count
        return BuiltContext(
            text="\n\n".join(parts),
            results=selected,
            citation_map=citation_map,
            token_count=used_tokens,
        )


def _source_label(result: RetrievalResult) -> str:
    return result.doc_id
