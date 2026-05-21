from __future__ import annotations

from raglab.core.interfaces import BaseGenerator
from raglab.core.schema import BuiltContext, RAGAnswer
from raglab.core.text import first_relevant_sentence


class ExtractiveGenerator(BaseGenerator):
    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(query=query, answer="Không tìm thấy đủ bằng chứng trong tài liệu.", contexts=[])
        best = context.results[0]
        answer = first_relevant_sentence(best.text, query)
        return RAGAnswer(query=query, answer=answer, contexts=context.results, metadata={"mode": "extractive"})


class CitationExtractiveGenerator(BaseGenerator):
    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(query=query, answer="Không tìm thấy đủ bằng chứng trong tài liệu.", contexts=[], citations=[])
        best = context.results[0]
        citation_id = best.metadata.get("citation_id", "C1")
        citation = best.metadata.get("citation", best.chunk_id)
        answer = first_relevant_sentence(best.text, query)
        return RAGAnswer(
            query=query,
            answer=f"{answer} [{citation_id}]",
            contexts=context.results,
            citations=[citation],
            metadata={"mode": "citation_extractive", "citation_ids": [citation_id]},
        )
