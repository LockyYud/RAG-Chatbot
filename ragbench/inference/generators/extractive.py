from __future__ import annotations

from ragbench.core.interfaces import BaseGenerator
from ragbench.core.schema import BuiltContext, Citation, RAGAnswer
from ragbench.core.text import first_relevant_sentence


class ExtractiveGenerator(BaseGenerator):
    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(
                query=query,
                answer="Không tìm thấy đủ bằng chứng trong tài liệu.",
                contexts=[],
                abstained=True,
            )
        best = context.results[0]
        answer = first_relevant_sentence(best.text, query)
        return RAGAnswer(query=query, answer=answer, contexts=context.results, metadata={"mode": "extractive"})


class CitationExtractiveGenerator(BaseGenerator):
    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(
                query=query,
                answer="Không tìm thấy đủ bằng chứng trong tài liệu.",
                contexts=[],
                citations=[],
                abstained=True,
            )
        best = context.results[0]
        citation_id = best.metadata.get("citation_id", "C1")
        answer = first_relevant_sentence(best.text, query)
        # The answer *is* an extracted substring of best.text (not
        # paraphrased), so — unlike ChatGenerator's free-text answer — its
        # exact supporting span is knowable, not just its source chunk.
        span_start = best.text.find(answer)
        start_char = span_start if span_start >= 0 else None
        end_char = span_start + len(answer) if span_start >= 0 else None
        citation = Citation(
            citation_id=citation_id,
            doc_id=best.doc_id,
            chunk_id=best.chunk_id,
            start_char=start_char,
            end_char=end_char,
        )
        return RAGAnswer(
            query=query,
            answer=f"{answer} [{citation_id}]",
            contexts=context.results,
            citations=[citation],
            metadata={"mode": "citation_extractive", "citation_ids": [citation_id]},
        )
