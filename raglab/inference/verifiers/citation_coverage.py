from __future__ import annotations

from raglab.core.interfaces import BaseVerifier
from raglab.core.schema import BuiltContext, RAGAnswer, VerificationReport


class CitationCoverageVerifier(BaseVerifier):
    def __init__(self, require_citations: bool = True, **_: object) -> None:
        self.require_citations = require_citations

    def verify(self, answer: RAGAnswer, context: BuiltContext) -> VerificationReport:
        supported = {
            result.metadata.get("citation")
            for result in context.results
            if result.metadata.get("citation")
        }
        unsupported = [citation for citation in answer.citations if citation not in supported]
        cited_count = len(answer.citations)
        covered_count = cited_count - len(unsupported)
        coverage = covered_count / cited_count if cited_count else 0.0
        has_required_citation = bool(answer.citations) or not self.require_citations
        grounded = has_required_citation and not unsupported and bool(context.results)

        notes: list[str] = []
        if self.require_citations and not answer.citations:
            notes.append("answer has no citations")
        if unsupported:
            notes.append("answer contains citations that are not present in the built context")

        return VerificationReport(
            grounded=grounded,
            citation_coverage=round(coverage, 6),
            evidence_count=len(context.results),
            unsupported_citations=unsupported,
            notes=notes,
        )
