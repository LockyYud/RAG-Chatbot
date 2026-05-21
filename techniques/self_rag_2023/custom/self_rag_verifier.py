from __future__ import annotations

import json
import re

from raglab.core.interfaces import BaseVerifier
from raglab.core.schema import BuiltContext, RAGAnswer, VerificationReport
from raglab.providers.openai_compatible import OpenAICompatibleClient


class SelfRAGCritiqueVerifier(BaseVerifier):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 350,
        **_: object,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAICompatibleClient()
        self.last_metadata: dict = {}

    def verify(self, answer: RAGAnswer, context: BuiltContext) -> VerificationReport:
        completion = self.client.create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG verifier. Judge whether ANSWER is fully supported by CONTEXT. "
                        "Return only JSON with keys: grounded (boolean), citation_coverage (number 0-1), "
                        "unsupported_citations (array of strings), notes (array of strings)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{answer.query}\n\n"
                        f"ANSWER:\n{answer.answer}\n\n"
                        f"ANSWER_CITATIONS:\n{json.dumps(answer.citations, ensure_ascii=False)}\n\n"
                        f"CONTEXT:\n{context.text}"
                    ),
                },
            ],
        )
        payload = _parse_json_object(completion.text)
        self.last_metadata = {
            "method": "self_rag_critique",
            "model": self.model,
            "usage": completion.usage,
            "latency_ms": completion.latency_ms,
            "estimated_cost": completion.estimated_cost,
        }
        grounded = bool(payload.get("grounded", False))
        citation_coverage = float(payload.get("citation_coverage", 0.0))
        unsupported = [str(item) for item in payload.get("unsupported_citations", [])]
        notes = [str(item) for item in payload.get("notes", [])]
        notes.append(f"self_rag_critique_model={self.model}")
        return VerificationReport(
            grounded=grounded,
            citation_coverage=round(max(0.0, min(1.0, citation_coverage)), 6),
            evidence_count=len(context.results),
            unsupported_citations=unsupported,
            notes=notes,
        )


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "grounded": False,
                "citation_coverage": 0.0,
                "unsupported_citations": [],
                "notes": ["verifier did not return parseable JSON"],
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "grounded": False,
                "citation_coverage": 0.0,
                "unsupported_citations": [],
                "notes": ["verifier returned malformed JSON"],
            }
