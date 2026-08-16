from __future__ import annotations

from raglab.core.interfaces import BaseGenerator
from raglab.core.schema import BuiltContext, RAGAnswer
from raglab.providers.llm_client import LLMClient, default_chat_model


class ChatGenerator(BaseGenerator):
    """Generate answers using any chat model supported by litellm.

    The model is taken from the ``CHAT_MODEL`` env var by default
    (``gpt-4.1-mini`` if unset).  Pass *model* explicitly to override.

    Example::

        ChatGenerator()                                 # uses CHAT_MODEL env
        ChatGenerator(model="anthropic/claude-haiku-3") # Anthropic
        ChatGenerator(model="ollama/llama3")             # local
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 600,
        **_: object,
    ) -> None:
        self.model = model or default_chat_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = LLMClient()

    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(
                query=query,
                answer="Không tìm thấy đủ bằng chứng trong tài liệu.",
                contexts=[],
                abstained=True,
            )

        completion = self._client.create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý RAG. Chỉ trả lời dựa trên CONTEXT. "
                        "Mỗi claim quan trọng phải có citation dạng [C1], [C2]. "
                        "Nếu CONTEXT không đủ, nói rõ là không đủ bằng chứng."
                    ),
                },
                {
                    "role": "user",
                    "content": f"QUESTION:\n{query}\n\nCONTEXT:\n{context.text}\n\nANSWER:",
                },
            ],
        )
        citations = [
            result.doc_id
            for citation_id, result in context.citation_map.items()
            if f"[{citation_id}]" in completion.text
        ]
        return RAGAnswer(
            query=query,
            answer=completion.text,
            contexts=context.results,
            citations=citations,
            abstained=_looks_like_abstention(completion.text),
            metadata={
                "mode": "chat",
                "model": self.model,
                "usage": completion.usage,
                "llm_latency_ms": completion.latency_ms,
                "estimated_llm_cost": completion.estimated_cost,
            },
        )


def _looks_like_abstention(text: str) -> bool:
    normalized = text.casefold()
    markers = (
        "không đủ bằng chứng",
        "không tìm thấy đủ",
        "không thể trả lời dựa trên",
        "insufficient evidence",
        "cannot answer from the context",
    )
    return any(marker in normalized for marker in markers)
