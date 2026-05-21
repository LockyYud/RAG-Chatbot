from __future__ import annotations

from raglab.core.interfaces import BaseGenerator
from raglab.core.schema import BuiltContext, RAGAnswer
from raglab.providers.openai_compatible import OpenAICompatibleClient


class OpenAIChatGenerator(BaseGenerator):
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 600,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
        **_: object,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAICompatibleClient(api_key_env=api_key_env, base_url_env=base_url_env, base_url=base_url)

    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        if not context.results:
            return RAGAnswer(query=query, answer="Không tìm thấy đủ bằng chứng trong tài liệu.", contexts=[])

        completion = self.client.create_chat_completion(
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
            result.metadata.get("citation", result.chunk_id)
            for citation_id, result in context.citation_map.items()
            if f"[{citation_id}]" in completion.text
        ]
        if not citations and context.results:
            citations = [context.results[0].metadata.get("citation", context.results[0].chunk_id)]
        return RAGAnswer(
            query=query,
            answer=completion.text,
            contexts=context.results,
            citations=citations,
            metadata={
                "mode": "openai_chat",
                "model": self.model,
                "usage": completion.usage,
                "llm_latency_ms": completion.latency_ms,
                "estimated_llm_cost": completion.estimated_cost,
            },
        )
