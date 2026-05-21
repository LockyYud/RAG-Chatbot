from __future__ import annotations

from raglab.core.interfaces import BaseRetriever
from raglab.core.schema import IndexedNode, RetrievalResult
from raglab.core.text import dense_cosine, mean_dense_vector, token_count
from raglab.indexing.embeddings import OpenAIEmbedder
from raglab.providers.openai_compatible import OpenAICompatibleClient


class HyDERetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[IndexedNode],
        embedding_model: str = "text-embedding-3-small",
        generator_model: str = "gpt-4.1-mini",
        samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 350,
        **_: object,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError("HyDE requires OpenAI-compatible embeddings saved during ingest.")
        self.nodes = nodes
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.samples = max(1, samples)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.client = OpenAICompatibleClient()
        self.last_metadata: dict = {}

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        hypothetical_documents, generation_metadata = self._generate_hypothetical_documents(query)
        hypothetical_vectors = self.embedder.embed_texts(hypothetical_documents)
        query_vector = mean_dense_vector(hypothetical_vectors)
        scored = [(node, dense_cosine(query_vector, node.embedding or [])) for node in self.nodes]
        results = _to_results(scored, top_k)
        for result in results:
            result.metadata["hyde_model"] = self.generator_model
            result.metadata["hyde_samples"] = self.samples
            result.metadata["hyde_hypothetical_documents"] = hypothetical_documents
        self.last_metadata = {
            "method": "hyde",
            "generated_texts": hypothetical_documents,
            "embedding_input_count": len(hypothetical_documents),
            "estimated_embedding_tokens": sum(token_count(document) for document in hypothetical_documents),
            **generation_metadata,
        }
        return results

    def _generate_hypothetical_documents(self, query: str) -> tuple[list[str], dict]:
        documents: list[str] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_latency = 0.0
        total_cost = 0.0
        for sample_index in range(self.samples):
            completion = self.client.create_chat_completion(
                model=self.generator_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate one concise hypothetical document passage that would answer the user's question. "
                            "Do not mention that it is hypothetical. Write in the same language as the question. "
                            "Use a plausible wording that may differ from prior samples."
                        ),
                    },
                    {"role": "user", "content": f"Question: {query}\nSample: {sample_index + 1}"},
                ],
            )
            documents.append(completion.text)
            for key in total_usage:
                total_usage[key] += int(completion.usage.get(key, 0))
            total_latency += completion.latency_ms
            total_cost += completion.estimated_cost
        return documents, {
            "generation_usage": total_usage,
            "generation_latency_ms": round(total_latency, 3),
            "estimated_cost": round(total_cost, 8),
        }


def _to_results(scored: list[tuple[IndexedNode, float]], top_k: int) -> list[RetrievalResult]:
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            node_id=node.node_id,
            chunk_id=node.chunk_id,
            doc_id=node.doc_id,
            text=node.text_for_generation,
            score=float(score),
            rank=rank,
            metadata=dict(node.metadata),
        )
        for rank, (node, score) in enumerate(ranked, start=1)
    ]
