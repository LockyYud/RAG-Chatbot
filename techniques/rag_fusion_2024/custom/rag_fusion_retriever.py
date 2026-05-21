from __future__ import annotations

import re

from raglab.core.interfaces import BaseRetriever
from raglab.core.schema import IndexedNode, RetrievalResult
from raglab.core.text import dense_cosine, token_count
from raglab.indexing.embeddings import OpenAIEmbedder
from raglab.providers.openai_compatible import OpenAICompatibleClient


class RAGFusionRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[IndexedNode],
        embedding_model: str = "text-embedding-3-small",
        generator_model: str = "gpt-4.1-mini",
        queries: int = 4,
        per_query_top_k: int = 8,
        rrf_k: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 300,
        **_: object,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError("RAG-Fusion requires OpenAI-compatible embeddings saved during ingest.")
        self.nodes = nodes
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.queries = queries
        self.per_query_top_k = per_query_top_k
        self.rrf_k = rrf_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedder = OpenAIEmbedder(model=embedding_model)
        self.client = OpenAICompatibleClient()
        self.last_metadata: dict = {}

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        generated_queries, generation_metadata = self._generate_queries(query)
        all_queries = [query, *generated_queries]
        fused_scores: dict[str, float] = {}
        best_node: dict[str, IndexedNode] = {}
        source_queries: dict[str, list[str]] = {}

        for expanded_query in all_queries:
            query_vector = self.embedder.embed_texts([expanded_query])[0]
            ranked = sorted(
                ((node, dense_cosine(query_vector, node.embedding or [])) for node in self.nodes),
                key=lambda item: item[1],
                reverse=True,
            )[: self.per_query_top_k]
            for rank, (node, _) in enumerate(ranked, start=1):
                fused_scores[node.node_id] = fused_scores.get(node.node_id, 0.0) + 1.0 / (self.rrf_k + rank)
                best_node[node.node_id] = node
                source_queries.setdefault(node.node_id, []).append(expanded_query)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        results: list[RetrievalResult] = []
        for rank, node_id in enumerate(ranked_ids, start=1):
            node = best_node[node_id]
            metadata = dict(node.metadata)
            metadata["rag_fusion_queries"] = all_queries
            metadata["rag_fusion_matched_queries"] = source_queries.get(node_id, [])
            results.append(
                RetrievalResult(
                    node_id=node.node_id,
                    chunk_id=node.chunk_id,
                    doc_id=node.doc_id,
                    text=node.text_for_generation,
                    score=float(fused_scores[node_id]),
                    rank=rank,
                    metadata=metadata,
                )
            )
        self.last_metadata = {
            "method": "rag_fusion",
            "queries": all_queries,
            "embedding_input_count": len(all_queries),
            "estimated_embedding_tokens": sum(token_count(item) for item in all_queries),
            **generation_metadata,
        }
        return results

    def _generate_queries(self, query: str) -> tuple[list[str], dict]:
        completion = self.client.create_chat_completion(
            model=self.generator_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate alternative search queries for retrieval. "
                        "Return one query per line. Keep the same language as the user query."
                    ),
                },
                {"role": "user", "content": f"Original query: {query}\nNumber of alternatives: {self.queries}"},
            ],
        )
        queries: list[str] = []
        for line in completion.text.splitlines():
            cleaned = re.sub(r"^[-*\d.)\s]+", "", line).strip()
            if cleaned and cleaned.lower() != query.lower() and cleaned not in queries:
                queries.append(cleaned)
        return queries[: self.queries], {
            "generation_usage": completion.usage,
            "generation_latency_ms": completion.latency_ms,
            "estimated_cost": completion.estimated_cost,
        }
