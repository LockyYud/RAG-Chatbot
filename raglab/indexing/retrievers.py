from __future__ import annotations

import math
from collections import Counter

from raglab.core.interfaces import BaseRetriever
from raglab.core.schema import IndexedNode, RetrievalResult
from raglab.core.text import cosine, dense_cosine, term_vector, tokenize
from raglab.indexing.embeddings import OpenAIEmbedder


class DenseRetriever(BaseRetriever):
    def __init__(self, nodes: list[IndexedNode], **_: object) -> None:
        self.nodes = nodes
        self.vectors = {node.node_id: term_vector(node.text_for_embedding) for node in nodes}

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = term_vector(query)
        scored = [(node, cosine(query_vector, self.vectors[node.node_id])) for node in self.nodes]
        return _to_results(scored, top_k)


class BM25Retriever(BaseRetriever):
    def __init__(self, nodes: list[IndexedNode], k1: float = 1.5, b: float = 0.75, **_: object) -> None:
        self.nodes = nodes
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(node.text_for_embedding) for node in nodes]
        self.avgdl = sum(len(document) for document in self.documents) / max(1, len(self.documents))
        self.doc_freq: Counter[str] = Counter()
        for document in self.documents:
            self.doc_freq.update(set(document))

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        scored = []
        total_docs = len(self.nodes)
        for node, document in zip(self.nodes, self.documents, strict=True):
            frequencies = Counter(document)
            score = 0.0
            for term in query_terms:
                if term not in frequencies:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                tf = frequencies[term]
                denominator = tf + self.k1 * (1 - self.b + self.b * len(document) / max(self.avgdl, 1e-9))
                score += idf * tf * (self.k1 + 1) / denominator
            scored.append((node, score))
        return _to_results(scored, top_k)


class HybridRetriever(BaseRetriever):
    def __init__(self, nodes: list[IndexedNode], alpha: float = 0.5, **_: object) -> None:
        self.nodes = nodes
        self.alpha = alpha
        self.dense = DenseRetriever(nodes)
        self.bm25 = BM25Retriever(nodes)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense = self.dense.retrieve(query, len(self.nodes))
        sparse = self.bm25.retrieve(query, len(self.nodes))
        dense_scores = _normalize({result.node_id: result.score for result in dense})
        sparse_scores = _normalize({result.node_id: result.score for result in sparse})
        scored = []
        by_id = {node.node_id: node for node in self.nodes}
        for node_id, node in by_id.items():
            score = self.alpha * dense_scores.get(node_id, 0.0) + (1 - self.alpha) * sparse_scores.get(node_id, 0.0)
            scored.append((node, score))
        return _to_results(scored, top_k)


class OpenAIDenseRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[IndexedNode],
        embedding_model: str = "text-embedding-3-small",
        **_: object,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError(
                "OpenAI dense retrieval requires embeddings saved during ingest. "
                "Use indexing.embedding.type=openai in the pipeline config."
            )
        self.nodes = nodes
        self.embedder = OpenAIEmbedder(model=embedding_model)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self.embedder.embed_texts([query])[0]
        scored = [(node, dense_cosine(query_vector, node.embedding or [])) for node in self.nodes]
        return _to_results(scored, top_k)


class OpenAIHybridRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[IndexedNode],
        alpha: float = 0.5,
        embedding_model: str = "text-embedding-3-small",
        **_: object,
    ) -> None:
        self.nodes = nodes
        self.alpha = alpha
        self.dense = OpenAIDenseRetriever(nodes, embedding_model=embedding_model)
        self.bm25 = BM25Retriever(nodes)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense = self.dense.retrieve(query, len(self.nodes))
        sparse = self.bm25.retrieve(query, len(self.nodes))
        dense_scores = _normalize({result.node_id: result.score for result in dense})
        sparse_scores = _normalize({result.node_id: result.score for result in sparse})
        by_id = {node.node_id: node for node in self.nodes}
        scored = []
        for node_id, node in by_id.items():
            score = self.alpha * dense_scores.get(node_id, 0.0) + (1 - self.alpha) * sparse_scores.get(node_id, 0.0)
            scored.append((node, score))
        return _to_results(scored, top_k)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low = min(scores.values())
    high = max(scores.values())
    if high <= low:
        return {key: 0.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def _to_results(scored: list[tuple[IndexedNode, float]], top_k: int) -> list[RetrievalResult]:
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    results: list[RetrievalResult] = []
    for rank, (node, score) in enumerate(ranked, start=1):
        results.append(
            RetrievalResult(
                node_id=node.node_id,
                chunk_id=node.chunk_id,
                doc_id=node.doc_id,
                text=node.text_for_generation,
                score=float(score),
                rank=rank,
                metadata=dict(node.metadata),
            )
        )
    return results
