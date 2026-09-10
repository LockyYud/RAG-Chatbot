from __future__ import annotations

import math
from collections import Counter

from ragbench.core.interfaces import BaseRetriever, BaseVectorStore
from ragbench.core.schema import IndexedNode, RetrievalResult
from ragbench.core.text import dense_cosine, tokenize
from ragbench.indexing.embeddings import Embedder


class DenseRetriever(BaseRetriever):
    """Retrieve using pre-computed neural embeddings (cosine similarity).

    Nodes must already have ``embedding`` populated — call
    ``Embedder().embed_nodes(nodes)`` during ingest before saving.
    At query time one embedding call is made to embed the question.
    """

    def __init__(
        self,
        nodes: list[IndexedNode],
        vector_store: BaseVectorStore | None = None,
        embedding_model: str | None = None,
        **_: object,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError(
                f"DenseRetriever requires embeddings saved during ingest "
                f"({len(missing)} node(s) have no embedding). "
                f"Call Embedder().embed_nodes(nodes) in ingest() before save_nodes()."
            )
        self.nodes = nodes
        self.vector_store = vector_store
        self.embedder = Embedder(model=embedding_model)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_vector = self.embedder.embed_texts([query])[0]
        # __init__ already guarantees no node has embedding=None when self.nodes is non-empty.
        first_embedding = self.nodes[0].embedding if self.nodes else None
        expected_dimension = len(first_embedding) if first_embedding is not None else len(query_vector)
        if len(query_vector) != expected_dimension:
            raise RuntimeError(
                f"Query embedding dimension {len(query_vector)} does not match artifact dimension "
                f"{expected_dimension} for model '{self.embedder.model}'."
            )
        if self.vector_store is not None:
            scored = self.vector_store.search(query_vector, top_k)
        else:
            scored = [
                (node, dense_cosine(query_vector, node.embedding if node.embedding is not None else []))
                for node in self.nodes
            ]
        return _to_results(scored, top_k)


class BM25Retriever(BaseRetriever):
    def __init__(self, nodes: list[IndexedNode], k1: float = 1.5, b: float = 0.75, **_: object) -> None:
        self.nodes = nodes
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(node.text_for_embedding) for node in nodes]
        self.avgdl = sum(len(document) for document in self.documents) / max(1, len(self.documents))
        self.doc_freq: Counter[str] = Counter()
        # Precomputed once at ingest/load time — retrieve() used to rebuild
        # this same Counter(document) from scratch on every single query,
        # for every document, regardless of how many queries actually ran.
        self.term_frequencies: list[Counter[str]] = []
        for document in self.documents:
            frequencies = Counter(document)
            self.term_frequencies.append(frequencies)
            self.doc_freq.update(frequencies.keys())

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        query_terms = tokenize(query)
        scored = []
        total_docs = len(self.nodes)
        for node, document, frequencies in zip(self.nodes, self.documents, self.term_frequencies, strict=True):
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
    """Combine dense (embedding) and sparse (BM25) retrieval with linear interpolation.

    *alpha* = 1.0 → pure dense, *alpha* = 0.0 → pure BM25.
    Nodes must have embeddings saved during ingest (same requirement as DenseRetriever).
    """

    def __init__(
        self,
        nodes: list[IndexedNode],
        vector_store: BaseVectorStore | None = None,
        alpha: float = 0.5,
        embedding_model: str | None = None,
        **_: object,
    ) -> None:
        self.nodes = nodes
        self.alpha = alpha
        self.dense = DenseRetriever(nodes, vector_store=vector_store, embedding_model=embedding_model)
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


class RRFHybridRetriever(BaseRetriever):
    """Fuse dense + BM25 result lists with Reciprocal Rank Fusion (Cormack et al., 2009).

    Unlike :class:`HybridRetriever` (linear interpolation of normalized scores),
    RRF ignores the raw scores entirely and fuses on *rank* alone::

        RRF_score(d) = Σ_i  1 / (k + rank_i(d))

    This sidesteps the score-scale mismatch between BM25 (unbounded log-TF-IDF)
    and cosine similarity ([-1, 1]) without any per-corpus ``alpha`` tuning —
    which is exactly why RRF is the default fusion in production hybrid search.

    ``k`` defaults to 60 (the value from the original paper); larger ``k``
    flattens the contribution of top ranks, smaller ``k`` sharpens it.

    Nodes must have embeddings saved during ingest (same requirement as
    :class:`DenseRetriever`).
    """

    def __init__(
        self,
        nodes: list[IndexedNode],
        vector_store: BaseVectorStore | None = None,
        k: float = 60.0,
        candidate_k: int | None = None,
        embedding_model: str | None = None,
        **_: object,
    ) -> None:
        self.nodes = nodes
        self.k = k
        # How many candidates to pull from each sub-retriever before fusing.
        # Defaults to the full corpus so no relevant node is dropped pre-fusion.
        self.candidate_k = candidate_k or len(nodes)
        self.dense = DenseRetriever(nodes, vector_store=vector_store, embedding_model=embedding_model)
        self.bm25 = BM25Retriever(nodes)

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        dense = self.dense.retrieve(query, self.candidate_k)
        sparse = self.bm25.retrieve(query, self.candidate_k)
        fused = reciprocal_rank_fusion([dense, sparse], k=self.k)
        by_id = {node.node_id: node for node in self.nodes}
        scored = [(by_id[node_id], score) for node_id, score in fused if node_id in by_id]
        return _to_results(scored, top_k)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: float = 60.0,
) -> list[tuple[str, float]]:
    """Fuse several ranked result lists into one ``(node_id, rrf_score)`` ranking.

    Pure and side-effect free so it can be unit-tested without a retriever or
    any API call.  A node absent from a list simply contributes nothing for
    that list.  Results are returned sorted by descending RRF score.
    """
    scores: dict[str, float] = {}
    for results in result_lists:
        for result in results:
            scores[result.node_id] = scores.get(result.node_id, 0.0) + 1.0 / (k + result.rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


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
