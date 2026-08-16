"""Contextual Retrieval (Anthropic, 2024) on a hybrid + rerank backbone.

Paper / source
--------------
"Introducing Contextual Retrieval", Anthropic, 2024 —
https://www.anthropic.com/news/contextual-retrieval

Core idea
---------
Naive chunking strips the context a chunk needs to be findable.  Before indexing,
an LLM writes a 1-2 sentence snippet situating each chunk inside its full
document; that snippet is prepended to the chunk's *indexing* text.  Since both
the dense embedder and BM25 read ``text_for_embedding`` in this repo, one prepend
yields both "Contextual Embeddings" and "Contextual BM25".

What's different from ``bm25_hybrid_rerank``
--------------------------------------------
*Only* the enricher.  Retrieval (RRF hybrid), reranking (cross-encoder, lexical
fallback), context construction, generation and verification are identical — so a
head-to-head benchmark isolates exactly what contextualization buys.  Anthropic
reports ~49% fewer top-20 retrieval failures (≈67% with reranking) over naive
chunking.

Cost / notes
------------
- One extra LLM call per chunk at ingest time (``context_model``).  Documents are
  truncated to ``max_doc_tokens`` to bound cost; a failed context call degrades
  to the plain chunk rather than aborting ingest.
- Requires embeddings (dense half) and a chat model (context + answer).
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, RAGAnswer
from raglab.indexing.artifacts import load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.indexing.retrievers import RRFHybridRetriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.cross_encoder import CrossEncoderReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.contextual import ContextualEnricher
from raglab.processing.parsers.text_parser import TextParser


class ContextualRetrievalPipeline(BasePipeline):
    """Contextual chunk enrichment + RRF hybrid retrieval + cross-encoder rerank.

    Best for:
    - Corpora of long documents split into small chunks that lose context
    - Queries that need disambiguation a bare chunk can't provide
    - Being a stronger indexing baseline than plain hybrid + rerank

    Weak for:
    - Tight ingest budgets (one LLM call per chunk)
    - Already self-contained chunks (FAQ entries) where context adds little
    """

    id = "contextual_retrieval_2024"
    name = "Contextual Retrieval (Anthropic, 2024)"
    implementation_level = "paper_inspired"
    query_override_fields = frozenset(
        {
            "generator_model",
            "generator_temperature",
            "generator_max_tokens",
            "rrf_k",
            "candidate_k",
            "rerank_top_k",
            "reranker_model",
            "max_context_tokens",
            "allow_fallback",
        }
    )
    _retriever: RRFHybridRetriever
    _reranker: CrossEncoderReranker

    def __init__(
        self,
        *,
        chunk_size: int = 220,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        context_model: str = "gpt-4.1-mini",
        max_doc_tokens: int = 4000,
        generator_model: str = "gpt-4.1-mini",
        generator_temperature: float = 0.0,
        generator_max_tokens: int = 700,
        rrf_k: float = 60.0,
        candidate_k: int = 30,
        rerank_top_k: int = 6,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_context_tokens: int = 2200,
        allow_fallback: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.context_model = context_model
        self.max_doc_tokens = max_doc_tokens
        self.generator_model = generator_model
        self.generator_temperature = generator_temperature
        self.generator_max_tokens = generator_max_tokens
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k
        self.rerank_top_k = rerank_top_k
        self.reranker_model = reranker_model
        self.max_context_tokens = max_context_tokens
        self.allow_fallback = allow_fallback

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        from raglab.providers.llm_client import check_provider_ready

        check_provider_ready(self.embedding_model)
        check_provider_ready(self.context_model)

        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        # Full document text per doc_id — the context the chunks lost.
        documents = self._documents_from_blocks(blocks)

        chunks = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap).chunk(blocks)

        # === The novelty: situate each chunk in its document before indexing ===
        nodes = ContextualEnricher(
            documents=documents,
            context_model=self.context_model,
            max_doc_tokens=self.max_doc_tokens,
        ).enrich(chunks)

        # Dense half of the hybrid needs vectors (computed on contextualized text).
        nodes = Embedder(
            model=self.embedding_model,
            batch_size=self.embedding_batch_size,
        ).embed_nodes(nodes)

        contextualized = sum(1 for node in nodes if node.metadata.get("contextualized"))
        embedding_spec = {"type": "dense", "model": self.embedding_model}
        store_backend = "json_memory"
        manifest = build_ingest_manifest(
            pipeline_id=self.id,
            pipeline_name=self.name,
            input_path=input_path,
            blocks=blocks,
            chunks=chunks,
            nodes=nodes,
            pipeline_config=self.resolved_config(),
            implementation_level=self.implementation_level,
            embedding_spec=embedding_spec,
            store_backend=store_backend,
        )
        manifest["extra"].update({"contextualized_nodes": contextualized, "context_model": self.context_model})
        save_nodes(output_path, nodes, manifest, store_spec={"type": store_backend})
        return manifest

    def load(self, artifact_path: str) -> None:
        from raglab.providers.llm_client import check_provider_ready

        check_provider_ready(self.embedding_model)
        manifest, nodes = self.load_artifact(artifact_path)
        vector_store = load_vector_store(artifact_path, nodes)
        # 1. Hybrid retrieve over the contextualized index (BM25 + dense, RRF fused).
        self._retriever = RRFHybridRetriever(
            nodes=nodes,
            vector_store=vector_store,
            k=self.rrf_k,
            candidate_k=self.candidate_k,
            embedding_model=self.embedding_model,
        )
        # 2. Cross-encoder rerank (lexical fallback when the extra is absent).
        self._reranker = CrossEncoderReranker(model=self.reranker_model, strict=not self.allow_fallback)
        self._mark_loaded(artifact_path, manifest, nodes)

    def query(self, question: str, mode: str = "full_rag") -> RAGAnswer:
        from raglab.providers.llm_client import check_provider_ready

        self._require_loaded()
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest = self._manifest
        if mode == "full_rag":
            check_provider_ready(self.generator_model)

        started = time.perf_counter()

        retriever = self._retriever
        retrieved = retriever.retrieve(question, self.candidate_k)

        reranker = self._reranker
        reranked = reranker.rerank(question, retrieved, self.rerank_top_k)

        # 3. Citation context for honest provenance.
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = ChatGenerator(
                model=self.generator_model,
                temperature=self.generator_temperature,
                max_tokens=self.generator_max_tokens,
            ).generate(question, context)
        verification = (
            CitationCoverageVerifier().verify(answer, context)
            if mode == "full_rag"
            else skipped_verification(len(context.results))
        )
        answer.metadata["components"] = {
            "retriever": "contextual_bm25_dense_rrf",
            "requested_reranker": self.reranker_model,
            "effective_reranker": self.reranker_model if reranker.available else "lexical_overlap_fallback",
            "generator": self.generator_model if mode == "full_rag" else None,
            "verifier": "citation_coverage" if mode == "full_rag" else None,
        }

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer.metadata.update(
            build_query_metadata(
                latency_ms=elapsed_ms,
                retrieved=retrieved,
                context=context,
                verification=verification,
                artifact_manifest=manifest,
                retriever_kind="contextual_bm25_dense_rrf",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer

    @staticmethod
    def _documents_from_blocks(blocks: list[Any]) -> dict[str, str]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for block in blocks:
            grouped[block.doc_id].append(block.text)
        return {doc_id: "\n\n".join(texts) for doc_id, texts in grouped.items()}
