"""RAG-Fusion — multi-query retrieval with Reciprocal Rank Fusion (Rackauckas, 2024).

Paper: "RAG-Fusion: A New Take on Retrieval-Augmented Generation"
       Rackauckas, 2024 — https://arxiv.org/abs/2402.03367

The key idea
------------
Any single phrasing of a query is just one stab at retrieval.  Different
phrasings hit different chunks.  RAG-Fusion asks the LLM to *generate ``N``
alternative queries*, runs dense retrieval for each (plus the original), and
**fuses** the ranked lists with Reciprocal Rank Fusion (RRF).

RRF aggregates rankings without needing comparable scores between retrievers:
each chunk gets ``1 / (k + rank)`` from every list it appears in, summed up.
A chunk that shows up at rank 3 in every query gets the highest fused score,
even if its raw cosine score varies wildly across queries.

What's different from RAG-Sequence
----------------------------------
Only the **retrieval step** — we run dense retrieval ``Q + 1`` times (once
for each generated query plus the original) and RRF-fuse the results.  The
generator, verifier, and ingest pipeline are identical to RAG-Sequence.
"""

from __future__ import annotations

import re
import time
from typing import Any

from raglab.core.base import BasePipeline
from raglab.core.interfaces import BaseRetriever
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, IndexedNode, RAGAnswer, RetrievalResult
from raglab.core.text import dense_cosine, token_count
from raglab.indexing.artifacts import load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.no_reranker import NoReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import LLMClient, check_provider_ready

# ─── The novelty ─────────────────────────────────────────────────────────────


class RAGFusionRetriever(BaseRetriever):
    """Generate alternative queries, dense-search each, fuse with RRF."""

    def __init__(
        self,
        nodes: list[IndexedNode],
        *,
        embedding_model: str = "text-embedding-3-small",
        generator_model: str = "gpt-4.1-mini",
        queries: int = 4,
        per_query_top_k: int = 8,
        rrf_k: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 300,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError("RAG-Fusion requires embeddings saved during ingest.")
        self.nodes = nodes
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.queries = queries
        self.per_query_top_k = per_query_top_k
        self.rrf_k = rrf_k
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedder = Embedder(model=embedding_model)
        self.client = LLMClient()
        self.last_metadata: dict[str, Any] = {}

    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        # Step 1: LLM generates Q alternative phrasings of the original query.
        generated_queries, gen_meta = self._generate_queries(query)
        all_queries = [query, *generated_queries]

        # Step 2: dense-retrieve per_query_top_k chunks for each query.
        # Step 3: Reciprocal Rank Fusion sums 1/(k + rank) across every list.
        fused_scores: dict[str, float] = {}
        best_node: dict[str, IndexedNode] = {}
        source_queries: dict[str, list[str]] = {}

        for expanded_query in all_queries:
            qv = self.embedder.embed_texts([expanded_query])[0]
            ranked = sorted(
                ((node, dense_cosine(qv, node.embedding or [])) for node in self.nodes),
                key=lambda pair: pair[1],
                reverse=True,
            )[: self.per_query_top_k]
            for rank, (node, _score) in enumerate(ranked, start=1):
                fused_scores[node.node_id] = fused_scores.get(node.node_id, 0.0) + 1.0 / (self.rrf_k + rank)
                best_node[node.node_id] = node
                source_queries.setdefault(node.node_id, []).append(expanded_query)

        # Step 4: rank by fused score and return the top_k.
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]  # type: ignore[arg-type]
        results: list[RetrievalResult] = []
        for rank, node_id in enumerate(ranked_ids, start=1):
            node = best_node[node_id]
            results.append(
                RetrievalResult(
                    node_id=node.node_id,
                    chunk_id=node.chunk_id,
                    doc_id=node.doc_id,
                    text=node.text_for_generation,
                    score=float(fused_scores[node_id]),
                    rank=rank,
                    metadata={
                        **dict(node.metadata),
                        "rag_fusion_queries": all_queries,
                        "rag_fusion_matched_queries": source_queries.get(node_id, []),
                    },
                )
            )

        self.last_metadata = {
            "method": "rag_fusion",
            "queries": all_queries,
            "embedding_input_count": len(all_queries),
            "estimated_embedding_tokens": sum(token_count(item) for item in all_queries),
            **gen_meta,
        }
        return results

    def _generate_queries(self, query: str) -> tuple[list[str], dict[str, Any]]:
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


# ─── Pipeline wiring ─────────────────────────────────────────────────────────


class RAGFusionPipeline(BasePipeline):
    """RAG-Sequence baseline, but retrieval fuses multiple query phrasings.

    Best for:
    - Queries where wording matters a lot (synonyms, paraphrases)
    - High-recall regimes across heterogeneous documents
    - Cases where a single phrasing reliably misses chunks the user wanted

    Weak for:
    - Strict latency / cost budgets (Q+1 embedding calls per query, plus
      the LLM call to generate alternatives)
    - Small corpora where one query already covers everything
    """

    id = "rag_fusion_2024"
    name = "RAG-Fusion — multi-query + RRF (Rackauckas, 2024)"
    implementation_level = "paper_inspired"
    query_override_fields = frozenset(
        {
            "generator_model",
            "fusion_queries",
            "fusion_per_query_top_k",
            "fusion_rrf_k",
            "fusion_temperature",
            "fusion_max_tokens",
            "answer_temperature",
            "answer_max_tokens",
            "top_k",
            "max_context_tokens",
        }
    )

    def __init__(
        self,
        *,
        chunk_size: int = 220,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        generator_model: str = "gpt-4.1-mini",
        fusion_queries: int = 4,
        fusion_per_query_top_k: int = 8,
        fusion_rrf_k: int = 60,
        fusion_temperature: float = 0.0,
        fusion_max_tokens: int = 300,
        answer_temperature: float = 0.0,
        answer_max_tokens: int = 700,
        top_k: int = 5,
        max_context_tokens: int = 2400,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.generator_model = generator_model
        self.fusion_queries = fusion_queries
        self.fusion_per_query_top_k = fusion_per_query_top_k
        self.fusion_rrf_k = fusion_rrf_k
        self.fusion_temperature = fusion_temperature
        self.fusion_max_tokens = fusion_max_tokens
        self.answer_temperature = answer_temperature
        self.answer_max_tokens = answer_max_tokens
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        check_provider_ready(self.embedding_model)
        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        chunks = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap).chunk(blocks)
        nodes = SectionTitleEnricher().enrich(chunks)
        nodes = Embedder(
            model=self.embedding_model,
            batch_size=self.embedding_batch_size,
        ).embed_nodes(nodes)

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
        save_nodes(output_path, nodes, manifest, store_spec={"type": store_backend})
        return manifest

    def query(self, artifact_path: str, question: str, mode: str = "full_rag") -> RAGAnswer:
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest, nodes = self.load_artifact(artifact_path)
        check_provider_ready(self.embedding_model)
        check_provider_ready(self.generator_model)
        _ = load_vector_store(artifact_path, nodes)

        started = time.perf_counter()

        # === The RAG-Fusion step ===
        retriever = RAGFusionRetriever(
            nodes=nodes,
            embedding_model=self.embedding_model,
            generator_model=self.generator_model,
            queries=self.fusion_queries,
            per_query_top_k=self.fusion_per_query_top_k,
            rrf_k=self.fusion_rrf_k,
            temperature=self.fusion_temperature,
            max_tokens=self.fusion_max_tokens,
        )
        retrieved = retriever.retrieve(question, self.top_k)
        # ===========================

        reranked = NoReranker().rerank(question, retrieved, self.top_k)
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = ChatGenerator(
                model=self.generator_model,
                temperature=self.answer_temperature,
                max_tokens=self.answer_max_tokens,
            ).generate(question, context)
        verification = (
            CitationCoverageVerifier().verify(answer, context)
            if mode == "full_rag"
            else skipped_verification(len(context.results))
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer.metadata.update(
            build_query_metadata(
                latency_ms=elapsed_ms,
                retrieved=retrieved,
                context=context,
                verification=verification,
                artifact_manifest=manifest,
                retrieval_runtime=retriever.last_metadata,
                retriever_kind="rag_fusion",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
