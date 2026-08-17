"""Agentic RAG (A-RAG-inspired, 2026) — the LLM decides how to retrieve.

Source / inspiration
--------------------
A-RAG: "Scaling Agentic Retrieval-Augmented Generation via Hierarchical
Retrieval Interfaces" (arXiv:2602.03442, 2026) and the System-1/System-2
reasoning-RAG survey (arXiv:2506.10408, 2025).

The 2026 shift
--------------
Static ``retrieve → generate`` is giving way to an agent that interleaves
reasoning and retrieval: it chooses *which* tool (keyword / semantic / hybrid /
chunk-read), *what* to query, and *when* it has enough evidence to answer.  This
is the inference-time, **training-free** version of that idea — no RL (unlike
Search-R1 / AutoSearch) — so it runs on any chat model.

How it reuses the rest of the lab
---------------------------------
The agent's *tools are the retrievers this repo already has*:
  - ``keyword``     → ``BM25Retriever``
  - ``semantic``    → ``DenseRetriever``
  - ``hybrid``      → ``RRFHybridRetriever`` (the bm25_hybrid_rerank backbone)
  - ``chunk_read``  → expand a known node to its full parent section
After the loop, evidence is reranked with the same ``CrossEncoderReranker`` and
the answer is synthesised by the standard ``ChatGenerator`` over a citation
context — so retrieval/citation metrics stay comparable to the other techniques.
The agentic novelty lives in *retrieval control flow*, and the loop emits a
structured trace (steps, tools, subqueries, evidence) for failure analysis.
"""

from __future__ import annotations

import time
from typing import Any, Literal

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, IndexedNode, RAGAnswer, RetrievalResult
from raglab.indexing.artifacts import default_store_backend, load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.indexing.retrievers import BM25Retriever, DenseRetriever, RRFHybridRetriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.controllers.agentic import AgenticRetrievalController, Tool, make_llm_policy
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.cross_encoder import CrossEncoderReranker, effective_reranker_name
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import capture_provider_usage


class AgenticRAGPipeline(BasePipeline):
    """LLM-driven multi-step retrieval over keyword/semantic/hybrid/chunk-read tools.

    Best for:
    - Multi-hop questions where one retrieval pass is not enough
    - Mixed queries that benefit from choosing the right retriever per step
    - Producing an auditable retrieval trace

    Weak for:
    - Latency/cost-sensitive paths (multiple LLM calls per question)
    - Simple single-fact lookups (a single hybrid pass is cheaper and as good)
    """

    id = "agentic_rag_arag"
    name = "Agentic RAG (A-RAG-inspired, 2026)"
    implementation_level = "paper_inspired"
    query_override_fields = frozenset(
        {
            "agent_model",
            "max_steps",
            "per_tool_top_k",
            "generator_model",
            "generator_temperature",
            "generator_max_tokens",
            "rrf_k",
            "rerank_top_k",
            "reranker_model",
            "reranker_backend",
            "max_context_tokens",
            "allow_fallback",
        }
    )
    _tools: dict[str, Tool]
    _reranker: CrossEncoderReranker

    def __init__(
        self,
        *,
        chunk_size: int = 220,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        agent_model: str = "gpt-4.1-mini",
        max_steps: int = 4,
        per_tool_top_k: int = 5,
        generator_model: str = "gpt-4.1-mini",
        generator_temperature: float = 0.0,
        generator_max_tokens: int = 700,
        rrf_k: float = 60.0,
        rerank_top_k: int = 6,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        reranker_backend: Literal["local", "api"] = "local",
        max_context_tokens: int = 2200,
        allow_fallback: bool = False,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.agent_model = agent_model
        self.max_steps = max_steps
        self.per_tool_top_k = per_tool_top_k
        self.generator_model = generator_model
        self.generator_temperature = generator_temperature
        self.generator_max_tokens = generator_max_tokens
        self.rrf_k = rrf_k
        self.rerank_top_k = rerank_top_k
        self.reranker_model = reranker_model
        self.reranker_backend = reranker_backend
        self.max_context_tokens = max_context_tokens
        self.allow_fallback = allow_fallback

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        from raglab.providers.llm_client import check_provider_ready

        check_provider_ready(self.embedding_model)

        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        chunks = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap).chunk(blocks)
        nodes = SectionTitleEnricher().enrich(chunks)
        with capture_provider_usage() as embedding_usage:
            nodes = Embedder(
                model=self.embedding_model,
                batch_size=self.embedding_batch_size,
            ).embed_nodes(nodes)

        embedding_spec = {"type": "dense", "model": self.embedding_model}
        store_backend = default_store_backend(len(nodes), has_embeddings=True)
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
        manifest["extra"]["embedding_usage"] = embedding_usage.to_dict()
        save_nodes(output_path, nodes, manifest, store_spec={"type": store_backend})
        return manifest

    def load(self, artifact_path: str) -> None:
        from raglab.providers.llm_client import check_provider_ready

        # The agent itself is an LLM, so a chat model is required in both modes.
        check_provider_ready(self.embedding_model)
        check_provider_ready(self.agent_model)
        check_provider_ready(self.reranker_model)
        manifest, nodes = self.load_artifact(artifact_path)
        vector_store = load_vector_store(artifact_path, nodes)
        self._tools = self._build_tools(nodes, vector_store)
        # Shared cross-encoder loaded once — reused across the agent's evidence
        # rerank on every query, not reloaded per question.
        self._reranker = CrossEncoderReranker(
            model=self.reranker_model, strict=not self.allow_fallback, backend=self.reranker_backend
        )
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

        # A fresh policy/controller per query: the policy closure accumulates
        # LLM call/cost stats across the steps of *this* question's agent loop
        # (see agent.runtime below), so it must not be shared across queries.
        policy = make_llm_policy(self.agent_model)
        controller = AgenticRetrievalController(
            tools=self._tools,
            policy=policy,
            max_steps=self.max_steps,
            per_tool_top_k=self.per_tool_top_k,
        )
        run = controller.run(question)

        # Rerank the agent-gathered evidence with the shared cross-encoder.
        reranker = self._reranker
        reranked = reranker.rerank(question, run.evidence, self.rerank_top_k)
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

        # Surface the agent's decision trace for failure analysis.
        answer.metadata["agent"] = {
            "steps": len(run.trace),
            "tool_calls": run.tool_calls,
            "stopped_reason": run.stopped_reason,
            "trace": run.trace,
            "runtime": getattr(policy, "runtime", {}),
        }
        answer.metadata["components"] = {
            "retriever": "agentic_multi_tool",
            "requested_reranker": self.reranker_model,
            "effective_reranker": effective_reranker_name(reranker, reranked),
            "generator": self.generator_model if mode == "full_rag" else None,
            "verifier": "citation_coverage" if mode == "full_rag" else None,
        }
        agent_runtime = answer.metadata["agent"]["runtime"]
        answer.metadata["auxiliary_llm_cost"] = float(agent_runtime.get("estimated_cost", 0.0))
        answer.metadata["auxiliary_llm_usage"] = agent_runtime.get("usage", {})

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer.metadata.update(
            build_query_metadata(
                latency_ms=elapsed_ms,
                retrieved=reranked,
                context=context,
                verification=verification,
                artifact_manifest=manifest,
                retriever_kind="agentic_arag",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer

    def _build_tools(self, nodes: list[IndexedNode], vector_store: Any) -> dict[str, Tool]:
        bm25 = BM25Retriever(nodes=nodes)
        dense = DenseRetriever(nodes=nodes, vector_store=vector_store, embedding_model=self.embedding_model)
        hybrid = RRFHybridRetriever(
            nodes=nodes, vector_store=vector_store, k=self.rrf_k, embedding_model=self.embedding_model
        )
        by_id = {node.node_id: node for node in nodes}

        def chunk_read(query: str, top_k: int) -> list[RetrievalResult]:
            # ``query`` is a node_id; return that node expanded to its parent text.
            node = by_id.get(query.strip())
            if node is None:
                return []
            return [
                RetrievalResult(
                    node_id=node.node_id,
                    chunk_id=node.chunk_id,
                    doc_id=node.doc_id,
                    text=node.text_for_generation,
                    score=1.0,
                    rank=1,
                    metadata=dict(node.metadata),
                )
            ]

        return {
            "keyword": bm25.retrieve,
            "semantic": dense.retrieve,
            "hybrid": hybrid.retrieve,
            "chunk_read": chunk_read,
        }
