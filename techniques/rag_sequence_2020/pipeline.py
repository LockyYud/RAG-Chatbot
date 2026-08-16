"""RAG-Sequence — the original RAG paper (Lewis et al., 2020).

Paper: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
       Lewis et al., NeurIPS 2020 — https://arxiv.org/abs/2005.11401

The original paper trained a dense retriever (DPR) jointly with a seq2seq
generator (BART) so the model could attend over multiple retrieved passages.
Here we reproduce only the *system shape* — pretrained dense embeddings for
retrieval + an LLM for generation — which is what most modern frameworks
mean when they say "RAG".  We do not retrain anything.

What's different from Naive RAG
-------------------------------
- **LLM chat generator** for answer synthesis instead of extractive sentence
  picking (configure via ``CHAT_MODEL`` env var).
- Recursive chunker (300 tokens) — better for prose answers than fixed-size
  windows.

Both Naive RAG and RAG-Sequence use ``DenseRetriever`` with the same
``EMBED_MODEL``.  The meaningful difference is the generator: extractive vs.
generative.

Everything else (cleaning, citation context, citation-coverage verifier) is
identical to Naive RAG.  This makes RAG-Sequence the natural "first LLM
upgrade" baseline above Naive RAG.
"""

from __future__ import annotations

import time

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, RAGAnswer
from raglab.indexing.artifacts import load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.indexing.retrievers import DenseRetriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.no_reranker import NoReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import NoEnricher
from raglab.processing.parsers.text_parser import TextParser


class RAGSequencePipeline(BasePipeline):
    """Dense embeddings + LLM generator — the canonical 'modern RAG' setup.

    Best for:
    - Multi-language corpora where TF cosine fails on inflection
    - Open-ended questions needing synthesised prose answers
    - Establishing the LLM-tier baseline (above Naive RAG, below tricks)

    Weak for:
    - Strict cost budgets (paid embeddings on every chunk + every query)
    - Exact-keyword lookup tasks (BM25 in parent_child is cheaper and good)
    """

    id = "rag_sequence_2020"
    name = "RAG-Sequence (Lewis et al., 2020)"
    implementation_level = "paper_inspired"
    query_override_fields = frozenset(
        {"generator_model", "generator_temperature", "generator_max_tokens", "top_k", "max_context_tokens"}
    )

    def __init__(
        self,
        *,
        chunk_size: int = 220,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        generator_model: str = "gpt-4.1-mini",
        generator_temperature: float = 0.0,
        generator_max_tokens: int = 700,
        top_k: int = 5,
        max_context_tokens: int = 2200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.generator_model = generator_model
        self.generator_temperature = generator_temperature
        self.generator_max_tokens = generator_max_tokens
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

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
        nodes = NoEnricher().enrich(chunks)

        # === The key step: compute dense embeddings for every chunk ===
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
        from raglab.providers.llm_client import check_provider_ready

        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest, nodes = self.load_artifact(artifact_path)
        check_provider_ready(self.embedding_model)
        if mode == "full_rag":
            check_provider_ready(self.generator_model)
        vector_store = load_vector_store(artifact_path, nodes)

        started = time.perf_counter()

        # 1. Retrieve: embed query, cosine-search the vector store.
        retriever = DenseRetriever(
            nodes=nodes,
            vector_store=vector_store,
            embedding_model=self.embedding_model,
        )
        retrieved = retriever.retrieve(question, self.top_k)

        # 2. No reranker — RAG-Sequence trusts the dense scores.
        reranked = NoReranker().rerank(question, retrieved, self.top_k)

        # 3. Citation context to keep the generator honest about provenance.
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = ChatGenerator(
                model=self.generator_model,
                temperature=self.generator_temperature,
                max_tokens=self.generator_max_tokens,
            ).generate(question, context)
        verifier = CitationCoverageVerifier()
        verification = (
            verifier.verify(answer, context) if mode == "full_rag" else skipped_verification(len(context.results))
        )

        elapsed_ms = (time.perf_counter() - started) * 1000
        answer.metadata.update(
            build_query_metadata(
                latency_ms=elapsed_ms,
                retrieved=retrieved,
                context=context,
                verification=verification,
                artifact_manifest=manifest,
                retriever_kind="openai_dense",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
