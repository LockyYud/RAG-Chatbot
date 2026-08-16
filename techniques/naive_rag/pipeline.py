"""Naive RAG — the textbook baseline (no paper).

The flow this file implements, top to bottom, is exactly the baseline every
paper in this lab is compared against:

    Ingest:
        parse text/markdown files  →  clean (Vietnamese normalize + whitespace)
        →  fixed-size token windows  →  embed with OpenAI  →  save nodes + vectors

    Query:
        load nodes  →  embed question  →  cosine similarity over stored vectors
        →  no reranker  →  citation context  →  extractive generation
        →  citation-coverage verification

Every other technique in ``techniques/`` is "Naive RAG plus a tweak":
HyDE swaps the retriever, Self-RAG swaps the verifier, Parent-Child swaps the
chunker, and so on.  Reading this file first makes every other technique
obvious by diff.

Use this as the floor of your benchmark — if a paper does not beat it on
your data, the paper's extra complexity is not worth it for your use case.
"""

from __future__ import annotations

import time

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, RAGAnswer
from raglab.indexing.artifacts import default_store_backend, load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.indexing.retrievers import DenseRetriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.extractive import CitationExtractiveGenerator
from raglab.inference.rerankers.no_reranker import NoReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.fixed_size import FixedSizeChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import NoEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import capture_provider_usage


class NaiveRAGPipeline(BasePipeline):
    """The baseline RAG flow with no clever tricks.

    Good for:
    - Establishing a floor for benchmarks
    - Quick iteration on tiny corpora
    - Documents where simple lexical matching is enough

    Weak for:
    - Long queries with vocabulary mismatch (use HyDE)
    - Tables / heading-heavy docs (use parent_child or heading_aware)
    - Tasks where hallucination detection matters (use self_rag_2023)
    """

    id = "naive_rag"
    name = "Naive RAG (baseline)"
    implementation_level = "baseline"
    query_override_fields = frozenset({"top_k", "rerank_top_k", "max_context_tokens"})
    _retriever: DenseRetriever

    def __init__(
        self,
        *,
        chunk_size: int = 120,
        chunk_overlap: int = 20,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        top_k: int = 5,
        rerank_top_k: int = 3,
        max_context_tokens: int = 800,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.max_context_tokens = max_context_tokens

    # ─── Ingest ────────────────────────────────────────────────────────────

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        from raglab.providers.llm_client import check_provider_ready

        check_provider_ready(self.embedding_model)
        # 1. Parse every text/markdown file into DocumentBlock objects.
        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))

        # 2. Clean: normalize Vietnamese characters then collapse whitespace.
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        # 3. Chunk: fixed-size token windows with overlap.  Cheap and predictable.
        chunks = FixedSizeChunker(
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap,
        ).chunk(blocks)

        # 4. Enrich: no extra processing.  Each chunk becomes one IndexedNode.
        nodes = NoEnricher().enrich(chunks)

        # 5. Embed: call OpenAI once per node and store vectors in each node.
        #    Wrapped in capture_provider_usage() purely to surface embedding
        #    cache hit/miss counts in the manifest — ingest cost wasn't
        #    tracked anywhere before this.
        embedder = Embedder(model=self.embedding_model, batch_size=self.embedding_batch_size)
        with capture_provider_usage() as embedding_usage:
            nodes = embedder.embed_nodes(nodes)

        # 6. Persist nodes (with embeddings) and the manifest. Backend picked by
        #    corpus size: json_memory (vectorized numpy) below the FAISS
        #    threshold, faiss_local above it — see default_store_backend().
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
            embedding_spec={"type": "openai", "model": embedder.model},
            store_backend=store_backend,
        )
        manifest["extra"]["embedding_usage"] = embedding_usage.to_dict()
        save_nodes(output_path, nodes, manifest, store_spec={"type": store_backend})
        return manifest

    # ─── Load / Query ────────────────────────────────────────────────────────

    def load(self, artifact_path: str) -> None:
        from raglab.providers.llm_client import check_provider_ready

        check_provider_ready(self.embedding_model)
        manifest, nodes = self.load_artifact(artifact_path)
        # Built once here (not per query): avoids re-reading nodes.json and
        # re-constructing the retriever for every question in an eval run.
        vector_store = load_vector_store(artifact_path, nodes)
        self._retriever = DenseRetriever(nodes=nodes, vector_store=vector_store, embedding_model=self.embedding_model)
        self._mark_loaded(artifact_path, manifest, nodes)

    def query(self, question: str, mode: str = "full_rag") -> RAGAnswer:
        self._require_loaded()
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest = self._manifest

        started = time.perf_counter()

        # 1. Retrieve: embed the question, then cosine search over stored node vectors.
        retrieved = self._retriever.retrieve(question, self.top_k)

        # 2. Rerank: identity — naive RAG has no learned reranker.
        reranked = NoReranker().rerank(question, retrieved, self.rerank_top_k)

        # 3. Build context: assemble retrieved chunks with [n] citation markers.
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        # 4. Generate: extractive — pick the best supporting sentence and
        #    refuse to answer if no citation can be attached.
        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = CitationExtractiveGenerator().generate(question, context)
        else:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")

        # 5. Verify: every cited claim must be backed by a retrieved chunk.
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
                retriever_kind="dense",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
