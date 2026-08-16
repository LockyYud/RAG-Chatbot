"""BM25 + dense hybrid retrieval with RRF fusion and cross-encoder reranking.

Pattern, not a single paper
---------------------------
This is the production hybrid baseline every fancier RAG technique should have
to beat.  It composes three well-established ideas:

1. **BM25** (Robertson & Zaragoza, 2009) — sparse lexical matching.  Reliable on
   exact terms: codes, identifiers, rare named entities, dates.
2. **Dense retrieval** (Karpukhin et al., 2020, DPR) — embedding cosine search.
   Catches paraphrase and semantic similarity that BM25 misses.
3. **Reciprocal Rank Fusion** (Cormack et al., 2009) — fuse the two ranked lists
   on *rank*, not raw score, so the BM25/cosine scale mismatch never matters and
   there is no per-corpus ``alpha`` to tune.
4. **Cross-encoder reranking** (Nogueira & Cho, 2019) — re-score the fused
   candidate pool with a model that reads ``(query, passage)`` jointly.  Big
   precision win for a few hundred ms; only run over a small pool.

Why it belongs in the lab
-------------------------
No single retriever wins across corpora (BEIR, Thakur et al. 2021): BM25 leads
on lexical/zero-shot sets, dense leads where it was trained.  Hybrid + rerank is
the strong default — and a fair yardstick.  Comparing HyDE or RAG-Fusion against
naive single-retriever baselines flatters them; comparing against *this* tells
you whether the trick actually earns its cost.

Implementation notes
---------------------
- Requires embeddings (same as ``rag_sequence_2020`` / ``naive_rag``): the dense
  half of the hybrid needs vectors saved at ingest.
- The cross-encoder uses ``sentence-transformers`` when the ``rerank`` extra is
  installed; otherwise it degrades to lexical-overlap reranking so the technique
  still runs offline.  Check ``retrieval_runtime`` / per-result metadata to see
  which path ran.
"""

from __future__ import annotations

import time

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, RAGAnswer
from raglab.indexing.artifacts import default_store_backend, load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.indexing.retrievers import RRFHybridRetriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.cross_encoder import CrossEncoderReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import capture_provider_usage


class BM25HybridRerankPipeline(BasePipeline):
    """Hybrid (BM25 + dense) retrieval, RRF fusion, cross-encoder rerank.

    Best for:
    - Mixed query distributions (some keyword-exact, some paraphrased)
    - Acting as the strong baseline new techniques must beat
    - Corpora with rare named entities / identifiers BM25 alone nails

    Weak for:
    - Strict latency budgets (two retrievers + a reranker per query)
    - Tiny homogeneous corpora where BM25 alone already saturates recall
    """

    id = "bm25_hybrid_rerank"
    name = "BM25 + Dense Hybrid with RRF and Cross-Encoder Reranking"
    implementation_level = "production_pattern"
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

        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        chunks = RecursiveChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap).chunk(blocks)
        # Section-title enrichment prefixes each chunk with its heading — helps
        # the BM25 half of the hybrid match heading keywords.
        nodes = SectionTitleEnricher().enrich(chunks)

        # Dense half of the hybrid needs vectors saved at ingest.
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

        check_provider_ready(self.embedding_model)
        manifest, nodes = self.load_artifact(artifact_path)
        vector_store = load_vector_store(artifact_path, nodes)
        # 1. Hybrid retrieve: BM25 + dense, fused by RRF (rank-based, no alpha).
        self._retriever = RRFHybridRetriever(
            nodes=nodes,
            vector_store=vector_store,
            k=self.rrf_k,
            candidate_k=self.candidate_k,
            embedding_model=self.embedding_model,
        )
        # 2. Cross-encoder rerank the fused pool for precision (lexical fallback
        #    when sentence-transformers is not installed). Loading the model
        #    here instead of per-query is the main win: strict-mode CI/eval
        #    would otherwise reload a sentence-transformers model per question.
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

        # 3. Citation-aware context for honest provenance.
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
            "retriever": "bm25_dense_rrf",
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
                retriever_kind="bm25_dense_rrf",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
