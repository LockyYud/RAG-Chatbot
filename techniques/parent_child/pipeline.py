"""Parent-child chunking — recall via small chunks, context via big parents.

The intuition
-------------
Embedding similarity is sharpest on **short** chunks (a 100-token paragraph is
either about your topic or it isn't), but a generator answers better when given
**long** context (the surrounding section grounds the answer).  Parent-child
chunking gets both: we index small "child" chunks for retrieval, but each
child carries a pointer to its larger "parent" section.  At query time we
retrieve children and feed the parent text to the generator.

The only departure from Naive RAG is the chunker.  Retrieval uses BM25 here
because heading-aware sections often share keywords with the query — sparse
matching is competitive without paying for embeddings.

Background reading: this is the "small-to-big retrieval" pattern popularised
by LlamaIndex / LangChain in 2023.  Not a paper per se, but a robust default
for heading-structured documents (policies, legal docs, technical manuals).
"""

from __future__ import annotations

import time

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, RAGAnswer
from raglab.indexing.artifacts import save_nodes
from raglab.indexing.retrievers import BM25Retriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.extractive import CitationExtractiveGenerator
from raglab.inference.rerankers.lexical_overlap import LexicalOverlapReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.parent_child import ParentChildChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser


class ParentChildPipeline(BasePipeline):
    """Small chunks for retrieval, big sections for generation.

    Best for:
    - Heading-structured corpora (policies, legal docs, manuals)
    - Mid-length questions that need surrounding context
    - Budget-conscious setups (BM25, no embeddings needed)

    Weak for:
    - Flat documents with no headings (chunker degrades to one big section)
    - Vocabulary-mismatched queries (HyDE handles those better)
    """

    id = "parent_child"
    name = "Parent-Child Chunking (small-to-big retrieval)"
    implementation_level = "production_pattern"
    query_override_fields = frozenset({"top_k", "rerank_top_k", "rerank_weight", "max_context_tokens"})
    _retriever: BM25Retriever

    def __init__(
        self,
        *,
        child_size: int = 100,
        child_overlap: int = 15,
        top_k: int = 6,
        rerank_top_k: int = 4,
        rerank_weight: float = 0.25,
        max_context_tokens: int = 1800,
    ) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_weight = rerank_weight
        self.max_context_tokens = max_context_tokens

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        # Standard processing — only the chunker is paper-specific.
        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        # === The novelty: split sections into small children, keep parent text ===
        chunks = ParentChildChunker(
            child_size=self.child_size,
            child_overlap=self.child_overlap,
        ).chunk(blocks)

        # Section-title enrichment prefixes each child with its heading — boosts
        # BM25 recall on heading keywords.
        nodes = SectionTitleEnricher().enrich(chunks)

        manifest = build_ingest_manifest(
            pipeline_id=self.id,
            pipeline_name=self.name,
            input_path=input_path,
            blocks=blocks,
            chunks=chunks,
            nodes=nodes,
            pipeline_config=self.resolved_config(),
            implementation_level=self.implementation_level,
        )
        save_nodes(output_path, nodes, manifest)
        return manifest

    def load(self, artifact_path: str) -> None:
        manifest, nodes = self.load_artifact(artifact_path)
        # BM25 term stats are built once here instead of once per query.
        self._retriever = BM25Retriever(nodes=nodes)
        self._mark_loaded(artifact_path, manifest, nodes)

    def query(self, question: str, mode: str = "full_rag") -> RAGAnswer:
        self._require_loaded()
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest = self._manifest

        started = time.perf_counter()

        # 1. Retrieve children with BM25 (cheap, no embeddings, strong on headings).
        retrieved = self._retriever.retrieve(question, self.top_k)

        # 2. Lexical-overlap rerank — boosts results that share rare terms
        #    with the query, a cheap stand-in for cross-encoder reranking.
        reranked = LexicalOverlapReranker(weight=self.rerank_weight).rerank(question, retrieved, self.rerank_top_k)

        # 3. Build context.  CitationContextBuilder uses each node's
        #    ``parent_text`` metadata (set by ParentChildChunker) so the
        #    generator sees the *parent* section, not just the child snippet.
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = CitationExtractiveGenerator().generate(question, context)
        else:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")

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
                retriever_kind="bm25",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
