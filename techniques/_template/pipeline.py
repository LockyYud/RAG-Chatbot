"""Template for adding a new RAG technique.

Copy this directory to ``techniques/<your_technique_id>/`` and follow the
TODO markers below.  When you are done, the new technique is automatically
discovered by ``raglab techniques list``, ``raglab ingest --technique <id>``,
and ``raglab bench`` — no registry to update, no config schema to edit.

Conventions
-----------
- One file (``pipeline.py``) per technique, end-to-end and self-contained.
- Read top to bottom: docstring → (optional) custom retriever/verifier classes
  inline → ``Pipeline`` class wiring everything together.
- Only split paper-specific helpers into ``custom/`` when the file would
  otherwise exceed ~250 lines or the paper modifies several phases at once.
- Always finish ``query()`` by calling :func:`raglab.core.measure.build_query_metadata`
  so benchmarks compare your technique fairly against the others.

Running your new technique::

    raglab ingest --technique <your_id> --input docs/ --output artifacts/
    raglab query  --technique <your_id> --artifact artifacts/<your_id> --query "..."
    raglab eval   --technique <your_id> --artifact artifacts/<your_id> \\
                  --dataset datasets/my_qa.jsonl --output results/
    raglab bench  --techniques naive_rag <your_id> --docs docs/ \\
                  --qa datasets/my_qa.jsonl --output benchmarks/results/

Override paper defaults from the CLI without editing this file::

    raglab ingest --technique <your_id> --input docs/ --output art/ \\
                  --param chunk_size=300 --param top_k=10
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
from raglab.inference.rerankers.no_reranker import NoReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.fixed_size import FixedSizeChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import NoEnricher
from raglab.processing.parsers.text_parser import TextParser

# TODO: If your paper introduces a new retriever / reranker / verifier,
#       define it here as a class inline.  See ``techniques/hyde_2022/pipeline.py``
#       and ``techniques/self_rag_2023/pipeline.py`` for examples.
#
# class MyNovelRetriever(BaseRetriever):
#     def __init__(self, nodes, ...): ...
#     def retrieve(self, query, top_k): ...


class TemplatePipeline(BasePipeline):
    """TODO: Replace this docstring with a 3-paragraph explanation of the paper.

    Best for:
    - ...

    Weak for:
    - ...
    """

    # TODO: set this to match your directory name under techniques/
    id = "technique_id"
    # TODO: paper title or short label shown in ``raglab techniques list``
    name = "TODO: short paper title"
    implementation_level = "paper_inspired"
    query_override_fields = frozenset({"top_k", "max_context_tokens"})

    def __init__(
        self,
        *,
        # TODO: declare paper-defaults here as keyword-only arguments.
        # CLI users can override any of them via ``--param key=value``.
        chunk_size: int = 250,
        chunk_overlap: int = 40,
        top_k: int = 5,
        max_context_tokens: int = 1500,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        # TODO: replace any of these steps with what your paper actually does.
        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        chunks = FixedSizeChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap).chunk(blocks)
        nodes = NoEnricher().enrich(chunks)

        manifest = build_ingest_manifest(
            pipeline_id=self.id,
            pipeline_name=self.name,
            input_path=input_path,
            blocks=blocks,
            chunks=chunks,
            nodes=nodes,
            pipeline_config=self.resolved_config(),
            implementation_level=self.implementation_level,
            # TODO: pass embedding_spec=... + store_backend=... if your paper
            #       uses learned embeddings (see rag_sequence_2020 / hyde_2022).
        )
        save_nodes(output_path, nodes, manifest)
        return manifest

    def query(self, artifact_path: str, question: str, mode: str = "full_rag") -> RAGAnswer:
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest, nodes = self.load_artifact(artifact_path)

        started = time.perf_counter()

        # TODO: swap in your paper-specific retriever / reranker / verifier.
        retriever = BM25Retriever(nodes=nodes)
        retrieved = retriever.retrieve(question, self.top_k)
        reranked = NoReranker().rerank(question, retrieved, self.top_k)
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = CitationExtractiveGenerator().generate(question, context)
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
                retriever_kind="bm25",  # TODO: set to your retriever kind
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
