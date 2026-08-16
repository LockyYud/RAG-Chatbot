"""HyDE — Hypothetical Document Embeddings (Gao et al., 2022).

Paper: "Precise Zero-Shot Dense Retrieval without Relevance Labels"
       Gao, Ma, Lin, Callan — https://arxiv.org/abs/2212.10496

The key idea
------------
Standard dense retrieval embeds the user's query and finds the nearest chunk
by cosine similarity.  This fails when the query is short or uses different
vocabulary than the documents — for example, ``"How do refunds work?"`` lives
in a different region of embedding space than ``"Returns are processed within
14 days of receipt."``

HyDE bridges the gap by asking an LLM to *imagine* an answer to the query
first, then embedding **that hypothetical document** and using it as the
search vector.  The imagined doc shares vocabulary and phrasing with real
answers, so cosine similarity matches the right chunks.  Hallucinated details
in the imagined doc do not matter — we only use its vector, not its content.

We generate ``samples`` hypothetical documents and mean-pool their embeddings,
which cancels out per-sample hallucination noise.

What's different from RAG-Sequence
----------------------------------
Only the **retrieval step**.  Everything else (chunking, embeddings, LLM
generator, citation verifier) is identical to ``rag_sequence_2020``.  Scan
``HyDEPipeline.query`` below — the only HyDE-specific code lives inside
``HyDERetriever.retrieve``.
"""

from __future__ import annotations

import time
from typing import Any

from raglab.core.base import BasePipeline
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, IndexedNode, RAGAnswer, RetrievalResult
from raglab.core.text import dense_cosine, mean_dense_vector, token_count
from raglab.indexing.artifacts import default_store_backend, load_vector_store, save_nodes
from raglab.indexing.embeddings import Embedder
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.no_reranker import NoReranker
from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
from raglab.processing.chunkers.recursive import RecursiveChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import LLMClient, capture_provider_usage, check_provider_ready

# ─── The novelty ─────────────────────────────────────────────────────────────


class HyDERetriever:
    """Generate N hypothetical docs, embed them, mean-pool, then search.

    Not a ``BaseRetriever``: its ``retrieve()`` returns ``(results, metadata)``,
    not a bare list — see that method's docstring for why. Nothing calls this
    class through the ``BaseRetriever`` abstraction, so the interface mismatch
    is intentional, not an oversight.
    """

    def __init__(
        self,
        nodes: list[IndexedNode],
        *,
        embedding_model: str = "text-embedding-3-small",
        generator_model: str = "gpt-4.1-mini",
        samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 350,
    ) -> None:
        missing = [node.node_id for node in nodes if node.embedding is None]
        if missing:
            raise RuntimeError("HyDE requires embeddings saved during ingest.")
        self.nodes = nodes
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.samples = max(1, samples)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.embedder = Embedder(model=embedding_model)
        self.client = LLMClient()

    def retrieve(self, query: str, top_k: int) -> tuple[list[RetrievalResult], dict[str, Any]]:
        """Return (results, runtime_metadata) — not a bare list.

        Deliberately not ``BaseRetriever.retrieve()``'s shape and not stored as
        ``self.last_metadata``: this retriever calls an LLM before it can even
        search (``_generate_hypothetical_docs``), a wide window during which a
        second concurrent ``query()`` call on the same shared, already-loaded
        pipeline (``raglab eval/bench --concurrency``) could interleave. A
        write-then-read-back on ``self`` would let one query's runtime
        metadata get silently attached to a different query's answer. Returning
        the metadata alongside the results keeps it call-local instead.
        """
        # Step 1: ask the LLM to imagine `samples` answers to the query.
        hypothetical_docs, gen_meta = self._generate_hypothetical_docs(query)

        # Step 2: embed every imagined doc and mean-pool to one proxy vector.
        vectors = self.embedder.embed_texts(hypothetical_docs)
        proxy_vec = mean_dense_vector(vectors)

        # Step 3: cosine search the existing chunk embeddings with proxy_vec
        # (NOT with the query's own embedding — that's the HyDE trick).
        scored = [(node, dense_cosine(proxy_vec, node.embedding or [])) for node in self.nodes]
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)[:top_k]

        results = [
            RetrievalResult(
                node_id=node.node_id,
                chunk_id=node.chunk_id,
                doc_id=node.doc_id,
                text=node.text_for_generation,
                score=float(score),
                rank=rank,
                metadata={
                    **dict(node.metadata),
                    "hyde_model": self.generator_model,
                    "hyde_samples": self.samples,
                    "hyde_hypothetical_documents": hypothetical_docs,
                },
            )
            for rank, (node, score) in enumerate(ranked, start=1)
        ]

        runtime_metadata = {
            "method": "hyde",
            "generated_texts": hypothetical_docs,
            "embedding_input_count": len(hypothetical_docs),
            "estimated_embedding_tokens": sum(token_count(doc) for doc in hypothetical_docs),
            **gen_meta,
        }
        return results, runtime_metadata

    def _generate_hypothetical_docs(self, query: str) -> tuple[list[str], dict[str, Any]]:
        docs: list[str] = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        total_latency = 0.0
        total_cost = 0.0
        for sample_index in range(self.samples):
            completion = self.client.create_chat_completion(
                model=self.generator_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate one concise hypothetical document passage that would answer the user's "
                            "question. Do not mention that it is hypothetical. Write in the same language as "
                            "the question. Use a plausible wording that may differ from prior samples."
                        ),
                    },
                    {"role": "user", "content": f"Question: {query}\nSample: {sample_index + 1}"},
                ],
            )
            docs.append(completion.text)
            for key in total_usage:
                total_usage[key] += int(completion.usage.get(key, 0))
            total_latency += completion.latency_ms
            total_cost += completion.estimated_cost
        return docs, {
            "generation_usage": total_usage,
            "generation_latency_ms": round(total_latency, 3),
            "estimated_cost": round(total_cost, 8),
        }


# ─── Pipeline wiring ─────────────────────────────────────────────────────────


class HyDEPipeline(BasePipeline):
    """RAG-Sequence baseline, but retrieval uses hypothetical-document embeddings.

    Best for:
    - Short or vague queries
    - Vocabulary mismatch between queries and documents (jargon, paraphrase)
    - Multilingual setups where the query language differs from the corpus

    Weak for:
    - Strict latency / cost budgets (each query spends ``samples`` LLM calls
      plus an extra embedding pass before retrieval even starts)
    - Exact-keyword lookup (LLM may drift; BM25 would be more faithful)
    """

    id = "hyde_2022"
    name = "HyDE — Hypothetical Document Embeddings (Gao et al., 2022)"
    implementation_level = "paper_inspired"
    # HyDE calls the chat model to imagine documents during *retrieval*, not
    # just during full_rag answer synthesis — so generator_model is needed in
    # retrieval_only mode too, unlike most other techniques.
    retrieval_time_models = frozenset({"generator_model"})
    query_override_fields = frozenset(
        {
            "generator_model",
            "hyde_samples",
            "hyde_temperature",
            "hyde_max_tokens",
            "answer_temperature",
            "answer_max_tokens",
            "top_k",
            "max_context_tokens",
        }
    )
    _retriever: HyDERetriever

    def __init__(
        self,
        *,
        chunk_size: int = 220,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        embedding_batch_size: int = 64,
        generator_model: str = "gpt-4.1-mini",
        hyde_samples: int = 5,
        hyde_temperature: float = 0.7,
        hyde_max_tokens: int = 350,
        answer_temperature: float = 0.0,
        answer_max_tokens: int = 700,
        top_k: int = 5,
        max_context_tokens: int = 2200,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.generator_model = generator_model
        self.hyde_samples = hyde_samples
        self.hyde_temperature = hyde_temperature
        self.hyde_max_tokens = hyde_max_tokens
        self.answer_temperature = answer_temperature
        self.answer_max_tokens = answer_max_tokens
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        # Ingest is identical to RAG-Sequence — HyDE only changes inference.
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
        check_provider_ready(self.embedding_model)
        check_provider_ready(self.generator_model)
        manifest, nodes = self.load_artifact(artifact_path)
        _ = load_vector_store(artifact_path, nodes)  # warm the store; HyDE searches nodes directly
        # Built once: everything on self is fixed at construction (nodes,
        # embedder, client) and retrieve() returns its per-call runtime
        # metadata instead of stashing it on self — safe to share across
        # concurrent query() calls (raglab eval/bench --concurrency).
        self._retriever = HyDERetriever(
            nodes=nodes,
            embedding_model=self.embedding_model,
            generator_model=self.generator_model,
            samples=self.hyde_samples,
            temperature=self.hyde_temperature,
            max_tokens=self.hyde_max_tokens,
        )
        self._mark_loaded(artifact_path, manifest, nodes)

    def query(self, question: str, mode: str = "full_rag") -> RAGAnswer:
        self._require_loaded()
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest = self._manifest

        started = time.perf_counter()

        # === The HyDE step replaces the standard dense retriever ===
        retriever = self._retriever
        retrieved, retrieval_runtime = retriever.retrieve(question, self.top_k)
        # ===========================================================

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
                retrieval_runtime=retrieval_runtime,
                retriever_kind="hyde",
                question=question,
                answer_metadata=dict(answer.metadata),
            )
        )
        return answer
