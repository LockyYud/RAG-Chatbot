"""Self-RAG — LLM-as-verifier post-generation critique (Asai et al., 2023).

Paper: "Self-RAG: Learning to Retrieve, Generate, and Critique through
        Self-Reflection"
       Asai, Wu, Wang, Sil, Hajishirzi — ICLR 2024
       https://arxiv.org/abs/2310.11511

The key idea
------------
The published Self-RAG trains a model to emit *reflection tokens* during
decoding so it can decide when to retrieve, when to critique, and how to
rerank candidate generations.  That whole training story is out of scope
here.  What we *do* reproduce — and what is the most portable contribution
of the paper — is the **post-generation critique step**: after the answer
exists, a separate LLM call judges whether every claim in the answer is
supported by the retrieved context.  The judge returns a structured verdict
``{grounded, citation_coverage, unsupported_citations, notes}``.

This swap moves verification from a rule-based citation checker (which only
verifies that citations are present and well-formed) to an LLM that reads
both the answer and the context and decides whether they actually agree.

What's different from Naive RAG / RAG-Sequence
----------------------------------------------
Only the **verifier**.  Retrieval uses BM25 + lexical-overlap reranking
(cheaper than embeddings, suits the prose-heavy datasets the paper targeted)
and the generator is a chat LLM — but those are choices, not the novelty.
Scan ``SelfRAGPipeline.query`` and the only paper-specific class is
``SelfRAGCritiqueVerifier`` below.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from raglab.core.base import BasePipeline
from raglab.core.interfaces import BaseVerifier
from raglab.core.io import iter_input_files
from raglab.core.measure import build_ingest_manifest, build_query_metadata, skipped_verification
from raglab.core.schema import ArtifactManifest, BuiltContext, RAGAnswer, VerificationReport
from raglab.indexing.artifacts import save_nodes
from raglab.indexing.retrievers import BM25Retriever
from raglab.inference.context_builders.citation_context import CitationContextBuilder
from raglab.inference.generators.chat import ChatGenerator
from raglab.inference.rerankers.lexical_overlap import LexicalOverlapReranker
from raglab.processing.chunkers.parent_child import ParentChildChunker
from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
from raglab.processing.enrichers.basic import SectionTitleEnricher
from raglab.processing.parsers.text_parser import TextParser
from raglab.providers.llm_client import LLMClient, check_provider_ready

# ─── The novelty ─────────────────────────────────────────────────────────────


class SelfRAGCritiqueVerifier(BaseVerifier):
    """Ask an LLM to judge whether the answer is fully grounded in the context."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0,
        max_tokens: int = 350,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = LLMClient()
        self.last_metadata: dict[str, Any] = {}

    def verify(self, answer: RAGAnswer, context: BuiltContext) -> VerificationReport:
        # Single LLM call: read answer + context, return structured verdict.
        completion = self.client.create_chat_completion(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG verifier. Judge whether ANSWER is fully supported "
                        "by CONTEXT. Return only JSON with keys: grounded (boolean), "
                        "citation_coverage (number 0-1), unsupported_citations (array of strings), "
                        "notes (array of strings)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{answer.query}\n\n"
                        f"ANSWER:\n{answer.answer}\n\n"
                        f"ANSWER_CITATIONS:\n{json.dumps(answer.citations, ensure_ascii=False)}\n\n"
                        f"CONTEXT:\n{context.text}"
                    ),
                },
            ],
        )
        payload = _parse_json_object(completion.text)

        self.last_metadata = {
            "method": "self_rag_critique",
            "model": self.model,
            "usage": completion.usage,
            "latency_ms": completion.latency_ms,
            "estimated_cost": completion.estimated_cost,
        }

        grounded = bool(payload.get("grounded", False))
        coverage = float(payload.get("citation_coverage", 0.0))
        unsupported = [str(x) for x in payload.get("unsupported_citations", [])]
        notes = [str(x) for x in payload.get("notes", [])]
        notes.append(f"self_rag_critique_model={self.model}")
        return VerificationReport(
            grounded=grounded,
            citation_coverage=round(max(0.0, min(1.0, coverage)), 6),
            evidence_count=len(context.results),
            unsupported_citations=unsupported,
            notes=notes,
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    """LLMs occasionally wrap JSON in prose; salvage the first ``{...}`` block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "grounded": False,
                "citation_coverage": 0.0,
                "unsupported_citations": [],
                "notes": ["verifier did not return parseable JSON"],
            }
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "grounded": False,
                "citation_coverage": 0.0,
                "unsupported_citations": [],
                "notes": ["verifier returned malformed JSON"],
            }


# ─── Pipeline wiring ─────────────────────────────────────────────────────────


class SelfRAGPipeline(BasePipeline):
    """RAG with LLM-based grounding check after generation.

    Best for:
    - Citation-sensitive QA where hallucination detection matters
    - Production review gates before serving answers
    - Research comparing rule-based vs. LLM-based verification

    Weak for:
    - Strict latency budgets (one extra LLM call per query)
    - Deterministic audits (LLM judge outputs may vary across runs)
    - Table-heavy documents (the prompt is tuned for prose)
    """

    id = "self_rag_2023"
    name = "Self-RAG — LLM critique verifier (Asai et al., 2023)"
    implementation_level = "concept_only"
    query_override_fields = frozenset(
        {
            "top_k",
            "rerank_top_k",
            "rerank_weight",
            "max_context_tokens",
            "generator_model",
            "generator_temperature",
            "generator_max_tokens",
            "verifier_model",
            "verifier_temperature",
            "verifier_max_tokens",
        }
    )
    _retriever: BM25Retriever

    def __init__(
        self,
        *,
        child_size: int = 120,
        child_overlap: int = 20,
        top_k: int = 8,
        rerank_top_k: int = 5,
        rerank_weight: float = 0.25,
        max_context_tokens: int = 2400,
        generator_model: str = "gpt-4.1-mini",
        generator_temperature: float = 0.0,
        generator_max_tokens: int = 700,
        verifier_model: str = "gpt-4.1-mini",
        verifier_temperature: float = 0.0,
        verifier_max_tokens: int = 350,
    ) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.rerank_weight = rerank_weight
        self.max_context_tokens = max_context_tokens
        self.generator_model = generator_model
        self.generator_temperature = generator_temperature
        self.generator_max_tokens = generator_max_tokens
        self.verifier_model = verifier_model
        self.verifier_temperature = verifier_temperature
        self.verifier_max_tokens = verifier_max_tokens

    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        # self_rag uses BM25 — no embedding needed at ingest, only chat at query.
        parser = TextParser()
        blocks = []
        for path in iter_input_files(input_path):
            blocks.extend(parser.parse(str(path)))
        blocks = VietnameseNormalizer().clean(blocks)
        blocks = WhitespaceCleaner().clean(blocks)

        chunks = ParentChildChunker(child_size=self.child_size, child_overlap=self.child_overlap).chunk(blocks)
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
        self._retriever = BM25Retriever(nodes=nodes)
        self._mark_loaded(artifact_path, manifest, nodes)

    def query(self, question: str, mode: str = "full_rag") -> RAGAnswer:
        self._require_loaded()
        if mode not in {"full_rag", "retrieval_only"}:
            raise ValueError(f"mode must be 'full_rag' or 'retrieval_only', got {mode!r}")
        manifest = self._manifest
        if mode == "full_rag":
            check_provider_ready(self.generator_model)
            check_provider_ready(self.verifier_model)

        started = time.perf_counter()

        retrieved = self._retriever.retrieve(question, self.top_k)
        reranked = LexicalOverlapReranker(weight=self.rerank_weight).rerank(question, retrieved, self.rerank_top_k)
        context = CitationContextBuilder(max_tokens=self.max_context_tokens).build_context(question, reranked)

        if mode == "retrieval_only":
            answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
        elif mode == "full_rag":
            answer = ChatGenerator(
                model=self.generator_model,
                temperature=self.generator_temperature,
                max_tokens=self.generator_max_tokens,
            ).generate(question, context)
        # === The Self-RAG step: LLM-based grounding check (full RAG only) ===
        verifier = None
        if mode == "full_rag":
            verifier = SelfRAGCritiqueVerifier(
                model=self.verifier_model,
                temperature=self.verifier_temperature,
                max_tokens=self.verifier_max_tokens,
            )
            verification = verifier.verify(answer, context)
        else:
            verification = skipped_verification(len(context.results))
        # ====================================================

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
                verification_runtime=verifier.last_metadata if verifier is not None else {},
            )
        )
        return answer
