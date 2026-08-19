"""Shared metadata builders so every pipeline produces comparable benchmark data.

Each technique writes its own end-to-end pipeline, but they all populate the
same shape of manifest (after ``ingest``) and the same shape of
``RAGAnswer.metadata`` (after ``query``).  Routing those through these helpers
keeps benchmark comparisons honest: no missing fields, no unit mismatches,
nothing hidden behind a registry.

This is composition, not inheritance — pipelines call these explicitly at the
end of their methods.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ragbench import __version__
from ragbench.core.schema import ArtifactManifest, JSONValue, VerificationReport
from ragbench.providers.env import env_float

if TYPE_CHECKING:
    from ragbench.core.schema import (
        BuiltContext,
        Chunk,
        DocumentBlock,
        IndexedNode,
        RetrievalResult,
        VerificationReport,
    )


ARTIFACT_VERSION = "5"


def build_ingest_manifest(
    *,
    pipeline_id: str,
    pipeline_name: str,
    input_path: str,
    blocks: list[DocumentBlock],
    chunks: list[Chunk],
    nodes: list[IndexedNode],
    pipeline_config: dict[str, JSONValue] | None = None,
    implementation_level: str = "unspecified",
    embedding_spec: dict[str, Any] | None = None,
    store_backend: str | None = None,
    extra: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Build the standard manifest dict to write alongside the artifact."""
    embedding_model: str | None = None
    if embedding_spec:
        params = embedding_spec.get("params") or {}
        embedding_model = params.get("model") or embedding_spec.get("model")

    config = pipeline_config or {}
    dimensions = {len(node.embedding) for node in nodes if node.embedding is not None}
    if len(dimensions) > 1:
        raise ValueError(f"Node embeddings have inconsistent dimensions: {sorted(dimensions)}")
    dimension = next(iter(dimensions), None)
    corpus_payload = [{"doc_id": block.doc_id, "block_id": block.block_id, "text": block.text} for block in blocks]
    manifest: ArtifactManifest = {
        "artifact_version": ARTIFACT_VERSION,
        "pipeline": {
            "id": pipeline_id,
            "implementation_level": implementation_level,
            "config": config,
            "config_fingerprint": canonical_fingerprint({"id": pipeline_id, "config": config}),
        },
        "embedding": {
            "type": str((embedding_spec or {}).get("type", "none")),
            "model": embedding_model,
            "dimension": dimension,
        },
        # ``nodes.json`` is the canonical local JSON store even for sparse
        # pipelines that do not carry vectors.
        "store": {"backend": store_backend or "json_memory"},
        "corpus": {
            "fingerprint": canonical_fingerprint(corpus_payload),
            "documents": sorted({block.doc_id for block in blocks}),
            "document_count": len({block.doc_id for block in blocks}),
            "block_count": len(blocks),
            "chunk_count": len(chunks),
            "node_count": len(nodes),
        },
        "runtime": {
            "created_at": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "input_path": input_path,
            "ingest_fingerprint": None,
            "runtime_fingerprint": None,
            "dependency_versions": {},
        },
        "extra": {"pipeline_name": pipeline_name},
    }
    if extra:
        manifest["extra"].update(extra)
    return manifest


def canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def skipped_verification(evidence_count: int) -> VerificationReport:
    return VerificationReport(
        grounded=False,
        citation_coverage=0.0,
        evidence_count=evidence_count,
        status="skipped",
        notes=["verification skipped in retrieval_only mode"],
    )


def build_query_metadata(
    *,
    latency_ms: float,
    retrieved: list[RetrievalResult],
    context: BuiltContext,
    verification: VerificationReport,
    artifact_manifest: dict[str, Any] | None = None,
    retrieval_runtime: dict[str, Any] | None = None,
    answer_metadata: dict[str, Any] | None = None,
    verification_runtime: dict[str, Any] | None = None,
    retriever_kind: str | None = None,
    question: str | None = None,
) -> dict[str, Any]:
    """Build the standard ``RAGAnswer.metadata`` payload.

    Centralising this is what lets ``ragbench bench`` compare paper A vs paper
    B fairly: latency, cost, verification, and provenance are always under
    the same keys regardless of which technique produced the answer.
    """
    retrieval_runtime = retrieval_runtime or {}
    answer_metadata = answer_metadata or {}
    verification_runtime = verification_runtime or {}
    artifact_manifest = artifact_manifest or {}

    retrieval_cost = _retrieval_cost(
        question=question,
        retriever_kind=retriever_kind,
        retrieval_runtime=retrieval_runtime,
    )
    cost_estimate = _cost_estimate(
        answer_metadata=answer_metadata,
        retrieval_cost=retrieval_cost,
        verification_runtime=verification_runtime,
    )
    components = dict(answer_metadata.get("components", {}))
    components.setdefault("retriever", retriever_kind)
    components.setdefault("generator", answer_metadata.get("model") or answer_metadata.get("mode"))
    components.setdefault(
        "verifier",
        "skipped"
        if verification.status == "skipped"
        else ("llm_verifier" if verification_runtime else "citation_coverage"),
    )

    return {
        "latency_ms": round(float(latency_ms), 3),
        "retrieved_count": len(retrieved),
        "context_token_count": context.token_count,
        "retrieval_runtime": retrieval_runtime,
        "retrieval_cost_estimate": retrieval_cost,
        "verification_runtime": verification_runtime,
        "cost_estimate": cost_estimate,
        "verification": verification.to_dict(),
        "components": components,
        "artifact_manifest": {
            "artifact_version": artifact_manifest.get("artifact_version"),
            "pipeline_id": artifact_manifest.get("pipeline", {}).get("id"),
            "config_fingerprint": artifact_manifest.get("pipeline", {}).get("config_fingerprint"),
            "corpus_fingerprint": artifact_manifest.get("corpus", {}).get("fingerprint"),
            "store_backend": artifact_manifest.get("store", {}).get("backend"),
            "embedding_model": artifact_manifest.get("embedding", {}).get("model"),
        },
    }


# ─── Cost helpers ────────────────────────────────────────────────────────────

_PAID_RETRIEVER_KINDS = {"openai_dense", "openai_hybrid", "hyde", "rag_fusion"}


def _retrieval_cost(
    *,
    question: str | None,
    retriever_kind: str | None,
    retrieval_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Estimate the USD cost of a single retrieval call.

    For paid retrievers we charge the embedding tokens (and any LLM calls
    the retriever made internally — recorded under ``estimated_cost`` inside
    ``last_metadata``).  For local retrievers we trust whatever the
    retriever itself reported, defaulting to zero.
    """
    if retriever_kind not in _PAID_RETRIEVER_KINDS:
        runtime_cost = float(retrieval_runtime.get("estimated_cost", 0.0))
        basis = "local retriever (no paid calls)" if runtime_cost == 0.0 else "custom retriever runtime estimate"
        return {"currency": "USD", "amount": runtime_cost, "basis": basis}

    from ragbench.core.text import token_count

    embedding_inputs = int(retrieval_runtime.get("embedding_input_count", 1))
    if "estimated_embedding_tokens" in retrieval_runtime:
        tokens = int(retrieval_runtime["estimated_embedding_tokens"])
    else:
        question_tokens = token_count(question) if question else 0
        tokens = question_tokens * embedding_inputs

    amount = tokens / 1000 * env_float("LLM_EMBEDDING_INPUT_COST_PER_1K", 0.0)
    amount += float(retrieval_runtime.get("estimated_cost", 0.0))
    return {
        "currency": "USD",
        "amount": round(amount, 8),
        "basis": "retrieval-time LLM/embedding calls (env per-1K-token rates)",
        "estimated_query_embedding_tokens": tokens,
    }


def _cost_estimate(
    *,
    answer_metadata: dict[str, Any],
    retrieval_cost: dict[str, Any],
    verification_runtime: dict[str, Any],
) -> dict[str, Any]:
    """Combine generation, retrieval, and verification costs into one USD figure."""
    verification_cost = float(verification_runtime.get("estimated_cost", 0.0))
    generation_cost = float(answer_metadata.get("estimated_llm_cost", 0.0))
    auxiliary_cost = float(answer_metadata.get("auxiliary_llm_cost", 0.0))
    retrieval_amount = float(retrieval_cost.get("amount", 0.0))

    total = generation_cost + auxiliary_cost + retrieval_amount + verification_cost
    if total > 0:
        return {
            "currency": "USD",
            "amount": round(total, 8),
            "basis": (
                "generation + auxiliary agent/query LLM + retrieval + verification estimates "
                "from env per-1K-token values"
            ),
            "status": "estimated",
        }
    return {
        "currency": "USD",
        "amount": 0.0,
        "basis": "unknown unless all model calls are local or pricing env rates are configured",
        "status": "unknown",
    }
