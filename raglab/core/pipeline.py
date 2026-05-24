from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from raglab.core.config import get_stage, load_config, validate_config
from raglab.core.registry import (
    chunkers,
    cleaners,
    context_builders,
    enrichers,
    generators,
    parsers,
    register_defaults,
    rerankers,
    retrievers,
    verifiers,
)
from raglab.core.schema import DocumentBlock, IndexedNode, RAGAnswer
from raglab.core.techniques import register_custom_for_config
from raglab.core.text import token_count
from raglab.indexing.artifacts import ARTIFACT_VERSION, load_manifest, load_nodes, load_vector_store, save_nodes
from raglab.indexing.embeddings import OpenAIEmbedder
from raglab.providers.env import env_float


def ingest(config_path: str, input_path: str, output_path: str) -> dict[str, Any]:
    register_defaults()
    register_custom_for_config(config_path)
    config = load_config(config_path)
    validate_config(config, for_ingest=True)
    parser = parsers.create(get_stage(config, "processing.parser", {"type": "text"}))
    cleaner_specs = get_stage(config, "processing.cleaners", [])
    chunker = chunkers.create(get_stage(config, "processing.chunker", {"type": "fixed_size"}))
    enricher_specs = get_stage(config, "processing.enrichers", [{"type": "none"}])

    all_blocks: list[DocumentBlock] = []
    for path in _iter_input_files(input_path):
        all_blocks.extend(parser.parse(str(path)))

    blocks = all_blocks
    for spec in cleaner_specs:
        blocks = cleaners.create(spec).clean(blocks)

    chunks = chunker.chunk(blocks)
    if not enricher_specs:
        enricher_specs = [{"type": "none"}]
    if len(enricher_specs) > 1:
        raise ValueError("Only one chunk-to-node enricher is supported per pipeline in this MVP")
    nodes: list[IndexedNode] = enrichers.create(enricher_specs[0]).enrich(chunks)

    embedding_spec = get_stage(config, "indexing.embedding")
    if isinstance(embedding_spec, dict) and embedding_spec.get("type") == "openai":
        nodes = OpenAIEmbedder(**dict(embedding_spec.get("params", {}))).embed_nodes(nodes)

    store_spec = get_stage(config, "indexing.store")
    if store_spec is None and isinstance(embedding_spec, dict) and embedding_spec.get("type") == "openai":
        store_spec = {"type": "json_memory"}
    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": config.get("name", Path(config_path).stem),
        "config_path": config_path,
        "config_hash": _hash_file(config_path),
        "input_path": input_path,
        "documents": sorted({block.doc_id for block in blocks}),
        "document_count": len({block.doc_id for block in blocks}),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "node_count": len(nodes),
        "embedding": embedding_spec or {"type": "none"},
        "embedding_model": _embedding_model(embedding_spec),
        "store_backend": _store_backend(store_spec),
    }
    save_nodes(output_path, nodes, manifest, store_spec=store_spec)
    return manifest


def query(config_path: str, artifact_path: str, question: str, mode: str = "full_rag") -> RAGAnswer:
    register_defaults()
    register_custom_for_config(config_path)
    config = load_config(config_path)
    validate_config(config, for_query=True)
    nodes = load_nodes(artifact_path)
    manifest = load_manifest(artifact_path)
    vector_store = load_vector_store(artifact_path, nodes)

    retriever_spec = get_stage(config, "inference.retriever", {"type": "dense", "params": {"top_k": 5}})
    retriever_params = dict(retriever_spec.get("params", {})) if isinstance(retriever_spec, dict) else {}
    top_k = int(retriever_params.pop("top_k", 5))
    _validate_artifact_for_retriever(retriever_spec, manifest, nodes)
    retriever = retrievers.create(retriever_spec, nodes=nodes, vector_store=vector_store)

    reranker_spec = get_stage(config, "inference.reranker", {"type": "none"})
    reranker_params = dict(reranker_spec.get("params", {})) if isinstance(reranker_spec, dict) else {}
    rerank_top_k = int(reranker_params.get("top_k", top_k))

    context_spec = get_stage(config, "inference.context_builder", {"type": "citation_context"})
    generator_spec = get_stage(config, "inference.generator", {"type": "citation_required"})
    verifier_spec = get_stage(config, "inference.verifier", {"type": "citation_coverage"})

    started = time.perf_counter()
    retrieved = retriever.retrieve(question, top_k)
    retrieval_runtime = getattr(retriever, "last_metadata", {})
    reranked = rerankers.create(reranker_spec).rerank(question, retrieved, rerank_top_k)
    context = context_builders.create(context_spec).build_context(question, reranked)
    if mode == "retrieval_only":
        answer = RAGAnswer(query=question, answer="", contexts=context.results, citations=[])
    elif mode == "full_rag":
        answer = generators.create(generator_spec).generate(question, context)
    else:
        raise ValueError("mode must be retrieval_only or full_rag")
    verifier = verifiers.create(verifier_spec)
    verification = verifier.verify(answer, context)
    verification_runtime = getattr(verifier, "last_metadata", {})
    elapsed_ms = (time.perf_counter() - started) * 1000
    retrieval_cost = _retrieval_cost(question, retriever_spec, retrieval_runtime)
    answer.metadata.update(
        {
            "latency_ms": round(elapsed_ms, 3),
            "retrieved_count": len(retrieved),
            "context_token_count": context.token_count,
            "retrieval_runtime": retrieval_runtime,
            "artifact_manifest": {
                "artifact_version": manifest.get("artifact_version"),
                "config_hash": manifest.get("config_hash"),
                "store_backend": manifest.get("store_backend"),
                "embedding_model": manifest.get("embedding_model"),
            },
            "retrieval_cost_estimate": retrieval_cost,
            "verification_runtime": verification_runtime,
            "cost_estimate": _cost_estimate(answer, retrieval_cost, verification_runtime),
            "verification": verification.to_dict(),
        }
    )
    return answer


def _cost_estimate(
    answer: RAGAnswer,
    retrieval_cost: dict[str, Any],
    verification_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    verification_runtime = verification_runtime or {}
    verification_cost = float(verification_runtime.get("estimated_cost", 0.0))
    if "estimated_llm_cost" in answer.metadata:
        return {
            "currency": "USD",
            "amount": round(
                float(answer.metadata["estimated_llm_cost"]) + float(retrieval_cost["amount"]) + verification_cost,
                8,
            ),
            "basis": "generation + retrieval + verification estimates from .env per-1K-token values",
        }
    if retrieval_cost["amount"] > 0:
        retrieval_cost = dict(retrieval_cost)
        retrieval_cost["amount"] = round(float(retrieval_cost["amount"]) + verification_cost, 8)
        return retrieval_cost
    return {
        "currency": "USD",
        "amount": 0.0,
        "basis": "local fallback; no paid model calls",
    }


def _retrieval_cost(
    question: str,
    retriever_spec: dict[str, Any] | str | None,
    retrieval_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    type_name = retriever_spec.get("type") if isinstance(retriever_spec, dict) else retriever_spec
    retrieval_runtime = retrieval_runtime or {}
    paid_retrievers = {
        "openai_dense",
        "openai_hybrid",
        "hyde",
        "hyde_retriever",
        "rag_fusion",
        "rag_fusion_retriever",
    }
    if type_name not in paid_retrievers:
        runtime_cost = float(retrieval_runtime.get("estimated_cost", 0.0))
        return {"currency": "USD", "amount": runtime_cost, "basis": "custom retriever runtime estimate"}
    embedding_inputs = int(retrieval_runtime.get("embedding_input_count", 1))
    tokens = int(retrieval_runtime.get("estimated_embedding_tokens", token_count(question) * embedding_inputs))
    amount = round(tokens / 1000 * env_float("OPENAI_EMBEDDING_INPUT_COST_PER_1K", 0.0), 8)
    amount += float(retrieval_runtime.get("estimated_cost", 0.0))
    return {
        "currency": "USD",
        "amount": round(amount, 8),
        "basis": "retrieval-time LLM calls plus OPENAI_EMBEDDING_INPUT_COST_PER_1K from .env",
        "estimated_query_embedding_tokens": tokens,
    }


def _iter_input_files(input_path: str) -> list[Path]:
    source = Path(input_path)
    if source.is_file():
        return [source]
    allowed = {".txt", ".md", ".markdown"}
    return sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in allowed)


def _hash_file(path: str) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16]


def _embedding_model(embedding_spec: Any) -> str | None:
    if isinstance(embedding_spec, dict):
        params = dict(embedding_spec.get("params", {}))
        return params.get("model")
    return None


def _store_backend(store_spec: Any) -> str | None:
    if isinstance(store_spec, dict):
        return str(store_spec.get("type", "json_memory"))
    if isinstance(store_spec, str):
        return store_spec
    return None


def _validate_artifact_for_retriever(
    retriever_spec: dict[str, Any] | str | None,
    manifest: dict[str, Any],
    nodes: list[IndexedNode],
) -> None:
    type_name = retriever_spec.get("type") if isinstance(retriever_spec, dict) else retriever_spec
    embedding_retrievers = {
        "openai_dense",
        "openai_hybrid",
        "hyde",
        "hyde_retriever",
        "rag_fusion",
        "rag_fusion_retriever",
    }
    if type_name in embedding_retrievers:
        if not any(node.embedding is not None for node in nodes):
            raise RuntimeError(
                f"Retriever '{type_name}' requires embeddings. Re-run ingest with indexing.embedding.type=openai."
            )
        if manifest and manifest.get("embedding_model") is None:
            raise RuntimeError(f"Artifact manifest is missing embedding_model for retriever '{type_name}'.")
