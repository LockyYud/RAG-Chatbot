from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

from raglab.core.config import get_stage, load_config
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
from raglab.core.text import token_count
from raglab.indexing.artifacts import load_nodes, save_nodes
from raglab.indexing.embeddings import OpenAIEmbedder
from raglab.providers.env import env_float


def ingest(config_path: str, input_path: str, output_path: str) -> dict[str, Any]:
    register_defaults()
    config = load_config(config_path)
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

    manifest = {
        "pipeline_name": config.get("name", Path(config_path).stem),
        "config_path": config_path,
        "config_hash": _hash_file(config_path),
        "input_path": input_path,
        "documents": sorted({block.doc_id for block in blocks}),
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "node_count": len(nodes),
        "embedding": embedding_spec or {"type": "none"},
    }
    save_nodes(output_path, nodes, manifest)
    return manifest


def query(config_path: str, artifact_path: str, question: str) -> RAGAnswer:
    register_defaults()
    config = load_config(config_path)
    nodes = load_nodes(artifact_path)

    retriever_spec = get_stage(config, "inference.retriever", {"type": "dense", "params": {"top_k": 5}})
    retriever_params = dict(retriever_spec.get("params", {})) if isinstance(retriever_spec, dict) else {}
    top_k = int(retriever_params.pop("top_k", 5))
    retriever = retrievers.create(retriever_spec, nodes=nodes)

    reranker_spec = get_stage(config, "inference.reranker", {"type": "none"})
    reranker_params = dict(reranker_spec.get("params", {})) if isinstance(reranker_spec, dict) else {}
    rerank_top_k = int(reranker_params.get("top_k", top_k))

    context_spec = get_stage(config, "inference.context_builder", {"type": "citation_context"})
    generator_spec = get_stage(config, "inference.generator", {"type": "citation_required"})
    verifier_spec = get_stage(config, "inference.verifier", {"type": "citation_coverage"})

    started = time.perf_counter()
    retrieved = retriever.retrieve(question, top_k)
    reranked = rerankers.create(reranker_spec).rerank(question, retrieved, rerank_top_k)
    context = context_builders.create(context_spec).build_context(question, reranked)
    answer = generators.create(generator_spec).generate(question, context)
    verification = verifiers.create(verifier_spec).verify(answer, context)
    elapsed_ms = (time.perf_counter() - started) * 1000
    retrieval_cost = _retrieval_cost(question, retriever_spec)
    answer.metadata.update(
        {
            "latency_ms": round(elapsed_ms, 3),
            "retrieved_count": len(retrieved),
            "context_token_count": context.token_count,
            "retrieval_cost_estimate": retrieval_cost,
            "cost_estimate": _cost_estimate(answer, retrieval_cost),
            "verification": verification.to_dict(),
        }
    )
    return answer


def _cost_estimate(answer: RAGAnswer, retrieval_cost: dict[str, Any]) -> dict[str, Any]:
    if "estimated_llm_cost" in answer.metadata:
        return {
            "currency": "USD",
            "amount": round(float(answer.metadata["estimated_llm_cost"]) + float(retrieval_cost["amount"]), 8),
            "basis": "chat + query embedding estimates from .env per-1K-token values",
        }
    if retrieval_cost["amount"] > 0:
        return retrieval_cost
    return {
        "currency": "USD",
        "amount": 0.0,
        "basis": "local fallback; no paid model calls",
    }


def _retrieval_cost(question: str, retriever_spec: dict[str, Any] | str | None) -> dict[str, Any]:
    type_name = retriever_spec.get("type") if isinstance(retriever_spec, dict) else retriever_spec
    if type_name not in {"openai_dense", "openai_hybrid"}:
        return {"currency": "USD", "amount": 0.0, "basis": "retriever does not call embedding API"}
    tokens = token_count(question)
    amount = round(tokens / 1000 * env_float("OPENAI_EMBEDDING_INPUT_COST_PER_1K", 0.0), 8)
    return {
        "currency": "USD",
        "amount": amount,
        "basis": "OPENAI_EMBEDDING_INPUT_COST_PER_1K from .env; query embedding only",
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
