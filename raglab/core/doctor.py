"""Preflight diagnostics for a technique before an expensive benchmark."""

from __future__ import annotations

import importlib.util
from typing import Any

from raglab.core.base import BasePipeline, get_pipeline_spec

# Attributes only checked for a full-RAG run, unless the technique declares
# them in ``retrieval_time_models`` (e.g. HyDE/RAG-Fusion call the chat model
# during retrieval itself, so it's needed in retrieval_only mode too).
_MODE_CONDITIONAL_ATTRS = ("generator_model", "verifier_model")
# Attributes needed regardless of mode: embedding at query time, the agent
# loop (itself an LLM) in agentic techniques, and the ingest-time context model.
_ALWAYS_CHECKED_ATTRS = ("embedding_model", "agent_model", "context_model")


def diagnose_technique(pipeline: BasePipeline, mode: str = "full_rag") -> dict[str, Any]:
    """Check provider/dependency readiness for one technique.

    ``mode`` narrows which chat-model checks are required: a
    ``retrieval_only`` run does not need a full_rag-only technique's
    ``generator_model``/``verifier_model`` key, unless the technique declares
    that attribute in ``retrieval_time_models`` (it calls that model during
    retrieval itself, not just full-RAG answer synthesis).
    """
    spec = get_pipeline_spec(pipeline.id)
    retrieval_time_models: frozenset[str] = getattr(pipeline, "retrieval_time_models", frozenset())
    checks: list[dict[str, Any]] = []
    for attribute in (*_ALWAYS_CHECKED_ATTRS, *_MODE_CONDITIONAL_ATTRS):
        model = getattr(pipeline, attribute, None)
        if not isinstance(model, str):
            continue
        mode_conditional = attribute in _MODE_CONDITIONAL_ATTRS and attribute not in retrieval_time_models
        if mode_conditional and mode != "full_rag":
            continue  # this technique does not call `attribute` in this mode
        try:
            from raglab.providers.llm_client import check_provider_ready

            check_provider_ready(model)
        except RuntimeError as exc:
            checks.append({"name": attribute, "status": "failed", "detail": str(exc)})
        else:
            checks.append({"name": attribute, "status": "ok", "detail": model})
    if hasattr(pipeline, "reranker_model"):
        available = importlib.util.find_spec("sentence_transformers") is not None
        checks.append(
            {
                "name": "cross_encoder",
                "status": "ok" if available else "failed",
                "detail": "sentence-transformers installed" if available else "install the [rerank] extra",
            }
        )
    return {
        "technique": pipeline.id,
        "mode": mode,
        "implementation_level": spec.implementation_level,
        "evaluation_profiles": sorted(spec.evaluation_profiles),
        "custom_artifacts": spec.custom_artifacts,
        "requirements": sorted(spec.requirements),
        "checks": checks,
        "ready": all(check["status"] == "ok" for check in checks),
    }
