"""Preflight diagnostics for a technique before an expensive benchmark."""

from __future__ import annotations

import importlib.util
from typing import Any

from ragbench.core.base import BasePipeline, get_pipeline_spec

# Attributes only checked for a full-RAG run, unless the technique declares
# them in ``retrieval_time_models`` (e.g. HyDE/RAG-Fusion call the chat model
# during retrieval itself, so it's needed in retrieval_only mode too).
_MODE_CONDITIONAL_ATTRS = ("generator_model", "verifier_model")
# Attributes needed regardless of mode: embedding at query time, the agent
# loop (itself an LLM) in agentic techniques, and the ingest-time context model.
_ALWAYS_CHECKED_ATTRS = ("embedding_model", "agent_model", "context_model")


def diagnose_technique(
    pipeline: BasePipeline,
    mode: str = "full_rag",
    offline: bool = False,
    _probe_cache: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Check provider/dependency readiness for one technique.

    ``mode`` narrows which chat-model checks are required: a
    ``retrieval_only`` run does not need a full_rag-only technique's
    ``generator_model``/``verifier_model`` key, unless the technique declares
    that attribute in ``retrieval_time_models`` (it calls that model during
    retrieval itself, not just full-RAG answer synthesis).

    By default (``offline=False``) every check that passes the static
    env-var check also gets a single live probe (one minimal real request,
    see ``ragbench.providers.llm_client.probe_provider``) — a present but
    revoked/expired key still passes the static check alone, and a dead key
    should be caught in seconds, not after hours of ingest. Pass
    ``offline=True`` to reproduce the old static-only behavior (for CI/
    debugging, or when network access is unavailable).

    ``_probe_cache`` lets a caller (``run_preflight``) share live-probe
    results across multiple techniques in the same run, deduplicating by
    ``(operation, model)`` — e.g. two techniques (or two attributes on the
    same technique) that both default to the same chat model must trigger
    exactly one live request, not one per attribute.
    """
    spec = get_pipeline_spec(pipeline.id)
    retrieval_time_models: frozenset[str] = getattr(pipeline, "retrieval_time_models", frozenset())
    checks: list[dict[str, Any]] = []
    probe_cache: dict[tuple[str, str], dict[str, Any]] = {} if _probe_cache is None else _probe_cache

    def _probe(operation: str, model: str) -> dict[str, Any]:
        key = (operation, model)
        if key not in probe_cache:
            from ragbench.providers.llm_client import probe_provider

            probe_cache[key] = probe_provider(model, operation)  # type: ignore[arg-type]
        return probe_cache[key]

    for attribute in (*_ALWAYS_CHECKED_ATTRS, *_MODE_CONDITIONAL_ATTRS):
        model = getattr(pipeline, attribute, None)
        if not isinstance(model, str):
            continue
        mode_conditional = attribute in _MODE_CONDITIONAL_ATTRS and attribute not in retrieval_time_models
        if mode_conditional and mode != "full_rag":
            continue  # this technique does not call `attribute` in this mode
        try:
            from ragbench.providers.llm_client import check_provider_ready

            check_provider_ready(model)
        except RuntimeError as exc:
            checks.append(
                {"name": attribute, "status": "failed", "detail": str(exc), "latency_ms": None, "error_type": None}
            )
            continue
        if offline:
            checks.append(
                {"name": attribute, "status": "ok", "detail": model, "latency_ms": None, "error_type": None}
            )
            continue
        operation = "embedding" if attribute == "embedding_model" else "chat"
        probe = _probe(operation, model)
        if probe["reachable"]:
            checks.append(
                {
                    "name": attribute,
                    "status": "ok",
                    "detail": model,
                    "latency_ms": probe["latency_ms"],
                    "error_type": None,
                }
            )
        else:
            checks.append(
                {
                    "name": attribute,
                    "status": "failed",
                    "detail": (
                        f"live probe failed for '{model}' ({operation}): "
                        f"{probe['error_type']}: {probe['error_detail']}"
                    ),
                    "latency_ms": probe["latency_ms"],
                    "error_type": probe["error_type"],
                }
            )
    if hasattr(pipeline, "reranker_model"):
        if getattr(pipeline, "reranker_backend", "local") == "api":
            try:
                from ragbench.providers.llm_client import check_provider_ready

                check_provider_ready(pipeline.reranker_model)
            except RuntimeError as exc:
                checks.append(
                    {
                        "name": "cross_encoder",
                        "status": "failed",
                        "detail": str(exc),
                        "latency_ms": None,
                        "error_type": None,
                    }
                )
            else:
                if offline:
                    checks.append(
                        {
                            "name": "cross_encoder",
                            "status": "ok",
                            "detail": f"api reranker: {pipeline.reranker_model}",
                            "latency_ms": None,
                            "error_type": None,
                        }
                    )
                else:
                    probe = _probe("rerank", pipeline.reranker_model)
                    if probe["reachable"]:
                        checks.append(
                            {
                                "name": "cross_encoder",
                                "status": "ok",
                                "detail": f"api reranker: {pipeline.reranker_model}",
                                "latency_ms": probe["latency_ms"],
                                "error_type": None,
                            }
                        )
                    else:
                        checks.append(
                            {
                                "name": "cross_encoder",
                                "status": "failed",
                                "detail": (
                                    f"live probe failed for reranker '{pipeline.reranker_model}': "
                                    f"{probe['error_type']}: {probe['error_detail']}"
                                ),
                                "latency_ms": probe["latency_ms"],
                                "error_type": probe["error_type"],
                            }
                        )
        else:
            available = importlib.util.find_spec("sentence_transformers") is not None
            checks.append(
                {
                    "name": "cross_encoder",
                    "status": "ok" if available else "failed",
                    "detail": "sentence-transformers installed" if available else "install the [rerank] extra",
                    "latency_ms": None,
                    "error_type": None,
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
