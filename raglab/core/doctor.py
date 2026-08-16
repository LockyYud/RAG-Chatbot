"""Preflight diagnostics for a technique before an expensive benchmark."""

from __future__ import annotations

import importlib.util
from typing import Any

from raglab.core.base import BasePipeline, get_pipeline_spec


def diagnose_technique(pipeline: BasePipeline) -> dict[str, Any]:
    spec = get_pipeline_spec(pipeline.id)
    checks: list[dict[str, Any]] = []
    for attribute in ("embedding_model", "generator_model", "agent_model", "context_model", "verifier_model"):
        model = getattr(pipeline, attribute, None)
        if not isinstance(model, str):
            continue
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
        "implementation_level": spec.implementation_level,
        "evaluation_profiles": sorted(spec.evaluation_profiles),
        "custom_artifacts": spec.custom_artifacts,
        "requirements": sorted(spec.requirements),
        "checks": checks,
        "ready": all(check["status"] == "ok" for check in checks),
    }
