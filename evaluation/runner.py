from __future__ import annotations

import importlib.metadata
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evaluation.judge import create_judge
from evaluation.metrics import evaluate_prediction_rows, evaluate_predictions
from evaluation.profiles import resolve_profile, validate_profile
from raglab import __version__
from raglab.core.base import get_pipeline_spec
from raglab.core.io import read_json, read_jsonl, write_json
from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import EvalItem, RAGAnswer
from raglab.datasets.schema import resolve_eval_dataset_path, validate_processed_dataset
from raglab.indexing.artifacts import load_manifest
from raglab.providers.llm_client import capture_provider_usage

if TYPE_CHECKING:
    from raglab.core.base import BasePipeline

REPORT_SCHEMA_VERSION = "2"


def run_eval(
    pipeline: BasePipeline,
    artifact_path: str,
    dataset_path: str,
    output_path: str,
    top_k: int = 5,
    mode: str = "full_rag",
    judge_spec: dict | None = None,
    profile: str = "auto",
    seed: int | None = None,
    cutoffs: list[int] | None = None,
    suite_metadata: dict[str, str] | None = None,
    warmup_queries: int = 0,
    latency_repetitions: int = 1,
) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if warmup_queries < 0 or latency_repetitions < 1:
        raise ValueError("warmup_queries must be non-negative and latency_repetitions at least 1")
    # Evaluation is always strict, even when the artifact was created by an
    # interactive demo that opted into component fallback.
    if hasattr(pipeline, "allow_fallback"):
        pipeline.allow_fallback = False
    cutoffs = sorted(set(cutoffs or [top_k]))
    if not cutoffs or min(cutoffs) < 1:
        raise ValueError("cutoffs must contain positive integers")
    if top_k not in cutoffs:
        cutoffs.append(top_k)
        cutoffs.sort()
    evaluation_overrides = _ensure_evaluation_depth(pipeline, max(cutoffs))
    if Path(dataset_path).is_dir():
        validate_processed_dataset(dataset_path)
    resolved_dataset_path = resolve_eval_dataset_path(dataset_path)
    rows = read_jsonl(resolved_dataset_path)
    items = [EvalItem.from_dict(row) for row in rows]
    if profile == "auto" and mode == "full_rag":
        profile = "citation_rag" if any(item.expected_citations for item in items) else "single_hop_rag"
    else:
        profile = resolve_profile(profile, mode)
    supported_profiles = get_pipeline_spec(pipeline.id).evaluation_profiles
    if profile not in supported_profiles:
        raise ValueError(f"Technique '{pipeline.id}' does not declare support for evaluation profile '{profile}'")
    validate_profile(profile, items)
    judge = create_judge(judge_spec) if mode == "full_rag" else None
    for item in items[:warmup_queries]:
        # Explicit warm-ups prevent initialization latency from contaminating
        # scored requests. They are excluded from all metrics and costs.
        with capture_provider_usage():
            pipeline.query(artifact_path, item.question, mode=mode)
    predictions: list[RAGAnswer] = []
    for item in items:
        timings: list[float] = []
        repeated_usage: list[dict[str, Any]] = []
        prediction: RAGAnswer | None = None
        pipeline_provider: dict[str, Any] | None = None
        for repetition in range(latency_repetitions):
            with capture_provider_usage() as provider_usage:
                started = time.perf_counter()
                result = pipeline.query(artifact_path, item.question, mode=mode)
                timings.append((time.perf_counter() - started) * 1000)
            usage = provider_usage.to_dict()
            if repetition == 0:
                prediction, pipeline_provider = result, usage
            else:
                repeated_usage.append(usage)
        assert prediction is not None and pipeline_provider is not None
        prediction.metadata["latency_ms"] = round(statistics.median(timings), 3)
        prediction.metadata["latency_measurements_ms"] = [round(value, 3) for value in timings]
        prediction.metadata["provider_usage"] = pipeline_provider
        prediction.metadata["pipeline_provider_usage"] = pipeline_provider
        if repeated_usage:
            prediction.metadata["latency_measurement_usage"] = repeated_usage
        prediction.metadata["cost_estimate"] = {
            "currency": "USD",
            "amount": pipeline_provider["estimated_cost"],
            "basis": "one pipeline request; excludes judge and repeated latency measurements",
            "status": pipeline_provider["cost_status"],
        }
        if judge is not None:
            with capture_provider_usage() as judge_usage:
                try:
                    judge_result = judge.judge(item, prediction)
                    prediction.metadata["judge"] = judge_result.to_dict()
                except Exception as exc:  # evaluator failure must not become a zero-quality answer
                    prediction.metadata["judge"] = {
                        "status": "provider_failure",
                        "error_type": type(exc).__name__,
                        "notes": [str(exc)],
                    }
            evaluation_provider = judge_usage.to_dict()
            prediction.metadata["evaluation_provider_usage"] = evaluation_provider
            prediction.metadata["evaluation_cost_estimate"] = {
                "currency": "USD",
                "amount": evaluation_provider["estimated_cost"],
                "basis": "LLM judge only; excluded from technique cost",
                "status": evaluation_provider["cost_status"],
            }
        predictions.append(prediction)

    evaluate_citations = profile == "citation_rag"
    metrics_by_cutoff = {
        str(cutoff): evaluate_predictions(items, predictions, k=cutoff, include_citation_metrics=evaluate_citations)
        for cutoff in cutoffs
    }
    query_metrics_by_cutoff = {
        str(cutoff): evaluate_prediction_rows(items, predictions, k=cutoff, include_citation_metrics=evaluate_citations)
        for cutoff in cutoffs
    }
    metrics = metrics_by_cutoff[str(top_k)]
    query_metrics = query_metrics_by_cutoff[str(top_k)]
    artifact_manifest = load_manifest(artifact_path)
    effective_components = sorted(
        {
            canonical_fingerprint(prediction.metadata.get("components", {})): prediction.metadata.get("components", {})
            for prediction in predictions
        }.values(),
        key=lambda value: str(value),
    )
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "run_metadata": _run_metadata(
            pipeline,
            artifact_path,
            dataset_path,
            resolved_dataset_path,
            rows,
            dict(artifact_manifest),
            mode,
            top_k,
            judge is not None,
            evaluation_overrides,
            profile,
            seed,
            suite_metadata,
            warmup_queries,
            latency_repetitions,
        ),
        "metrics": metrics,
        "metrics_by_cutoff": metrics_by_cutoff,
        "query_metrics": query_metrics,
        "query_metrics_by_cutoff": query_metrics_by_cutoff,
        "effective_components": effective_components,
        "cost_summary": _cost_summary(predictions),
        "failures": _failures(items, predictions, top_k, check_citations=evaluate_citations),
        "predictions": [
            {
                "question_id": item.question_id,
                "question": item.question,
                "answer": prediction.answer,
                "abstained": prediction.abstained,
                "citations": prediction.citations,
                "contexts": [context.to_dict() for context in prediction.contexts],
                "metadata": prediction.metadata,
            }
            for item, prediction in zip(items, predictions, strict=True)
        ],
    }
    report["result_fingerprint"] = _result_fingerprint(items, predictions, metrics)
    write_json(output_path, report)
    return report


def _ensure_evaluation_depth(pipeline: BasePipeline, top_k: int) -> dict[str, int]:
    """Ensure a technique returns enough ranked evidence for Recall@K/MRR@K.

    Pipeline parameters in this list are explicitly query-time overrides.  We
    only increase them, preserving a technique's wider candidate pool while
    preventing a requested metric cutoff from silently exceeding the number of
    results produced by retrieval or reranking.
    """
    adjusted: dict[str, int] = {}
    for name in ("top_k", "candidate_k", "rerank_top_k", "per_tool_top_k", "fusion_per_query_top_k"):
        if name not in pipeline.query_override_fields or not hasattr(pipeline, name):
            continue
        current = getattr(pipeline, name)
        if isinstance(current, int) and current < top_k:
            setattr(pipeline, name, top_k)
            adjusted[name] = top_k
    return adjusted


def _result_fingerprint(items: list[EvalItem], predictions: list[RAGAnswer], metrics: dict[str, Any]) -> str:
    """Fingerprint deterministic outputs, excluding wall-clock measurements."""
    stable_metrics = {key: value for key, value in metrics.items() if not key.startswith("latency_ms_")}
    stable_predictions = []
    for item, prediction in zip(items, predictions, strict=True):
        stable_predictions.append(
            {
                "question_id": item.question_id,
                "answer": prediction.answer,
                "abstained": prediction.abstained,
                "citations": prediction.citations,
                "contexts": [
                    {
                        "node_id": context.node_id,
                        "chunk_id": context.chunk_id,
                        "doc_id": context.doc_id,
                        "score": context.score,
                        "rank": context.rank,
                    }
                    for context in prediction.contexts
                ],
                "components": prediction.metadata.get("components", {}),
                "verification": prediction.metadata.get("verification", {}),
            }
        )
    return canonical_fingerprint({"metrics": stable_metrics, "predictions": stable_predictions})


def _run_metadata(
    pipeline: BasePipeline,
    artifact_path: str,
    dataset_input_path: str,
    dataset_path: str,
    dataset_rows: list[dict[str, Any]],
    artifact_manifest: dict[str, Any],
    mode: str,
    top_k: int,
    judge_enabled: bool,
    evaluation_overrides: dict[str, int],
    profile: str,
    seed: int | None,
    suite_metadata: dict[str, str] | None,
    warmup_queries: int,
    latency_repetitions: int,
) -> dict[str, Any]:
    dataset_manifest_path = Path(dataset_path).parent / "manifest.json"
    dataset_manifest = read_json(dataset_manifest_path) if dataset_manifest_path.exists() else {}
    return {
        "pipeline_id": pipeline.id,
        "pipeline_name": getattr(pipeline, "name", pipeline.id),
        "artifact_path": artifact_path,
        "artifact_fingerprint": artifact_manifest["corpus"]["fingerprint"],
        "pipeline_config_fingerprint": artifact_manifest["pipeline"]["config_fingerprint"],
        "dataset_path": dataset_path,
        "dataset_input_path": dataset_input_path,
        "dataset_fingerprint": dataset_manifest.get("fingerprint") or canonical_fingerprint(dataset_rows),
        "seed": seed if seed is not None else dataset_manifest.get("metadata", {}).get("seed"),
        "suite": suite_metadata,
        "mode": mode,
        "evaluation_profile": profile,
        "top_k": top_k,
        "evaluation_query_overrides": evaluation_overrides,
        "judge_enabled": judge_enabled,
        "latency_protocol": {
            "warmup_queries": warmup_queries,
            "repetitions_per_query": latency_repetitions,
            "statistic": "median",
        },
        "raglab_version": __version__,
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "git": _git_metadata(),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("litellm", "numpy", "faiss-cpu", "sentence-transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True, timeout=2
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, check=True, text=True, timeout=2
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _cost_summary(predictions: list[RAGAnswer]) -> dict[str, Any]:
    pipeline_costs: list[float] = []
    evaluation_costs: list[float] = []
    for prediction in predictions:
        cost = prediction.metadata.get("cost_estimate", {})
        pipeline_costs.append(float(cost.get("amount", 0.0)) if isinstance(cost, dict) else 0.0)
        evaluation_cost = prediction.metadata.get("evaluation_cost_estimate", {})
        evaluation_costs.append(float(evaluation_cost.get("amount", 0.0)) if isinstance(evaluation_cost, dict) else 0.0)
    return {
        "currency": "USD",
        "total_estimated_cost": round(sum(pipeline_costs), 8),
        "avg_estimated_cost": round(sum(pipeline_costs) / len(pipeline_costs), 8) if pipeline_costs else 0.0,
        "evaluation_total_estimated_cost": round(sum(evaluation_costs), 8),
        "evaluation_avg_estimated_cost": round(sum(evaluation_costs) / len(evaluation_costs), 8)
        if evaluation_costs
        else 0.0,
    }


def _failures(
    items: list[EvalItem],
    predictions: list[RAGAnswer],
    top_k: int,
    check_citations: bool = True,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for item, prediction in zip(items, predictions, strict=True):
        contexts = prediction.contexts[:top_k]
        retrieved_docs = {context.doc_id for context in contexts}
        retrieved_chunks = {context.chunk_id for context in contexts}
        missing_docs = sorted(set(item.expected_doc_ids) - retrieved_docs)
        missing_chunks = sorted(set(item.expected_chunk_ids) - retrieved_chunks)
        expected_citations = set(item.expected_citations)
        predicted_citations = set(prediction.citations)
        types: list[str] = []
        is_answerable = bool(item.metadata.get("is_answerable", True))
        if missing_docs or missing_chunks:
            types.append("retrieval_miss")
        if check_citations and is_answerable and expected_citations:
            if not predicted_citations:
                types.append("citation_missing")
            elif not expected_citations & predicted_citations:
                types.append("citation_wrong_document")
        if is_answerable and prediction.abstained:
            types.append("incorrect_abstention")
        if not is_answerable and not prediction.abstained:
            types.append("missed_abstention")
        verification = prediction.metadata.get("verification", {})
        if check_citations and isinstance(verification, dict) and verification.get("status") == "run":
            if not verification.get("grounded", False) and prediction.answer and not prediction.abstained:
                types.append("unsupported_answer")
        if not types:
            continue
        failures.append(
            {
                "question_id": item.question_id,
                "question": item.question,
                "types": sorted(set(types)),
                "severity": _severity(types),
                "expected": {
                    "doc_ids": item.expected_doc_ids,
                    "chunk_ids": item.expected_chunk_ids,
                    "citations": item.expected_citations,
                    "is_answerable": is_answerable,
                },
                "predicted": {
                    "doc_ids": sorted(retrieved_docs),
                    "chunk_ids": sorted(retrieved_chunks),
                    "citations": prediction.citations,
                    "abstained": prediction.abstained,
                },
                "components": prediction.metadata.get("components", {}),
            }
        )
    return sorted(failures, key=lambda row: row["severity"], reverse=True)


def _severity(types: list[str]) -> int:
    weights = {
        "provider_or_component_failure": 5,
        "retrieval_miss": 4,
        "unsupported_answer": 4,
        "missed_abstention": 4,
        "incorrect_abstention": 3,
        "citation_wrong_document": 3,
        "citation_missing": 2,
    }
    return max((weights.get(item, 1) for item in types), default=0)
