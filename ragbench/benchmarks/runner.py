from __future__ import annotations

import csv
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from ragbench.benchmarks.statistics import paired_bootstrap_delta
from ragbench.benchmarks.suites import claim_eligibility, dataset_manifest, is_git_dirty, load_suite, resolve_suite
from ragbench.core.base import load_pipeline, load_pipeline_for_artifact
from ragbench.core.doctor import diagnose_technique
from ragbench.core.io import read_jsonl, write_json
from ragbench.core.measure import canonical_fingerprint
from ragbench.core.schema import ArtifactManifest
from ragbench.datasets.schema import resolve_eval_dataset_path
from ragbench.evaluation.runner import run_eval
from ragbench.indexing.artifacts import load_manifest


def run_benchmarks(
    *,
    technique_ids: list[str],
    docs: str | None,
    qa: str | None,
    output: str,
    mode: str | None = None,
    top_k: int | None = None,
    profile: str = "auto",
    resume: bool = False,
    seed: int = 42,
    suite_path: str | None = None,
    judge_spec: dict[str, Any] | None = None,
    warmup_queries: int | None = None,
    latency_repetitions: int = 1,
    max_estimated_cost_usd: float | None = None,
    concurrency: int | None = None,
    latency_sample_size: int | None = None,
) -> dict[str, Any]:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not suite_path and (not docs or not qa):
        raise ValueError("docs and qa are required unless --suite supplies them")
    suite = load_suite(suite_path) if suite_path else None
    if suite:
        # ``warmup_queries=None``/``concurrency=None``/``latency_sample_size=None``
        # mean "the caller did not explicitly ask for a value" — only a value
        # the caller actually passed can conflict with a suite lock. Otherwise
        # a claim-eligible suite's lock would reject its own CLI default every
        # time.
        resolved = resolve_suite(
            suite,
            docs=docs,
            qa=qa,
            mode=mode,
            top_k=top_k,
            warmup_queries=warmup_queries,
            concurrency=concurrency,
            latency_sample_size=latency_sample_size,
        )
        docs, qa = str(resolved["docs"]), str(resolved["qa"])
        mode, top_k = str(resolved["mode"]), int(resolved["top_k"])
        if "warmup_queries" in resolved:
            warmup_queries = int(resolved["warmup_queries"])
        if "concurrency" in resolved:
            concurrency = int(resolved["concurrency"])
        if "latency_sample_size" in resolved:
            latency_sample_size = int(resolved["latency_sample_size"])
        profile = str(suite.get("profile", profile))
        missing = sorted(set(suite["required_baselines"]) - set(technique_ids))
        if missing:
            raise ValueError(f"Suite requires techniques: {', '.join(missing)}")
    mode = mode or "full_rag"
    top_k = top_k or 5
    warmup_queries = warmup_queries if warmup_queries is not None else 0
    concurrency = concurrency if concurrency is not None else 1
    latency_sample_size = latency_sample_size if latency_sample_size is not None else 5
    cutoffs = list(suite.get("cutoffs", [top_k])) if suite else [top_k]
    bootstrap_samples = int(suite.get("bootstrap_samples", 10_000)) if suite else 10_000
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Do NOT call random.seed(seed) here (removed): it mutated *global*
    # random state for the rest of the process — every unrelated random.random()
    # call downstream (e.g. the LLM client's retry-backoff jitter) silently
    # became a function of this benchmark's seed, while the one thing a
    # researcher actually wants seeded — LLM generation — was never touched
    # by it at all (see providers.llm_client.generation_seed for the real
    # mechanism). paired_bootstrap_delta already takes its own explicit
    # `seed=` below and uses an isolated random.Random(seed), so nothing here
    # depended on the global call in the first place.
    rows: list[dict[str, Any]] = []
    assert docs is not None and qa is not None
    dataset_fingerprint = _dataset_fingerprint(qa)
    for technique_id in technique_ids:
        artifact = output_dir / "artifacts" / technique_id
        index_time_ms: float | None = None
        try:
            manifest: ArtifactManifest | None = load_manifest(artifact) if resume and artifact.exists() else None
            report = (
                _matching_report(
                    output_dir,
                    technique_id,
                    manifest,
                    dataset_fingerprint,
                    mode,
                    top_k,
                    profile,
                    seed=seed,
                    suite_fingerprint=suite.get("suite_fingerprint") if suite else None,
                    judge_enabled=judge_spec is not None,
                    warmup_queries=warmup_queries,
                    latency_repetitions=latency_repetitions,
                    concurrency=concurrency,
                    latency_sample_size=latency_sample_size,
                )
                if resume and manifest
                else None
            )
            if report is not None:
                evaluation = json.loads(report.read_text(encoding="utf-8"))
            else:
                if manifest is None:
                    pipeline = load_pipeline(technique_id)
                    if pipeline is None:
                        raise RuntimeError(f"Unknown bundled technique '{technique_id}'")
                    ingest_started = time.perf_counter()
                    manifest = pipeline.ingest(docs, str(artifact))
                    index_time_ms = round((time.perf_counter() - ingest_started) * 1000, 3)
                # else: --resume and a valid artifact already exists but no
                # report matched (the prior attempt never finished eval, or
                # finished under different eval params) — reuse the existing
                # artifact instead of paying to ingest again; run_eval()'s own
                # per-query checkpoint resumes from wherever it stopped.
                fingerprint = manifest["corpus"]["fingerprint"].split(":", 1)[-1][:10]
                report = output_dir / f"{technique_id}_{fingerprint}_eval.json"
                query_pipeline = load_pipeline_for_artifact(technique_id, artifact)
                evaluation = run_eval(
                    query_pipeline,
                    str(artifact),
                    qa,
                    str(report),
                    top_k=top_k,
                    mode=mode,
                    profile=profile,
                    seed=seed,
                    cutoffs=cutoffs,
                    suite_metadata={"id": suite["id"], "fingerprint": suite["suite_fingerprint"]} if suite else None,
                    judge_spec=judge_spec,
                    warmup_queries=warmup_queries,
                    latency_repetitions=latency_repetitions,
                    max_estimated_cost_usd=max_estimated_cost_usd,
                    concurrency=concurrency,
                    latency_sample_size=latency_sample_size,
                    profile_coverage=suite.get("coverage") if suite else None,
                )
                evaluation["index"] = {
                    "build_time_ms": index_time_ms,
                    "size_bytes": sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file()),
                }
                write_json(report, evaluation)
            assert manifest is not None and report is not None
            rows.append(_row(technique_id, artifact, report, manifest, evaluation, "ok", index_time_ms=index_time_ms))
        except Exception as exc:  # one failed technique must not hide the remaining runs
            rows.append(
                {
                    "technique": technique_id,
                    "artifact": str(artifact),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    warnings = _comparison_warnings(rows)
    comparisons = _comparisons(
        rows,
        top_k=top_k,
        cutoffs=cutoffs,
        seed=seed,
        reference_baseline=(
            str(suite.get("reference_baseline")) if suite and suite.get("reference_baseline") else None
        ),
        bootstrap_samples=bootstrap_samples,
    )
    eligibility = claim_eligibility(suite, rows, qa, comparisons)
    payload = {
        "runs": rows,
        "warnings": warnings,
        "seed": seed,
        "comparisons": comparisons,
        "suite": suite,
        "claim_eligibility": eligibility,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, payload)
    _write_csv(output_dir / "summary.csv", rows)
    _write_markdown(output_dir / "summary.md", rows, warnings, top_k=top_k)
    return {**payload, "summary": str(summary_path)}


def run_preflight(
    *,
    technique_ids: list[str],
    docs: str | None,
    qa: str | None,
    mode: str | None = None,
    top_k: int | None = None,
    suite_path: str | None = None,
    warmup_queries: int | None = None,
    concurrency: int | None = None,
    latency_sample_size: int | None = None,
) -> dict[str, Any]:
    """Check everything that can fail *before* spending an ingest/query/API call.

    Runs the suite/dataset resolution, the git-dirty and dataset-eligibility
    gates from ``claim_eligibility`` (the ones that do not depend on any
    run's results), and ``doctor`` for every required technique — all without
    ingesting a single document. A long benchmark should never discover a
    missing API key or a dirty worktree after hours of work; it should
    discover that in seconds, here.
    """
    reasons: list[str] = []
    if not suite_path and (not docs or not qa):
        raise ValueError("docs and qa are required unless --suite supplies them")
    suite = load_suite(suite_path) if suite_path else None
    if suite:
        try:
            resolved = resolve_suite(
                suite,
                docs=docs,
                qa=qa,
                mode=mode,
                top_k=top_k,
                warmup_queries=warmup_queries,
                concurrency=concurrency,
                latency_sample_size=latency_sample_size,
            )
            docs, qa = str(resolved["docs"]), str(resolved["qa"])
            mode = str(resolved["mode"])
            if "warmup_queries" in resolved:
                warmup_queries = int(resolved["warmup_queries"])
            if "concurrency" in resolved:
                concurrency = int(resolved["concurrency"])
            if "latency_sample_size" in resolved:
                latency_sample_size = int(resolved["latency_sample_size"])
        except ValueError as exc:
            reasons.append(str(exc))
        missing = sorted(set(suite.get("required_baselines", [])) - set(technique_ids))
        if missing:
            reasons.append(f"suite requires techniques not in --techniques: {', '.join(missing)}")
        if suite.get("tier") == "claim_eligible" and importlib.util.find_spec("faiss") is None:
            # FaissLocalVectorStore is exact search (IndexFlatIP), not approximate —
            # a silent json_memory fallback would not change any ranking, only
            # speed. But a claim-eligible run must not have its backend chosen by
            # what happens to be installed on this machine: if the corpus crosses
            # RAGLAB_FAISS_NODE_THRESHOLD after chunking, this environment would
            # silently run json_memory and fail claim_eligibility() after the
            # fact. Catch that here, in seconds, instead of after a full run.
            reasons.append(
                "claim_eligible suite requires faiss installed (pip install '.[vector]') in case the corpus "
                "crosses RAGLAB_FAISS_NODE_THRESHOLD after chunking"
            )
    mode = mode or "full_rag"

    if is_git_dirty():
        reasons.append("git worktree is dirty")
    if qa:
        manifest = dataset_manifest(qa)
        metadata = manifest.get("metadata", {})
        policy = metadata.get("corpus_policy") if isinstance(metadata, dict) else None
        if suite and suite.get("tier") == "claim_eligible" and policy != "full_upstream_corpus":
            reasons.append(f"dataset corpus policy is {policy or 'unspecified'}")
        minimum = int(suite.get("minimum_queries", 0)) if suite else 0
        if minimum and int(manifest.get("queries", 0)) < minimum:
            reasons.append(f"dataset has {manifest.get('queries', 0)} queries; suite requires {minimum}")

    technique_checks: list[dict[str, Any]] = []
    for technique_id in technique_ids:
        pipeline = load_pipeline(technique_id)
        if pipeline is None:
            reasons.append(f"unknown bundled technique '{technique_id}'")
            continue
        diagnosis = diagnose_technique(pipeline, mode=mode)
        technique_checks.append(diagnosis)
        if not diagnosis["ready"]:
            failed = "; ".join(check["detail"] for check in diagnosis["checks"] if check["status"] == "failed")
            reasons.append(f"{technique_id} not ready: {failed}")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "mode": mode,
        "docs": docs,
        "qa": qa,
        "suite": suite,
        "technique_checks": technique_checks,
    }


def has_failed_runs(result: dict[str, Any]) -> bool:
    """Return whether a benchmark summary contains a failed technique run."""
    return any(run.get("status") == "failed" for run in result.get("runs", []))


def _dataset_fingerprint(dataset_path: str) -> str:
    resolved = resolve_eval_dataset_path(dataset_path)
    manifest_path = Path(resolved).parent / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("fingerprint"):
            return str(payload["fingerprint"])
    return canonical_fingerprint(read_jsonl(resolved))


def _matching_report(
    output_dir: Path,
    technique_id: str,
    manifest: ArtifactManifest | None,
    dataset_fingerprint: str,
    mode: str,
    top_k: int,
    profile: str,
    seed: int | None = None,
    suite_fingerprint: str | None = None,
    judge_enabled: bool = False,
    warmup_queries: int = 0,
    latency_repetitions: int = 1,
    concurrency: int = 1,
    latency_sample_size: int = 5,
) -> Path | None:
    if manifest is None:
        return None
    for candidate in sorted(output_dir.glob(f"{technique_id}_*_eval.json"), reverse=True):
        report = json.loads(candidate.read_text(encoding="utf-8"))
        metadata = report.get("run_metadata", {})
        latency_protocol = metadata.get("latency_protocol", {})
        if (
            report.get("report_schema_version") == "2"
            and metadata.get("artifact_fingerprint") == manifest["corpus"]["fingerprint"]
            and metadata.get("pipeline_config_fingerprint") == manifest["pipeline"]["config_fingerprint"]
            and metadata.get("dataset_fingerprint") == dataset_fingerprint
            and metadata.get("mode") == mode
            and metadata.get("top_k") == top_k
            and metadata.get("evaluation_profile") == _resolved_profile(profile, mode, candidate)
            and (seed is None or metadata.get("seed") == seed)
            and (suite_fingerprint is None or metadata.get("suite", {}).get("fingerprint") == suite_fingerprint)
            and metadata.get("judge_enabled", False) == judge_enabled
            and latency_protocol.get("warmup_queries", 0) == warmup_queries
            and latency_protocol.get("repetitions_per_query", 1) == latency_repetitions
            # A report produced under a different concurrency/latency-sample
            # protocol isn't reusable: its latency_ms_p95 (and, for a resumed
            # run, which predictions counted as the trustworthy sequential
            # sample) was computed under a different meaning of "trustworthy."
            and latency_protocol.get("concurrency", 1) == concurrency
            and latency_protocol.get("latency_sample_size", 5) == latency_sample_size
        ):
            return candidate
    return None


def _resolved_profile(requested: str, mode: str, report_path: Path) -> str:
    if requested != "auto":
        return requested
    if mode == "retrieval_only":
        return "retrieval"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return "citation_rag" if report.get("metrics", {}).get("citation_queries_evaluated", 0) else "single_hop_rag"


def _row(
    technique: str,
    artifact: Path,
    report: Path,
    manifest: ArtifactManifest,
    evaluation: dict[str, Any],
    status: str,
    index_time_ms: float | None = None,
) -> dict[str, Any]:
    components = sorted(
        {
            json.dumps(item.get("metadata", {}).get("components", {}), sort_keys=True)
            for item in evaluation.get("predictions", [])
        }
    )
    cutoff_metrics = evaluation.get("metrics_by_cutoff", {})
    flattened_cutoffs = {
        metric: value
        for values in cutoff_metrics.values()
        if isinstance(values, dict)
        for metric, value in values.items()
        if metric.startswith(("recall_at_", "ndcg_at_", "map_at_", "context_precision_at_"))
    }
    cost_statuses = {
        item.get("metadata", {}).get("cost_estimate", {}).get("status")
        for item in evaluation.get("predictions", [])
        if isinstance(item.get("metadata", {}).get("cost_estimate"), dict)
    }
    return {
        "technique": technique,
        "artifact": str(artifact),
        "report": str(report),
        "status": status,
        "node_count": manifest["corpus"]["node_count"],
        "artifact_fingerprint": manifest["corpus"]["fingerprint"],
        "config_fingerprint": manifest["pipeline"]["config_fingerprint"],
        "effective_components": ";".join(components),
        "index_time_ms": evaluation.get("index", {}).get("build_time_ms", index_time_ms),
        "index_size_bytes": evaluation.get("index", {}).get(
            "size_bytes", sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())
        ),
        "store_backend": manifest["store"].get("backend"),
        "cost_status": "estimated" if cost_statuses == {"estimated"} else "unknown",
        **evaluation.get("metrics", {}),
        **flattened_cutoffs,
    }


def _comparison_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        if "lexical_overlap_fallback" in str(row.get("effective_components", "")):
            warnings.append(f"{row.get('technique')} used a fallback implementation")
    return warnings


def _comparisons(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    cutoffs: list[int],
    seed: int = 42,
    reference_baseline: str | None = None,
    bootstrap_samples: int = 10_000,
) -> list[dict[str, Any]]:
    """Report comparable metric deltas against the first successful baseline."""
    successful = [row for row in rows if row.get("status") == "ok"]
    if len(successful) < 2:
        return []
    baseline = next((row for row in successful if row["technique"] == reference_baseline), successful[0])
    metrics = (
        "mrr",
        *(f"recall_at_{cutoff}" for cutoff in cutoffs),
        *(f"ndcg_at_{cutoff}" for cutoff in cutoffs),
        *(f"map_at_{cutoff}" for cutoff in cutoffs),
        *(f"context_precision_at_{cutoff}" for cutoff in cutoffs),
        "evidence_complete_rate",
        f"recall_at_{top_k}",
        "citation_f1",
        "latency_ms_avg",
        "estimated_cost_avg",
    )
    comparisons: list[dict[str, Any]] = []
    for candidate in successful:
        if candidate is baseline:
            continue
        deltas = {
            metric: round(float(candidate[metric]) - float(baseline[metric]), 6)
            for metric in metrics
            if isinstance(candidate.get(metric), int | float) and isinstance(baseline.get(metric), int | float)
        }
        base_report = json.loads(Path(str(baseline["report"])).read_text(encoding="utf-8"))
        candidate_report = json.loads(Path(str(candidate["report"])).read_text(encoding="utf-8"))
        intervals = {}
        for metric in metrics:
            cutoff = _metric_cutoff(metric)
            base_rows = _query_rows_for_metric(base_report, cutoff)
            candidate_rows = _query_rows_for_metric(candidate_report, cutoff)
            query_metric = _query_metric_name(metric)
            if any(query_metric in row for row in base_rows) and any(query_metric in row for row in candidate_rows):
                intervals[metric] = paired_bootstrap_delta(
                    list(base_rows), list(candidate_rows), query_metric, seed=seed, samples=bootstrap_samples
                )
        comparisons.append(
            {
                "baseline": baseline["technique"],
                "candidate": candidate["technique"],
                "deltas": deltas,
                "paired_ci95": intervals,
            }
        )
    return comparisons


def _metric_cutoff(metric: str) -> int | None:
    if "_at_" not in metric:
        return None
    try:
        return int(metric.rsplit("_at_", 1)[1])
    except ValueError:
        return None


def _query_metric_name(metric: str) -> str:
    if "_at_" in metric:
        return metric.split("_at_", 1)[0]
    return {
        "evidence_complete_rate": "evidence_complete",
        "citation_f1": "citation_document_f1",
        "latency_ms_avg": "latency_ms",
        "estimated_cost_avg": "estimated_cost",
    }.get(metric, metric)


def _query_rows_for_metric(report: dict[str, Any], cutoff: int | None) -> list[dict[str, Any]]:
    if cutoff is None:
        return list(report.get("query_metrics", []))
    return list(report.get("query_metrics_by_cutoff", {}).get(str(cutoff), []))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], warnings: list[str], *, top_k: int) -> None:
    columns = [
        "technique",
        "status",
        f"recall_at_{top_k}",
        "mrr",
        f"ndcg_at_{top_k}",
        "citation_f1",
        "abstention_accuracy",
        "answer_correctness",
        "faithfulness",
        "latency_ms_avg",
        "latency_ms_p95",
        "estimated_cost_avg",
    ]
    lines = ["# Benchmark Summary", ""]
    if warnings:
        lines.extend(["## Warnings", "", *(f"- {item}" for item in warnings), ""])
    lines.extend(["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"])
    lines.extend("|" + "|".join(str(row.get(column, "")) for column in columns) + "|" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
