from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ragbench.core.base import (
    BasePipeline,
    get_pipeline_metadata,
    list_pipelines,
    load_pipeline,
    load_pipeline_for_artifact,
)
from ragbench.core.doctor import diagnose_technique
from ragbench.core.io import write_json
from ragbench.datasets.adapters import DATASET_ADAPTERS, prepare_dataset
from ragbench.datasets.golden import validate_golden_dataset
from ragbench.datasets.schema import (
    PROTOCOL_SPLITS,
    sample_processed_dataset,
    validate_processed_dataset,
)
from ragbench.datasets.synthetic import (
    ROLE_ENV_VARS,
    GenerationConfig,
    SyntheticBenchmarkBuilder,
    resolve_role_model,
    role_collisions,
    validate_synthetic_dataset,
)
from ragbench.evaluation.audit import DEFAULT_METRICS as DEFAULT_AUDIT_METRICS
from ragbench.evaluation.audit import run_audit, write_label_sample
from ragbench.evaluation.runner import BudgetExceededError, run_eval
from ragbench.indexing.artifacts import inspect_artifact


def _parse_params(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--param key=value`` flags into a kwargs dict.

    Values are JSON-decoded when possible (so ``--param top_k=10`` yields
    ``{"top_k": 10}``); plain strings stay strings.
    """
    kwargs: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        try:
            kwargs[key] = json.loads(raw)
        except json.JSONDecodeError:
            kwargs[key] = raw
    return kwargs


def _resolve_pipeline(args: argparse.Namespace) -> BasePipeline:
    technique_id = getattr(args, "technique", None)
    if not technique_id:
        raise SystemExit("Provide --technique TECHNIQUE_ID.")
    kwargs = _parse_params(getattr(args, "param", None))
    pipeline = load_pipeline(technique_id, params=kwargs)
    if pipeline is None:
        raise SystemExit(
            f"No pipeline.py found for '{technique_id}'. Use `ragbench techniques list` to see bundled technique ids."
        )
    return pipeline


def _resolve_artifact_pipeline(args: argparse.Namespace, *, interactive: bool = False) -> BasePipeline:
    overrides = _parse_params(getattr(args, "param", None))
    if interactive and getattr(args, "allow_fallback", False):
        overrides["allow_fallback"] = True
    if not interactive and overrides.get("allow_fallback"):
        raise SystemExit("Evaluation and benchmark runs do not allow component fallback.")
    try:
        return load_pipeline_for_artifact(args.technique, args.artifact, overrides)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


def _add_technique_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--technique",
        required=True,
        metavar="TECHNIQUE_ID",
        help="Technique id (directory name under techniques/), e.g. hyde_2022",
    )
    p.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        help="Override a pipeline constructor parameter (repeatable). Values are JSON-parsed.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="ragbench", description="RAG Pipeline Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse, chunk, enrich, and save index artifacts")
    _add_technique_args(ingest_parser)
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--output", required=True)
    ingest_parser.add_argument(
        "--no-embedding-cache", action="store_true", help="Disable the persistent (model, text) embedding cache"
    )

    query_parser = subparsers.add_parser("query", help="Run a single query against saved artifacts")
    _add_technique_args(query_parser)
    query_parser.add_argument("--artifact", required=True)
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    query_parser.add_argument(
        "--allow-fallback", action="store_true", help="Allow demo-only lexical fallback when a reranker is unavailable"
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluate a pipeline on a JSONL QA dataset")
    _add_technique_args(eval_parser)
    eval_parser.add_argument("--artifact", required=True)
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    eval_parser.add_argument(
        "--profile", choices=["auto", "retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"], default="auto"
    )
    eval_parser.add_argument("--judge", action="store_true", help="Enable OpenAI-compatible LLM-as-judge metrics")
    eval_parser.add_argument("--judge-model", help=f"Judge model (default: ${ROLE_ENV_VARS['judge']}, else CHAT_MODEL)")
    eval_parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help="Allow the judge to share a model with the dataset's generator or verifier, recording the "
        "circularity instead of refusing.",
    )
    eval_parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=None,
        help="Abort once estimated pipeline+judge cost exceeds this many USD (checked after each query). "
        "No-op unless pricing env vars are configured for every call type this run makes.",
    )
    eval_parser.add_argument(
        "--no-embedding-cache", action="store_true", help="Disable the persistent (model, text) embedding cache"
    )
    eval_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run the quality pass with this many concurrent workers (default 1 = fully sequential, "
        "unchanged behavior). Above 1, a sequential --latency-sample-size prefix runs first for honest "
        "per-request latency, then the remainder runs concurrently for throughput.",
    )
    eval_parser.add_argument(
        "--latency-sample-size",
        type=int,
        default=5,
        help="Number of queries run sequentially before switching to concurrency (only used when "
        "--concurrency > 1); source of the report's latency_pass p50/p95.",
    )

    compare_parser = subparsers.add_parser("compare", help="Evaluate multiple technique/artifact pairs")
    compare_parser.add_argument("--runs", nargs="+", required=True, help="Items in technique_id=artifact format")
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--top-k", type=int, default=5)

    bench_parser = subparsers.add_parser("bench", help="Run benchmarks across technique ids")
    bench_parser.add_argument("--techniques", nargs="+", required=True)
    bench_parser.add_argument("--docs")
    bench_parser.add_argument("--qa")
    bench_parser.add_argument("--output", help="Required unless --preflight")
    bench_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"])
    bench_parser.add_argument("--top-k", type=int)
    bench_parser.add_argument(
        "--profile", choices=["auto", "retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"], default="auto"
    )
    bench_parser.add_argument("--resume", action="store_true")
    bench_parser.add_argument("--seed", type=int, default=42)
    bench_parser.add_argument("--suite", help="Path to a machine-readable benchmark suite contract")
    bench_parser.add_argument("--judge", action="store_true", help="Enable OpenAI-compatible LLM judge metrics")
    bench_parser.add_argument(
        "--judge-model", help=f"Judge model (default: ${ROLE_ENV_VARS['judge']}, else CHAT_MODEL)"
    )
    bench_parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help="Allow the judge to share a model with the dataset's generator or verifier, recording the "
        "circularity instead of refusing.",
    )
    bench_parser.add_argument(
        "--warmup-queries",
        type=int,
        default=None,
        help="Default 0 unless a suite locks a value; explicit values conflicting with a locked suite are rejected",
    )
    bench_parser.add_argument("--latency-repetitions", type=int, default=1)
    bench_parser.add_argument(
        "--max-estimated-cost-usd",
        type=float,
        default=None,
        help="Abort a technique's run once its estimated pipeline+judge cost exceeds this many USD "
        "(checked after each query, applies per technique). No-op unless pricing env vars are configured "
        "for every call type that technique's run makes.",
    )
    bench_parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check suite/dataset/provider/dependency readiness only; do not ingest or query anything",
    )
    bench_parser.add_argument(
        "--offline",
        action="store_true",
        help="With --preflight, skip live provider probes and check only that required env vars are set "
        "(no network calls) — for CI/debugging without network access",
    )
    bench_parser.add_argument(
        "--no-embedding-cache", action="store_true", help="Disable the persistent (model, text) embedding cache"
    )
    bench_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Run each technique's quality pass with this many concurrent workers. Default 1 (fully "
        "sequential) unless a suite locks a value; explicit values conflicting with a locked suite are rejected.",
    )
    bench_parser.add_argument(
        "--latency-sample-size",
        type=int,
        default=None,
        help="Number of queries run sequentially before switching to concurrency (only used when "
        "--concurrency > 1); source of the report's latency_pass p50/p95. Default 5 unless a suite locks a value.",
    )

    experiment_parser = subparsers.add_parser("experiment", help="Run a repeatable multi-trial benchmark matrix")
    experiment_parser.add_argument("--techniques", nargs="+", required=True)
    experiment_parser.add_argument("--docs", required=True)
    experiment_parser.add_argument("--qa", required=True)
    experiment_parser.add_argument("--output", required=True)
    experiment_parser.add_argument("--trials", type=int, default=1)
    experiment_parser.add_argument("--seed", type=int, default=42)
    experiment_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    experiment_parser.add_argument(
        "--profile", choices=["auto", "retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"], default="auto"
    )
    experiment_parser.add_argument("--top-k", type=int, default=5)
    experiment_parser.add_argument("--judge", action="store_true", help="Enable OpenAI-compatible LLM judge metrics")
    experiment_parser.add_argument(
        "--judge-model", help=f"Judge model (default: ${ROLE_ENV_VARS['judge']}, else CHAT_MODEL)"
    )
    experiment_parser.add_argument("--warmup-queries", type=int, default=0)
    experiment_parser.add_argument("--latency-repetitions", type=int, default=1)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    generate_parser = dataset_subparsers.add_parser(
        "generate", help="Generate a synthetic benchmark from raw documents (see docs/synthetic_benchmark_v2.md)"
    )
    generate_parser.add_argument("--docs", required=True)
    generate_parser.add_argument("--output", required=True, help="Output directory; receives qa.jsonl + manifest.json")
    generate_parser.add_argument("--dataset-id")
    generate_parser.add_argument(
        "--model", help=f"Generator model (default: ${ROLE_ENV_VARS['generator']}, else CHAT_MODEL)"
    )
    generate_parser.add_argument(
        "--verifier-model",
        help=f"Model that checks answerability and grades the pool (default: ${ROLE_ENV_VARS['verifier']}). "
        "Must differ from the generator.",
    )
    generate_parser.add_argument(
        "--pool-embedding-model",
        help="Add dense retrieval to the label pool. Costs one embedding pass over the corpus; without it "
        "the pool is BM25-only and the trust block warns about single-retriever pooling.",
    )
    generate_parser.add_argument("--questions-per-chunk", type=int, default=2)
    generate_parser.add_argument(
        "--max-chunks", type=int, help="Cap generation at this many chunks; the main cost control"
    )
    generate_parser.add_argument("--chunk-size", type=int, default=250)
    generate_parser.add_argument("--chunk-overlap", type=int, default=40)
    generate_parser.add_argument("--overlap-threshold", type=float, default=0.6)
    generate_parser.add_argument("--dedup-threshold", type=float, default=0.92)
    generate_parser.add_argument("--pool-k", type=int, default=20)
    generate_parser.add_argument("--seed", type=int, default=42)
    generate_parser.add_argument(
        "--allow-same-model",
        action="store_true",
        help="Proceed even though two roles share a model. Recorded in the manifest and downgrades every "
        "trust verdict built on this dataset to synthetic_untrusted.",
    )
    prepare_parser = dataset_subparsers.add_parser("prepare", help="Prepare a fixed research dataset for evaluation")
    prepare_parser.add_argument("name", choices=sorted(DATASET_ADAPTERS))
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--split", help="Upstream split to pull from (adapter-specific, e.g. validation/test)")
    prepare_parser.add_argument(
        "--protocol-split",
        choices=list(PROTOCOL_SPLITS),
        default=None,
        help=(
            "This snapshot's role in the benchmark protocol: 'dev' for tuning, 'test' for a held-out claim set. "
            "Distinct from --split (upstream provenance) and never inferred from it. Left unset by default; "
            "claim-eligible runs require 'test'."
        ),
    )
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--overwrite", action="store_true")
    validate_parser = dataset_subparsers.add_parser("validate", help="Validate a processed evaluation dataset")
    validate_parser.add_argument("path")
    validate_golden_parser = dataset_subparsers.add_parser(
        "validate-golden", help="Validate a human-curated end-to-end RAG golden set"
    )
    validate_golden_parser.add_argument("path")
    validate_synthetic_parser = dataset_subparsers.add_parser(
        "validate-synthetic", help="Validate a generated benchmark (labels, split, manifest)"
    )
    validate_synthetic_parser.add_argument("path")
    audit_parser = dataset_subparsers.add_parser(
        "audit", help="Measure how far a synthetic benchmark can be trusted, against a golden set"
    )
    audit_parser.add_argument(
        "--synthetic-runs",
        nargs="+",
        required=True,
        metavar="TECHNIQUE=REPORT",
        help="Eval report per technique, run on the synthetic dataset",
    )
    audit_parser.add_argument(
        "--golden-runs",
        nargs="+",
        required=True,
        metavar="TECHNIQUE=REPORT",
        help="Eval report per technique, run on the golden dataset",
    )
    audit_parser.add_argument("--output", required=True, help="Directory to write audit.json into")
    audit_parser.add_argument("--synthetic-qa", help="Path to the synthetic qa.jsonl; the audit is recorded on it")
    audit_parser.add_argument("--label-review", help="Filled label review file (see --emit-label-sample)")
    audit_parser.add_argument(
        "--emit-label-sample",
        metavar="PATH",
        help="Write a review file of (question, labelled document) pairs and exit. Requires --synthetic-qa.",
    )
    audit_parser.add_argument("--label-sample", type=int, default=40)
    audit_parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_AUDIT_METRICS))
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.add_argument("--bootstrap-samples", type=int, default=1000)
    sample_parser = dataset_subparsers.add_parser("sample", help="Create a smaller processed dataset sample")
    sample_parser.add_argument("path")
    sample_parser.add_argument("--output", required=True)
    sample_parser.add_argument("--limit", type=int, required=True)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.add_argument("--overwrite", action="store_true")
    dataset_subparsers.add_parser("list", help="List available fixed dataset adapters")

    artifacts_parser = subparsers.add_parser("artifacts", help="Artifact utilities")
    artifacts_subparsers = artifacts_parser.add_subparsers(dest="artifacts_command", required=True)
    inspect_parser = artifacts_subparsers.add_parser("inspect", help="Inspect saved artifact metadata")
    inspect_parser.add_argument("--artifact", required=True)

    techniques_parser = subparsers.add_parser("techniques", help="List or inspect paper-driven techniques")
    techniques_subparsers = techniques_parser.add_subparsers(dest="techniques_command", required=True)
    techniques_subparsers.add_parser("list", help="List registered technique metadata")
    show_parser = techniques_subparsers.add_parser("show", help="Show one technique metadata")
    show_parser.add_argument("technique_id")

    doctor_parser = subparsers.add_parser("doctor", help="Check technique dependencies and provider configuration")
    _add_technique_args(doctor_parser)
    doctor_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    doctor_parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip live provider probes and check only that required env vars are set (no network calls)",
    )

    args = parser.parse_args()
    if getattr(args, "no_embedding_cache", False):
        # Simplest way to reach every technique's internal Embedder/LLMClient
        # construction without threading a new constructor kwarg through every
        # technique — the cache is already env-var-driven (RAGLAB_EMBEDDING_CACHE).
        os.environ["RAGLAB_EMBEDDING_CACHE"] = "0"
    if args.command == "ingest":
        pipeline = _resolve_pipeline(args)
        _print(pipeline.ingest(args.input, args.output))
    elif args.command == "query":
        pipeline = _resolve_artifact_pipeline(args, interactive=True)
        pipeline.load(args.artifact)
        answer = pipeline.query(args.query, mode=args.mode)
        _print(answer.to_dict())
    elif args.command == "eval":
        pipeline = _resolve_artifact_pipeline(args)
        args.judge_model = resolve_role_model("judge", args.judge_model)
        _guard_judge_role(args)
        judge_spec = {"type": "openai", "params": {"model": args.judge_model}} if args.judge else None
        try:
            report = run_eval(
                pipeline,
                args.artifact,
                args.dataset,
                args.output,
                top_k=args.top_k,
                mode=args.mode,
                judge_spec=judge_spec,
                profile=args.profile,
                max_estimated_cost_usd=args.max_estimated_cost_usd,
                concurrency=args.concurrency,
                latency_sample_size=args.latency_sample_size,
            )
        except BudgetExceededError as exc:
            raise SystemExit(str(exc)) from exc
        _print(report["metrics"])
    elif args.command == "compare":
        _compare(args.runs, args.dataset, args.output, args.top_k)
    elif args.command == "bench":
        args.judge_model = resolve_role_model("judge", args.judge_model)
        _guard_judge_role(args)
        _bench(args)
    elif args.command == "experiment":
        from ragbench.benchmarks.experiments import run_experiment_matrix

        args.judge_model = resolve_role_model("judge", args.judge_model)
        _guard_judge_role(args)

        _print(
            run_experiment_matrix(
                technique_ids=args.techniques,
                docs=args.docs,
                qa=args.qa,
                output=args.output,
                trials=args.trials,
                seed=args.seed,
                mode=args.mode,
                profile=args.profile,
                top_k=args.top_k,
                judge_spec={"type": "openai", "params": {"model": args.judge_model}} if args.judge else None,
                warmup_queries=args.warmup_queries,
                latency_repetitions=args.latency_repetitions,
            )
        )
    elif args.command == "dataset":
        if args.dataset_command == "generate":
            _print_or_exit(lambda: _generate_synthetic(args))
        elif args.dataset_command == "prepare":
            _print_or_exit(
                lambda: prepare_dataset(
                    args.name,
                    args.output,
                    split=args.split,
                    limit=args.limit,
                    seed=args.seed,
                    overwrite=args.overwrite,
                    protocol_split=args.protocol_split,
                )
            )
        elif args.dataset_command == "validate":
            _print_or_exit(lambda: validate_processed_dataset(args.path))
        elif args.dataset_command == "validate-golden":
            _print_or_exit(lambda: validate_golden_dataset(args.path))
        elif args.dataset_command == "validate-synthetic":
            _print_or_exit(lambda: validate_synthetic_dataset(args.path))
        elif args.dataset_command == "audit":
            _print_or_exit(lambda: _audit(args))
        elif args.dataset_command == "sample":
            _print_or_exit(
                lambda: sample_processed_dataset(
                    args.path, args.output, args.limit, seed=args.seed, overwrite=args.overwrite
                )
            )
        elif args.dataset_command == "list":
            _print({"datasets": sorted(DATASET_ADAPTERS)})
    elif args.command == "artifacts":
        if args.artifacts_command == "inspect":
            _print(inspect_artifact(args.artifact))
    elif args.command == "techniques":
        if args.techniques_command == "list":
            _print({"techniques": list_pipelines()})
        elif args.techniques_command == "show":
            _print(get_pipeline_metadata(args.technique_id))
    elif args.command == "doctor":
        diagnosis = diagnose_technique(_resolve_pipeline(args), mode=args.mode, offline=args.offline)
        if not args.offline:
            for check in diagnosis["checks"]:
                symbol = "✓" if check["status"] == "ok" else "✗"
                latency = f" — {check['latency_ms']:.0f} ms" if check.get("latency_ms") is not None else ""
                print(f"{symbol} {check['name']}: {check['detail']}{latency}")
        _print(diagnosis)


def _generate_synthetic(args: argparse.Namespace) -> dict[str, Any]:
    config = GenerationConfig(
        generator_model=resolve_role_model("generator", args.model),
        verifier_model=resolve_role_model("verifier", args.verifier_model),
        questions_per_chunk=args.questions_per_chunk,
        max_chunks=args.max_chunks,
        seed=args.seed,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        overlap_threshold=args.overlap_threshold,
        dedup_threshold=args.dedup_threshold,
        pool_k=args.pool_k,
        pool_embedding_model=args.pool_embedding_model,
        allow_same_model=args.allow_same_model,
    )
    manifest = SyntheticBenchmarkBuilder(config).build(args.docs, args.output, dataset_id=args.dataset_id)
    return {
        "output": args.output,
        "questions": manifest["stages"]["final"],
        "models": manifest["models"],
        "pool_retrievers": manifest["pool_retrievers"],
        "stages": manifest["stages"],
    }


def _parse_runs(items: list[str], flag: str) -> dict[str, str]:
    runs: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"{flag} expects TECHNIQUE=REPORT, got {item!r}")
        technique, _, report = item.partition("=")
        runs[technique] = report
    return runs


def _audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.emit_label_sample:
        if not args.synthetic_qa:
            raise SystemExit("--emit-label-sample requires --synthetic-qa")
        return write_label_sample(
            args.synthetic_qa, args.emit_label_sample, sample_size=args.label_sample, seed=args.seed
        )
    return run_audit(
        synthetic_runs=_parse_runs(args.synthetic_runs, "--synthetic-runs"),
        golden_runs=_parse_runs(args.golden_runs, "--golden-runs"),
        output_dir=args.output,
        synthetic_qa_path=args.synthetic_qa,
        label_review_path=args.label_review,
        metrics=tuple(args.metrics),
        seed=args.seed,
        samples=args.bootstrap_samples,
    )


def _guard_judge_role(args: argparse.Namespace) -> None:
    """Refuse to judge a generated dataset with the model that wrote its questions.

    A judge that also authored the questions is scoring answers to prompts it
    chose, against a ground truth it wrote — the two failures correlate, so
    faithfulness and correctness both come out flattering.
    """
    if not getattr(args, "judge", False):
        return
    from ragbench.datasets.synthetic import load_synthetic_manifest

    dataset = getattr(args, "dataset", None) or getattr(args, "qa", None)
    manifest = load_synthetic_manifest(dataset) if dataset else None
    if manifest is None:
        return
    models = {role: name for role, name in (manifest.get("models") or {}).items() if isinstance(name, str)}
    collisions = role_collisions({**models, "judge": args.judge_model})
    judge_collisions = [item for item in collisions if "judge" in item]
    if judge_collisions and not getattr(args, "allow_same_model", False):
        raise SystemExit(
            f"Judge model {args.judge_model!r} also generated this dataset ({', '.join(judge_collisions)}). "
            f"Set {ROLE_ENV_VARS['judge']} to a different model, or pass --allow-same-model to record the "
            "circularity and downgrade the trust verdict."
        )


def _compare(runs: list[str], dataset: str, output: str, top_k: int) -> None:
    rows = []
    output_path = Path(output)
    for run in runs:
        if "=" not in run:
            raise SystemExit(f"Invalid run '{run}', expected technique_id=artifact")
        technique_id, artifact = run.split("=", 1)
        pipeline = load_pipeline_for_artifact(technique_id, artifact)
        fingerprint = pipeline.load_artifact(artifact)[0]["corpus"]["fingerprint"].split(":", 1)[-1][:10]
        report_path = output_path.with_name(f"{pipeline.id}_{fingerprint}_eval.json")
        report = run_eval(pipeline, artifact, dataset, str(report_path), top_k=top_k)
        metadata = report["run_metadata"]
        rows.append(
            {
                "pipeline": pipeline.id,
                "artifact_fingerprint": metadata["artifact_fingerprint"],
                "config_fingerprint": metadata["pipeline_config_fingerprint"],
                "effective_components": report["effective_components"],
                "report": str(report_path),
                **report["metrics"],
            }
        )
    warnings = _compare_warnings(rows)
    payload = {"runs": rows, "warnings": warnings}
    write_json(output, payload)
    _print({**payload, "output": output})


def _compare_warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    pipelines = {str(row["pipeline"]) for row in rows}
    for pipeline_id in sorted(pipelines):
        matching = [row for row in rows if row["pipeline"] == pipeline_id]
        signatures = {
            (
                str(row["artifact_fingerprint"]),
                str(row["config_fingerprint"]),
                json.dumps(row["effective_components"], ensure_ascii=False, sort_keys=True),
            )
            for row in matching
        }
        if len(signatures) > 1:
            warnings.append(
                f"Pipeline '{pipeline_id}' có nhiều artifact/config/effective implementation; không nên gộp metric."
            )
    return warnings


def _bench(args: argparse.Namespace) -> None:
    from ragbench.benchmarks.runner import has_failed_runs, run_benchmarks, run_preflight

    if not args.suite and (not args.docs or not args.qa):
        raise SystemExit("Provide --docs and --qa, or use --suite.")
    judge_spec = {"type": "openai", "params": {"model": args.judge_model}} if args.judge else None
    if args.preflight:
        try:
            result = run_preflight(
                technique_ids=args.techniques,
                docs=args.docs,
                qa=args.qa,
                mode=args.mode,
                top_k=args.top_k,
                suite_path=args.suite,
                warmup_queries=args.warmup_queries,
                concurrency=args.concurrency,
                latency_sample_size=args.latency_sample_size,
                offline=args.offline,
                judge_spec=judge_spec,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        _print(result)
        if not result["ready"]:
            raise SystemExit(1)
        return
    if not args.output:
        raise SystemExit("--output is required unless --preflight is set.")
    try:
        result = run_benchmarks(
            technique_ids=args.techniques,
            docs=args.docs,
            qa=args.qa,
            output=args.output,
            mode=args.mode,
            top_k=args.top_k,
            profile=args.profile,
            resume=args.resume,
            seed=args.seed,
            suite_path=args.suite,
            judge_spec=judge_spec,
            warmup_queries=args.warmup_queries,
            latency_repetitions=args.latency_repetitions,
            max_estimated_cost_usd=args.max_estimated_cost_usd,
            concurrency=args.concurrency,
            latency_sample_size=args.latency_sample_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _print(result)
    if has_failed_runs(result):
        raise SystemExit(1)


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _print_or_exit(callback: Callable[[], dict[str, Any]]) -> None:
    try:
        _print(callback())
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
