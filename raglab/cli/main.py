from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluation.runner import run_eval
from raglab.core.base import (
    BasePipeline,
    get_pipeline_metadata,
    list_pipelines,
    load_pipeline,
    load_pipeline_for_artifact,
)
from raglab.core.doctor import diagnose_technique
from raglab.core.io import write_json
from raglab.datasets.adapters import DATASET_ADAPTERS, prepare_dataset
from raglab.datasets.golden import validate_golden_dataset
from raglab.datasets.schema import sample_processed_dataset, validate_processed_dataset
from raglab.datasets.synthetic import SyntheticQAGenerator
from raglab.indexing.artifacts import inspect_artifact


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
            f"No pipeline.py found for '{technique_id}'. Use `raglab techniques list` to see bundled technique ids."
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
    parser = argparse.ArgumentParser(prog="raglab", description="RAG Pipeline Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse, chunk, enrich, and save index artifacts")
    _add_technique_args(ingest_parser)
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--output", required=True)

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
    eval_parser.add_argument("--judge-model", default="gpt-4.1-mini")

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
    bench_parser.add_argument("--judge-model", default="gpt-4.1-mini")
    bench_parser.add_argument(
        "--warmup-queries",
        type=int,
        default=None,
        help="Default 0 unless a suite locks a value; explicit values conflicting with a locked suite are rejected",
    )
    bench_parser.add_argument("--latency-repetitions", type=int, default=1)
    bench_parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check suite/dataset/provider/dependency readiness only; do not ingest or query anything",
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
    experiment_parser.add_argument("--judge-model", default="gpt-4.1-mini")
    experiment_parser.add_argument("--warmup-queries", type=int, default=0)
    experiment_parser.add_argument("--latency-repetitions", type=int, default=1)

    dataset_parser = subparsers.add_parser("dataset", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    generate_parser = dataset_subparsers.add_parser("generate", help="Generate synthetic QA from documents")
    generate_parser.add_argument("--docs", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--limit", type=int, default=50)
    generate_parser.add_argument("--model", default="gpt-4.1-mini")
    generate_parser.add_argument("--questions-per-chunk", type=int, default=2)
    prepare_parser = dataset_subparsers.add_parser("prepare", help="Prepare a fixed research dataset for evaluation")
    prepare_parser.add_argument("name", choices=sorted(DATASET_ADAPTERS))
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--split")
    prepare_parser.add_argument("--limit", type=int)
    prepare_parser.add_argument("--seed", type=int, default=42)
    prepare_parser.add_argument("--overwrite", action="store_true")
    validate_parser = dataset_subparsers.add_parser("validate", help="Validate a processed evaluation dataset")
    validate_parser.add_argument("path")
    validate_golden_parser = dataset_subparsers.add_parser(
        "validate-golden", help="Validate a human-curated end-to-end RAG golden set"
    )
    validate_golden_parser.add_argument("path")
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

    args = parser.parse_args()
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
        judge_spec = {"type": "openai", "params": {"model": args.judge_model}} if args.judge else None
        report = run_eval(
            pipeline,
            args.artifact,
            args.dataset,
            args.output,
            top_k=args.top_k,
            mode=args.mode,
            judge_spec=judge_spec,
            profile=args.profile,
        )
        _print(report["metrics"])
    elif args.command == "compare":
        _compare(args.runs, args.dataset, args.output, args.top_k)
    elif args.command == "bench":
        _bench(args)
    elif args.command == "experiment":
        from raglab.benchmarks.experiments import run_experiment_matrix

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
            rows = SyntheticQAGenerator(model=args.model, questions_per_chunk=args.questions_per_chunk).generate(
                args.docs,
                args.output,
                args.limit,
            )
            _print({"output": args.output, "questions": len(rows)})
        elif args.dataset_command == "prepare":
            _print_or_exit(
                lambda: prepare_dataset(
                    args.name, args.output, split=args.split, limit=args.limit, seed=args.seed, overwrite=args.overwrite
                )
            )
        elif args.dataset_command == "validate":
            _print_or_exit(lambda: validate_processed_dataset(args.path))
        elif args.dataset_command == "validate-golden":
            _print_or_exit(lambda: validate_golden_dataset(args.path))
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
        _print(diagnose_technique(_resolve_pipeline(args), mode=args.mode))


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
    from raglab.benchmarks.runner import has_failed_runs, run_benchmarks, run_preflight

    if not args.suite and (not args.docs or not args.qa):
        raise SystemExit("Provide --docs and --qa, or use --suite.")
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
            judge_spec={"type": "openai", "params": {"model": args.judge_model}} if args.judge else None,
            warmup_queries=args.warmup_queries,
            latency_repetitions=args.latency_repetitions,
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
