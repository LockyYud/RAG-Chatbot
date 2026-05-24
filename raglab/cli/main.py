from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluation.runner import run_eval
from raglab.core.io import write_json
from raglab.core.pipeline import ingest, query
from raglab.core.techniques import get_technique, list_techniques
from raglab.datasets.synthetic import SyntheticQAGenerator
from raglab.indexing.artifacts import inspect_artifact


def main() -> None:
    parser = argparse.ArgumentParser(prog="raglab", description="RAG Pipeline Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Parse, chunk, enrich, and save index artifacts")
    ingest_parser.add_argument("--config", required=True)
    ingest_parser.add_argument("--input", required=True)
    ingest_parser.add_argument("--output", required=True)

    query_parser = subparsers.add_parser("query", help="Run a single query against saved artifacts")
    query_parser.add_argument("--config", required=True)
    query_parser.add_argument("--artifact", required=True)
    query_parser.add_argument("--query", required=True)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a pipeline on a JSONL QA dataset")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--artifact", required=True)
    eval_parser.add_argument("--dataset", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    eval_parser.add_argument("--judge", action="store_true", help="Enable OpenAI-compatible LLM-as-judge metrics")
    eval_parser.add_argument("--judge-model", default="gpt-4.1-mini")

    compare_parser = subparsers.add_parser("compare", help="Evaluate multiple config/artifact pairs")
    compare_parser.add_argument("--runs", nargs="+", required=True, help="Items in config=artifact format")
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--top-k", type=int, default=5)

    bench_parser = subparsers.add_parser("bench", help="Run benchmarks across technique ids")
    bench_parser.add_argument("--techniques", nargs="+", required=True)
    bench_parser.add_argument("--docs", required=True)
    bench_parser.add_argument("--qa", required=True)
    bench_parser.add_argument("--output", required=True)
    bench_parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    bench_parser.add_argument("--top-k", type=int, default=5)
    bench_parser.add_argument("--resume", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Dataset utilities")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command", required=True)
    generate_parser = dataset_subparsers.add_parser("generate", help="Generate synthetic QA from documents")
    generate_parser.add_argument("--docs", required=True)
    generate_parser.add_argument("--output", required=True)
    generate_parser.add_argument("--limit", type=int, default=50)
    generate_parser.add_argument("--model", default="gpt-4.1-mini")
    generate_parser.add_argument("--questions-per-chunk", type=int, default=2)

    artifacts_parser = subparsers.add_parser("artifacts", help="Artifact utilities")
    artifacts_subparsers = artifacts_parser.add_subparsers(dest="artifacts_command", required=True)
    inspect_parser = artifacts_subparsers.add_parser("inspect", help="Inspect saved artifact metadata")
    inspect_parser.add_argument("--artifact", required=True)

    techniques_parser = subparsers.add_parser("techniques", help="List or inspect paper-driven techniques")
    techniques_subparsers = techniques_parser.add_subparsers(dest="techniques_command", required=True)
    techniques_subparsers.add_parser("list", help="List registered technique metadata")
    show_parser = techniques_subparsers.add_parser("show", help="Show one technique metadata")
    show_parser.add_argument("technique_id")

    args = parser.parse_args()
    if args.command == "ingest":
        _print(ingest(args.config, args.input, args.output))
    elif args.command == "query":
        answer = query(args.config, args.artifact, args.query)
        _print(answer.to_dict())
    elif args.command == "eval":
        judge_spec = {"type": "openai", "params": {"model": args.judge_model}} if args.judge else None
        report = run_eval(
            args.config,
            args.artifact,
            args.dataset,
            args.output,
            args.top_k,
            args.mode,
            judge_spec,
        )
        _print(report["metrics"])
    elif args.command == "compare":
        _compare(args.runs, args.dataset, args.output, args.top_k)
    elif args.command == "bench":
        _bench(args)
    elif args.command == "dataset":
        if args.dataset_command == "generate":
            rows = SyntheticQAGenerator(model=args.model, questions_per_chunk=args.questions_per_chunk).generate(
                args.docs,
                args.output,
                args.limit,
            )
            _print({"output": args.output, "questions": len(rows)})
    elif args.command == "artifacts":
        if args.artifacts_command == "inspect":
            _print(inspect_artifact(args.artifact))
    elif args.command == "techniques":
        if args.techniques_command == "list":
            _print({"techniques": list_techniques()})
        elif args.techniques_command == "show":
            _print(get_technique(args.technique_id))


def _compare(runs: list[str], dataset: str, output: str, top_k: int) -> None:
    rows = []
    output_path = Path(output)
    for run in runs:
        if "=" not in run:
            raise SystemExit(f"Invalid run '{run}', expected config=artifact")
        config, artifact = run.split("=", 1)
        report_path = output_path.with_name(f"{Path(config).stem}_eval.json")
        report = run_eval(config, artifact, dataset, str(report_path), top_k)
        row = {"pipeline": Path(config).stem, **report["metrics"]}
        rows.append(row)
    write_json(output, {"runs": rows})
    _print({"runs": rows, "output": output})


def _bench(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        "benchmarks/run_all.py",
        "--techniques",
        *args.techniques,
        "--docs",
        args.docs,
        "--qa",
        args.qa,
        "--output",
        args.output,
        "--mode",
        args.mode,
        "--top-k",
        str(args.top_k),
    ]
    if args.resume:
        command.append("--resume")
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout, end="")


def _print(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
