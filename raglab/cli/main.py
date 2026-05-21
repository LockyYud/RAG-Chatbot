from __future__ import annotations

import argparse
import json
from pathlib import Path

from raglab.core.io import write_json
from raglab.core.pipeline import ingest, query
from raglab.core.techniques import get_technique, list_techniques
from evaluation.runner import run_eval


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

    compare_parser = subparsers.add_parser("compare", help="Evaluate multiple config/artifact pairs")
    compare_parser.add_argument("--runs", nargs="+", required=True, help="Items in config=artifact format")
    compare_parser.add_argument("--dataset", required=True)
    compare_parser.add_argument("--output", required=True)
    compare_parser.add_argument("--top-k", type=int, default=5)

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
        _print(run_eval(args.config, args.artifact, args.dataset, args.output, args.top_k)["metrics"])
    elif args.command == "compare":
        _compare(args.runs, args.dataset, args.output, args.top_k)
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


def _print(data: dict) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
