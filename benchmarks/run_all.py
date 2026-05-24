from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.runner import run_eval
from raglab.core.pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected techniques on one dataset.")
    parser.add_argument("--techniques", nargs="+", required=True, help="Technique ids under techniques/")
    parser.add_argument("--docs", required=True, help="Document directory")
    parser.add_argument("--qa", required=True, help="QA JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Skip runs with an existing report")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for technique in args.techniques:
        config = Path("techniques") / technique / "config.yaml"
        artifact = output_dir / "artifacts" / technique
        report = output_dir / f"{technique}_eval.json"
        try:
            if args.resume and report.exists():
                evaluation = json.loads(report.read_text(encoding="utf-8"))
                manifest = evaluation.get("run_metadata", {})
            else:
                manifest = ingest(str(config), args.docs, str(artifact))
                evaluation = run_eval(
                    str(config),
                    str(artifact),
                    args.qa,
                    str(report),
                    top_k=args.top_k,
                    mode=args.mode,
                )
            rows.append(_row(technique, config, artifact, report, manifest, evaluation, status="ok"))
        except Exception as exc:  # noqa: BLE001 - benchmark should continue and report failures.
            rows.append(
                {
                    "technique": technique,
                    "config": str(config),
                    "artifact": str(artifact),
                    "report": str(report),
                    "status": "failed",
                    "error": str(exc),
                }
            )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "summary.csv", rows)
    _write_markdown(output_dir / "summary.md", rows)
    print(json.dumps({"runs": rows, "summary": str(summary_path)}, ensure_ascii=False, indent=2))

def _row(
    technique: str,
    config: Path,
    artifact: Path,
    report: Path,
    manifest: dict,
    evaluation: dict,
    status: str,
) -> dict:
    metrics = evaluation.get("metrics", {})
    return {
        "technique": technique,
        "config": str(config),
        "artifact": str(artifact),
        "report": str(report),
        "status": status,
        "node_count": manifest.get("node_count", ""),
        **metrics,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict]) -> None:
    columns = [
        "technique",
        "status",
        "recall_at_5",
        "mrr",
        "citation_accuracy",
        "answer_correctness",
        "faithfulness",
        "latency_ms_avg",
        "estimated_cost_avg",
    ]
    lines = ["# Benchmark Summary", "", "|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        lines.append("|" + "|".join(str(row.get(column, "")) for column in columns) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
