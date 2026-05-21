from __future__ import annotations

import argparse
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
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for technique in args.techniques:
        config = Path("techniques") / technique / "config.yaml"
        artifact = output_dir / "artifacts" / technique
        report = output_dir / f"{technique}_eval.json"
        manifest = ingest(str(config), args.docs, str(artifact))
        evaluation = run_eval(str(config), str(artifact), args.qa, str(report))
        rows.append(
            {
                "technique": technique,
                "config": str(config),
                "artifact": str(artifact),
                "report": str(report),
                "node_count": manifest["node_count"],
                **evaluation["metrics"],
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": rows, "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
