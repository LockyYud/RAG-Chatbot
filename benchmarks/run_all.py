from __future__ import annotations

import argparse
import json

from ragbench.benchmarks.runner import has_failed_runs, run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected techniques on one dataset.")
    parser.add_argument("--techniques", nargs="+", required=True)
    parser.add_argument("--docs", required=True)
    parser.add_argument("--qa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["full_rag", "retrieval_only"], default="full_rag")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_benchmarks(
        technique_ids=args.techniques,
        docs=args.docs,
        qa=args.qa,
        output=args.output,
        mode=args.mode,
        top_k=args.top_k,
        profile=args.profile,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if has_failed_runs(result):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
