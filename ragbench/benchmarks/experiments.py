"""Repeatable experiment matrices built on the normal benchmark contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.benchmarks.runner import run_benchmarks
from ragbench.core.io import write_json
from ragbench.core.measure import canonical_fingerprint


def run_experiment_matrix(
    *,
    technique_ids: list[str],
    docs: str,
    qa: str,
    output: str,
    trials: int = 1,
    seed: int = 42,
    mode: str = "full_rag",
    profile: str = "auto",
    top_k: int = 5,
    judge_spec: dict[str, Any] | None = None,
    warmup_queries: int = 0,
    latency_repetitions: int = 1,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least 1")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    matrix_id = canonical_fingerprint(
        {
            "techniques": technique_ids,
            "docs": docs,
            "qa": qa,
            "trials": trials,
            "seed": seed,
            "mode": mode,
            "profile": profile,
            "top_k": top_k,
            "judge_spec": judge_spec,
            "warmup_queries": warmup_queries,
            "latency_repetitions": latency_repetitions,
        }
    )
    trial_runs = []
    for trial in range(trials):
        trial_seed = seed + trial
        result = run_benchmarks(
            technique_ids=technique_ids,
            docs=docs,
            qa=qa,
            output=str(root / f"trial-{trial + 1:03d}"),
            mode=mode,
            profile=profile,
            top_k=top_k,
            seed=trial_seed,
            judge_spec=judge_spec,
            warmup_queries=warmup_queries,
            latency_repetitions=latency_repetitions,
        )
        trial_runs.append({"trial": trial + 1, "seed": trial_seed, "result": result})
    payload = {
        "matrix_id": matrix_id,
        "trials": trial_runs,
        "aggregate": _aggregate_trials(trial_runs),
    }
    write_json(root / "matrix.json", payload)
    return payload


def _aggregate_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-technique metrics across independent trial summaries."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for trial in trials:
        for row in trial["result"].get("runs", []):
            if row.get("status") != "ok":
                continue
            metrics = grouped.setdefault(str(row["technique"]), {})
            for name, value in row.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    metrics.setdefault(name, []).append(float(value))
    return {
        technique: {
            metric: {
                "trials": len(values),
                "mean": round(sum(values) / len(values), 6),
                "std": round(_sample_std(values), 6),
            }
            for metric, values in metrics.items()
        }
        for technique, metrics in grouped.items()
    }


def _sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
