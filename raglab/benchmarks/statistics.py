"""Small dependency-free statistical helpers for paired RAG comparisons."""

from __future__ import annotations

import random
from typing import Any


def paired_bootstrap_delta(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: str,
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> dict[str, float | int | None]:
    """Estimate a paired query-level delta and percentile 95% interval.

    Rows are joined by question id. Missing or non-numeric measurements are
    omitted so judge outages cannot silently become zero-score observations.
    """
    base = {str(row["question_id"]): row.get(metric) for row in baseline}
    pairs: list[tuple[float, float]] = []
    for row in candidate:
        key = str(row.get("question_id"))
        baseline_value, candidate_value = base.get(key), row.get(metric)
        valid_pair = (
            baseline_value is not None
            and candidate_value is not None
            and _number(baseline_value)
            and _number(candidate_value)
        )
        if valid_pair:
            assert baseline_value is not None and candidate_value is not None
            pairs.append((float(baseline_value), float(candidate_value)))
    if not pairs:
        return {"paired_queries": 0, "delta": None, "ci95_low": None, "ci95_high": None}
    deltas = [right - left for left, right in pairs]
    point = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    resampled = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(samples)]
    resampled.sort()
    low = resampled[int((samples - 1) * 0.025)]
    high = resampled[int((samples - 1) * 0.975)]
    return {
        "paired_queries": len(deltas),
        "delta": round(point, 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
    }


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
