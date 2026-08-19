from __future__ import annotations

from typing import Any

from ragbench.benchmarks.suites import _improvement_supported


def _interval(low: float, high: float, *, paired_queries: int = 50) -> dict[str, Any]:
    return {"paired_queries": paired_queries, "delta": (low + high) / 2, "ci95_low": low, "ci95_high": high}


def _comparison(candidate: str, baseline: str, intervals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"baseline": baseline, "candidate": candidate, "deltas": {}, "paired_ci95": intervals}


def _suite(**overrides: Any) -> dict[str, Any]:
    base = {
        "tier": "claim_eligible",
        "reference_baseline": "naive_rag",
        "primary_metrics": ["ndcg_at_10", "recall_at_10", "mrr"],
        "primary_metric": "ndcg_at_10",
        "minimum_effect": 0.02,
        "non_inferiority": {"recall_at_10": -0.01, "mrr": -0.01},
    }
    base.update(overrides)
    return base


def test_regression_one_metric_improving_no_longer_masks_others_regressing() -> None:
    """This is the exact bug being fixed: a candidate that raises recall_at_10
    while cratering ndcg_at_10 and mrr used to be reported
    improvement_supported=True just because *a* primary metric's CI cleared
    zero. It must now be rejected — recall_at_10 is not the primary_metric,
    and even if it were the decision metric, ndcg_at_10/mrr regressing
    outside their non_inferiority margin must veto the claim."""
    suite = _suite()
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(-0.08, -0.03),  # regressed hard
                "recall_at_10": _interval(0.05, 0.15),  # improved
                "mrr": _interval(-0.10, -0.04),  # regressed hard
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is False


def test_primary_metric_improvement_supported_when_effect_clears_minimum_and_guards_hold() -> None:
    suite = _suite()
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(0.03, 0.09),  # clears minimum_effect=0.02
                "recall_at_10": _interval(-0.005, 0.02),  # within -0.01 non-inferiority margin
                "mrr": _interval(0.0, 0.03),  # improved, well within margin
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is True
    assert verdict["supported_metrics"] == ["candidate_x:ndcg_at_10"]


def test_primary_metric_effect_below_minimum_effect_is_not_supported() -> None:
    """CI clears zero but not the minimum_effect margin — statistically
    nonzero is not the same as a real effect size."""
    suite = _suite(minimum_effect=0.05)
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(0.005, 0.02),  # clears 0 but not 0.05
                "recall_at_10": _interval(0.0, 0.01),
                "mrr": _interval(0.0, 0.01),
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is False


def test_non_inferiority_guard_vetoes_an_otherwise_qualifying_primary_metric() -> None:
    suite = _suite()
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(0.05, 0.10),  # clears minimum_effect comfortably
                "recall_at_10": _interval(-0.05, -0.02),  # regressed beyond -0.01 margin
                "mrr": _interval(0.0, 0.02),
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is False
    assert any("recall_at_10" in reason for reason in verdict["reasons"])


def test_pareto_mode_supported_when_one_improves_and_none_regress() -> None:
    suite = _suite(pareto_improvement=True)
    del suite["primary_metric"]
    del suite["minimum_effect"]
    del suite["non_inferiority"]
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(0.01, 0.05),  # improved
                "recall_at_10": _interval(-0.01, 0.01),  # inconclusive, not regressed
                "mrr": _interval(0.0, 0.02),  # improved
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is True


def test_pareto_mode_rejects_when_any_primary_metric_regresses() -> None:
    suite = _suite(pareto_improvement=True)
    del suite["primary_metric"]
    del suite["minimum_effect"]
    del suite["non_inferiority"]
    comparisons = [
        _comparison(
            "candidate_x",
            "naive_rag",
            {
                "ndcg_at_10": _interval(0.03, 0.08),  # improved
                "recall_at_10": _interval(-0.08, -0.03),  # regressed
                "mrr": _interval(0.0, 0.02),
            },
        )
    ]
    verdict = _improvement_supported(suite, comparisons)
    assert verdict["supported"] is False


def test_no_candidate_against_reference_baseline_is_not_supported() -> None:
    suite = _suite()
    verdict = _improvement_supported(suite, [_comparison("candidate_x", "some_other_baseline", {})])
    assert verdict["supported"] is False
    assert "no candidate comparison against reference baseline" in verdict["reasons"]


def test_non_claim_eligible_suite_is_never_supported() -> None:
    suite = _suite(tier="exploratory")
    verdict = _improvement_supported(suite, [_comparison("candidate_x", "naive_rag", {})])
    assert verdict["supported"] is False
