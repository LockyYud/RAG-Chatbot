from __future__ import annotations

from evaluation.judge import parse_json_object
from evaluation.runner import _cost_summary
from raglab.core.schema import RAGAnswer
from raglab.datasets.synthetic import parse_json_array


def test_parse_json_array_extracts_embedded_array() -> None:
    rows = parse_json_array('prefix [{"question":"Q","ground_truth_answer":"A"}] suffix')
    assert rows == [{"question": "Q", "ground_truth_answer": "A"}]


def test_parse_json_object_extracts_embedded_object() -> None:
    payload = parse_json_object('```json\n{"faithfulness": 1}\n```')
    assert payload["faithfulness"] == 1


def test_cost_summary_does_not_double_count_judge_in_provider_ledger() -> None:
    prediction = RAGAnswer(
        "Q",
        "A",
        [],
        metadata={
            "cost_estimate": {"amount": 0.25},
            "evaluation_cost_estimate": {"amount": 0.1},
        },
    )
    summary = _cost_summary([prediction])
    assert summary["total_estimated_cost"] == 0.25
    assert summary["evaluation_total_estimated_cost"] == 0.1
