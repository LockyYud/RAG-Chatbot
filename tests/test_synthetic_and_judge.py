from __future__ import annotations

from ragbench.core.schema import RAGAnswer
from ragbench.datasets.synthetic import parse_json_array
from ragbench.evaluation.judge import _judge_status, parse_json_object
from ragbench.evaluation.runner import _cost_summary


def test_parse_json_array_extracts_embedded_array() -> None:
    rows = parse_json_array('prefix [{"question":"Q","ground_truth_answer":"A"}] suffix')
    assert rows == [{"question": "Q", "ground_truth_answer": "A"}]


def test_parse_json_object_extracts_embedded_object() -> None:
    payload = parse_json_object('```json\n{"faithfulness": 1}\n```')
    assert payload["faithfulness"] == 1


def test_judge_status_is_ok_only_when_every_required_score_field_is_numeric() -> None:
    complete = {"answer_correctness": 1, "abstention_correctness": 1}
    assert _judge_status(complete, ("answer_correctness", "abstention_correctness")) == ("ok", [])


def test_judge_status_flags_schema_failure_instead_of_defaulting_to_zero() -> None:
    """Regression: valid JSON missing a required field used to fall through to
    _score(None) == 0.0 while still being reported as status="ok" — silently
    dragging the mean judge metric toward 0 instead of surfacing a failure.
    Parseable-but-incomplete JSON must be classified as schema_failure, not ok."""
    incomplete = {"faithfulness": 0.5}  # missing citation_support
    status, notes = _judge_status(incomplete, ("faithfulness", "citation_support"))
    assert status == "schema_failure"
    assert "citation_support" in notes[0]


def test_judge_status_flags_schema_failure_for_non_numeric_field() -> None:
    payload = {"faithfulness": "not a number", "citation_support": 0}
    status, notes = _judge_status(payload, ("faithfulness", "citation_support"))
    assert status == "schema_failure"
    assert "faithfulness" in notes[0]


def test_judge_status_is_parse_failure_when_json_never_parsed() -> None:
    payload = parse_json_object("not json at all")
    status, _notes = _judge_status(payload, ("answer_correctness", "abstention_correctness"))
    assert status == "parse_failure"


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
