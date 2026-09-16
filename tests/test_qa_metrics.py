from __future__ import annotations

import pytest

from ragbench.core.schema import EvalItem, RAGAnswer
from ragbench.evaluation.metrics import evaluate_prediction_rows, evaluate_predictions
from ragbench.evaluation.metrics.qa import exact_match, token_f1


def test_exact_match_ignores_case_punctuation_and_whitespace() -> None:
    assert exact_match("Nghỉ  hằng năm.", "nghỉ hằng năm") == 1.0
    assert exact_match("Nghỉ hằng năm", "Nghỉ phép") == 0.0


def test_token_f1_partial_overlap() -> None:
    assert token_f1("người lao động", "người lao động được nghỉ") == pytest.approx(0.75)
    assert token_f1("", "something") == 0.0
    assert token_f1("same", "same") == 1.0


def test_evaluate_predictions_scores_extractive_qa_and_skips_unanswerable() -> None:
    items = [
        EvalItem(question_id="q1", question="Q1?", ground_truth_answer="Nghỉ hằng năm"),
        EvalItem(question_id="q2", question="Q2?", ground_truth_answer=None, metadata={"is_answerable": False}),
    ]
    predictions = [
        RAGAnswer(query="Q1?", answer="Nghỉ hằng năm.", contexts=[]),
        RAGAnswer(query="Q2?", answer="", contexts=[], abstained=True),
    ]
    rows = evaluate_prediction_rows(items, predictions)
    assert rows[0]["exact_match"] == 1.0
    assert "exact_match" not in rows[1]

    metrics = evaluate_predictions(items, predictions)
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["qa_queries_evaluated"] == 1
