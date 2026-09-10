from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ragbench.benchmarks.statistics import paired_bootstrap_delta
from ragbench.core.schema import Citation, EvalItem, RAGAnswer, RetrievalResult
from ragbench.evaluation.judge import LLMJudge
from ragbench.evaluation.metrics import evaluate_prediction_rows, evaluate_predictions


def _fake_response(text: str, prompt_tokens: int = 5, completion_tokens: int = 5) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    )


def test_faithfulness_call_never_receives_ground_truth_or_expected_doc_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: the single-prompt judge showed the faithfulness/citation
    judgment the ground truth answer and expected_doc_ids — evaluator
    coupling. The faithfulness sub-call's messages must not contain either,
    even serialized as JSON, no matter how the prompt is phrased later."""
    calls: list[list[dict[str, str]]] = []

    def fake_completion(**kwargs: Any) -> Any:
        calls.append(kwargs["messages"])
        payload = {"answer_correctness": 1, "abstention_correctness": 1, "faithfulness": 1, "citation_support": 1}
        return _fake_response(json.dumps(payload))

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(completion=fake_completion)
    )
    judge = LLMJudge()
    item = EvalItem(
        question_id="q1",
        question="What is the capital of Vietnam?",
        ground_truth_answer="SECRET_GROUND_TRUTH_HANOI",
        expected_doc_ids=["SECRET_DOC_42"],
    )
    prediction = RAGAnswer(
        query=item.question,
        answer="Hanoi",
        contexts=[
            RetrievalResult(node_id="n1", chunk_id="c1", doc_id="d1", text="Hanoi is the capital.", score=1.0, rank=1)
        ],
        citations=[Citation(citation_id="C1", doc_id="d1", chunk_id="c1")],
    )

    judge.judge(item, prediction)

    assert len(calls) == 2
    correctness_call, faithfulness_call = calls
    correctness_text = json.dumps(correctness_call)
    faithfulness_text = json.dumps(faithfulness_call)
    assert "SECRET_GROUND_TRUTH_HANOI" in correctness_text
    assert "SECRET_DOC_42" not in correctness_text  # correctness never saw expected_doc_ids either
    assert "SECRET_GROUND_TRUTH_HANOI" not in faithfulness_text
    assert "SECRET_DOC_42" not in faithfulness_text


def test_judge_merges_both_sub_calls_and_sums_their_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        json.dumps({"answer_correctness": 0.8, "abstention_correctness": 1.0}),
        json.dumps({"faithfulness": 0.6, "citation_support": 0.4}),
    ]

    def fake_completion(**kwargs: Any) -> Any:
        return _fake_response(responses.pop(0))

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(completion=fake_completion)
    )
    judge = LLMJudge()
    item = EvalItem(question_id="q1", question="Q", ground_truth_answer="A")
    prediction = RAGAnswer(query="Q", answer="A", contexts=[])

    result = judge.judge(item, prediction)

    assert result.answer_correctness == 0.8
    assert result.abstention_correctness == 1.0
    assert result.faithfulness == 0.6
    assert result.citation_support == 0.4
    assert result.status == "ok"
    assert result.correctness_status == "ok"
    assert result.faithfulness_status == "ok"
    # Two calls' usage must be summed, not just the last one's.
    assert result.usage["prompt_tokens"] == 10
    assert result.usage["completion_tokens"] == 10


def test_a_failure_in_one_sub_judge_does_not_hide_which_one_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        "not json at all",  # correctness call fails to parse
        json.dumps({"faithfulness": 0.9, "citation_support": 0.9}),  # faithfulness succeeds
    ]

    def fake_completion(**kwargs: Any) -> Any:
        return _fake_response(responses.pop(0))

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(completion=fake_completion)
    )
    judge = LLMJudge()
    item = EvalItem(question_id="q1", question="Q", ground_truth_answer="A")
    prediction = RAGAnswer(query="Q", answer="A", contexts=[])

    result = judge.judge(item, prediction)

    assert result.correctness_status == "parse_failure"
    assert result.faithfulness_status == "ok"
    assert result.status == "parse_failure"  # blanket status still reflects the failure


def _item(question_id: str) -> EvalItem:
    return EvalItem(question_id=question_id, question="Q", ground_truth_answer="A")


def _prediction(judge_payload: dict[str, Any]) -> RAGAnswer:
    return RAGAnswer(query="Q", answer="A", contexts=[], metadata={"judge": judge_payload})


def test_a_faithfulness_outage_does_not_discard_a_valid_correctness_score() -> None:
    """Regression: aggregation gated all four judge fields on the blanket
    ``judge_status``, which is "not ok" when *either* sub-judge failed. A
    malformed faithfulness response therefore threw away a perfectly good
    answer_correctness from the other, independent provider call."""
    judge_payload = {
        "status": "parse_failure",
        "correctness_status": "ok",
        "faithfulness_status": "parse_failure",
        "answer_correctness": 0.92,
        "abstention_correctness": 1.0,
    }

    metrics = evaluate_predictions([_item("q1")], [_prediction(judge_payload)], k=5)

    assert metrics["answer_correctness"] == 0.92
    assert metrics["correctness_judge_queries_evaluated"] == 1
    assert metrics["correctness_judge_failure_rate"] == 0.0
    assert "faithfulness" not in metrics
    assert metrics["faithfulness_judge_queries_evaluated"] == 0
    assert metrics["faithfulness_judge_failure_rate"] == 1.0
    # The blanket figures still report "at least one half failed".
    assert metrics["judge_failure_rate"] == 1.0


def test_a_correctness_outage_does_not_discard_a_valid_faithfulness_score() -> None:
    judge_payload = {
        "status": "schema_failure",
        "correctness_status": "schema_failure",
        "faithfulness_status": "ok",
        "faithfulness": 0.75,
        "citation_support": 0.5,
    }

    metrics = evaluate_predictions([_item("q1")], [_prediction(judge_payload)], k=5)

    assert metrics["faithfulness"] == 0.75
    assert metrics["citation_support"] == 0.5
    assert "answer_correctness" not in metrics
    assert metrics["correctness_judge_failure_rate"] == 1.0
    assert metrics["faithfulness_judge_failure_rate"] == 0.0


def test_judge_payload_without_split_statuses_behaves_exactly_as_before() -> None:
    """Back-compat: reports and fixtures written before the judge split carry
    only ``status``, which must still gate both halves together."""
    both_ok = {"status": "ok", "answer_correctness": 0.8, "faithfulness": 0.6}
    metrics = evaluate_predictions([_item("q1")], [_prediction(both_ok)], k=5)
    assert metrics["answer_correctness"] == 0.8
    assert metrics["faithfulness"] == 0.6

    both_failed = {"status": "parse_failure", "answer_correctness": 0.8, "faithfulness": 0.6}
    metrics = evaluate_predictions([_item("q1")], [_prediction(both_failed)], k=5)
    assert "answer_correctness" not in metrics
    assert "faithfulness" not in metrics


def test_paired_bootstrap_drops_only_the_half_whose_judge_failed() -> None:
    """The per-query rows feed paired_bootstrap_delta, which joins on
    question_id and skips pairs with a missing measurement. A faithfulness
    outage on one query must shrink only the faithfulness pairing — the
    correctness comparison keeps every query it legitimately measured."""
    items = [_item("q1"), _item("q2")]
    ok = {"status": "ok", "correctness_status": "ok", "faithfulness_status": "ok"}
    baseline_rows = evaluate_prediction_rows(
        items,
        [
            _prediction({**ok, "answer_correctness": 0.5, "faithfulness": 0.5}),
            _prediction({**ok, "answer_correctness": 0.5, "faithfulness": 0.5}),
        ],
        k=5,
    )
    candidate_rows = evaluate_prediction_rows(
        items,
        [
            _prediction({**ok, "answer_correctness": 0.9, "faithfulness": 0.9}),
            _prediction(
                {
                    "status": "parse_failure",
                    "correctness_status": "ok",
                    "faithfulness_status": "parse_failure",
                    "answer_correctness": 0.9,
                }
            ),
        ],
        k=5,
    )

    correctness = paired_bootstrap_delta(baseline_rows, candidate_rows, "answer_correctness", samples=50)
    faithfulness = paired_bootstrap_delta(baseline_rows, candidate_rows, "faithfulness", samples=50)

    assert correctness["paired_queries"] == 2
    assert faithfulness["paired_queries"] == 1
