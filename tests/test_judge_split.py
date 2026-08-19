from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from ragbench.core.schema import Citation, EvalItem, RAGAnswer, RetrievalResult
from ragbench.evaluation.judge import LLMJudge


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
