from __future__ import annotations

import pytest

from evaluation.metrics import evaluate_predictions
from raglab.core.config import ConfigError, load_config, validate_config
from raglab.core.registry import Registry
from raglab.core.schema import EvalItem, RAGAnswer, RetrievalResult


def test_load_config_supports_json_style_yaml() -> None:
    config = load_config("techniques/naive_rag/config.yaml")
    validate_config(config, for_ingest=True)
    assert config["name"] == "naive_rag"


def test_validate_config_rejects_bad_store() -> None:
    with pytest.raises(ConfigError):
        validate_config({"indexing": {"store": {"type": "unknown"}}})


def test_registry_unknown_strategy_error_lists_known() -> None:
    registry = Registry()
    registry.register("known", lambda: object())
    with pytest.raises(KeyError, match="Known: known"):
        registry.create({"type": "missing"})


def test_metrics_include_judge_scores_when_present() -> None:
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["doc"])
    result = RetrievalResult("n1", "c1", "doc", "text", 1.0, 1)
    prediction = RAGAnswer(
        query="Q",
        answer="A",
        contexts=[result],
        citations=["doc:c1"],
        metadata={"judge": {"answer_correctness": 0.8, "faithfulness": 0.7, "citation_support": 0.9}},
    )
    metrics = evaluate_predictions([item], [prediction])
    assert metrics["recall_at_5"] == 1.0
    assert metrics["answer_correctness"] == 0.8
