from __future__ import annotations

import pytest

from evaluation.metrics import evaluate_prediction_rows, evaluate_predictions
from raglab.benchmarks.runner import _query_metric_name
from raglab.benchmarks.statistics import paired_bootstrap_delta
from raglab.benchmarks.suites import claim_eligibility, load_suite, resolve_suite
from raglab.core.base import list_pipelines, load_pipeline
from raglab.core.config import load_config
from raglab.core.schema import EvalItem, RAGAnswer, RetrievalResult


def test_list_pipelines_finds_every_technique() -> None:
    ids = {item["id"] for item in list_pipelines()}
    assert {"naive_rag", "parent_child", "hyde_2022", "rag_fusion_2024", "self_rag_2023"} <= ids


def test_load_pipeline_returns_concrete_instance() -> None:
    pipeline = load_pipeline("naive_rag")
    assert pipeline is not None
    assert pipeline.id == "naive_rag"


def test_load_pipeline_forwards_kwargs() -> None:
    pipeline = load_pipeline("naive_rag", params={"top_k": 99, "chunk_size": 42})
    assert pipeline is not None
    assert pipeline.top_k == 99  # type: ignore[attr-defined]
    assert pipeline.chunk_size == 42  # type: ignore[attr-defined]


def test_technique_yaml_metadata_loads() -> None:
    metadata = load_config("techniques/naive_rag/technique.yaml")
    assert metadata.get("id") == "naive_rag"


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


def test_retrieval_only_metrics_omit_citation_accuracy() -> None:
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["doc"], expected_citations=["doc"])
    result = RetrievalResult("n1", "c1", "doc", "text", 1.0, 1)
    prediction = RAGAnswer(query="Q", answer="", contexts=[result], citations=[])

    metrics = evaluate_predictions([item], [prediction], include_citation_accuracy=False)

    assert metrics["recall_at_5"] == 1.0
    assert "citation_f1" not in metrics


def test_metrics_report_ranking_operational_and_query_level_values() -> None:
    item = EvalItem(
        question_id="q1",
        question="Q",
        expected_doc_ids=["d1", "d2"],
        metadata={"relevance_by_doc_id": {"d1": 2, "d2": 1}},
    )
    contexts = [RetrievalResult("n1", "c1", "d1", "one", 1.0, 1), RetrievalResult("n2", "c2", "d3", "two", 0.5, 2)]
    prediction = RAGAnswer("Q", "A", contexts, metadata={"latency_ms": 10, "context_token_count": 5})
    metrics = evaluate_predictions([item], [prediction], k=2)
    rows = evaluate_prediction_rows([item], [prediction], k=2)
    assert metrics["ndcg_at_2"] > 0
    assert metrics["map_at_2"] == 0.5
    assert metrics["latency_ms_p95"] == 10.0
    assert rows[0]["context_precision"] == 0.5


def test_paired_bootstrap_reports_confidence_interval() -> None:
    baseline = [{"question_id": "q1", "mrr": 0.0}, {"question_id": "q2", "mrr": 0.5}]
    candidate = [{"question_id": "q1", "mrr": 1.0}, {"question_id": "q2", "mrr": 1.0}]
    result = paired_bootstrap_delta(baseline, candidate, "mrr", samples=100)
    assert result["paired_queries"] == 2
    assert result["delta"] == 0.75


def test_aggregate_metric_names_map_to_query_level_measurements() -> None:
    assert _query_metric_name("evidence_complete_rate") == "evidence_complete"
    assert _query_metric_name("citation_f1") == "citation_document_f1"
    assert _query_metric_name("latency_ms_avg") == "latency_ms"
    assert _query_metric_name("estimated_cost_avg") == "estimated_cost"


def test_suite_contract_locks_runs_and_rejects_ineligible_claims(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        f'{{"id":"s","tier":"claim_eligible","dataset":{{"docs":"d","qa":"{fixture}","fingerprint":"sha256:fixture"}},'
        '"mode":"retrieval_only","top_k":5,"required_baselines":["bm25"],"minimum_queries":1,'
        '"reference_baseline":"bm25","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],'
        '"warmup_queries":1,"concurrency":1,"latency_sample_size":5}',
        encoding="utf-8",
    )
    suite = load_suite(suite_path)
    assert resolve_suite(suite, docs=None, qa=None, mode=None, top_k=None)["docs"] == "d"
    verdict = claim_eligibility({**suite, "tier": "smoke_only"}, [], "missing.jsonl")
    assert verdict["eligible"] is False

    # concurrency/latency_sample_size are locked exactly like warmup_queries:
    # no explicit value from the caller resolves to the suite's, an explicit
    # matching value is accepted, and a conflicting one is rejected outright.
    resolved = resolve_suite(
        suite, docs=None, qa=None, mode=None, top_k=None, concurrency=None, latency_sample_size=None
    )
    assert resolved["concurrency"] == 1
    assert resolved["latency_sample_size"] == 5
    resolve_suite(suite, docs=None, qa=None, mode=None, top_k=None, concurrency=1, latency_sample_size=5)
    with pytest.raises(ValueError, match="Suite locks concurrency"):
        resolve_suite(suite, docs=None, qa=None, mode=None, top_k=None, concurrency=8)
    with pytest.raises(ValueError, match="Suite locks latency_sample_size"):
        resolve_suite(suite, docs=None, qa=None, mode=None, top_k=None, latency_sample_size=0)


def test_claim_eligible_suite_requires_concurrency_protocol_fields(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        f'{{"id":"s","tier":"claim_eligible","dataset":{{"docs":"d","qa":"{fixture}","fingerprint":"sha256:fixture"}},'
        '"mode":"retrieval_only","top_k":5,"required_baselines":["bm25"],"minimum_queries":1,'
        '"reference_baseline":"bm25","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],'
        '"warmup_queries":1,"concurrency":4,"latency_sample_size":0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="latency_sample_size must be at least 1 when suite.concurrency > 1"):
        load_suite(suite_path)
