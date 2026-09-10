from __future__ import annotations

import json

import pytest

from ragbench.benchmarks.runner import _query_metric_name
from ragbench.benchmarks.statistics import paired_bootstrap_delta
from ragbench.benchmarks.suites import claim_eligibility, load_suite, resolve_suite
from ragbench.core.base import list_pipelines, load_pipeline
from ragbench.core.config import load_config
from ragbench.core.schema import Citation, EvalItem, RAGAnswer, RetrievalResult
from ragbench.evaluation.metrics import evaluate_prediction_rows, evaluate_predictions


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
    metadata = load_config("ragbench/techniques/naive_rag/technique.yaml")
    assert metadata.get("id") == "naive_rag"


def test_metrics_include_judge_scores_when_present() -> None:
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["doc"])
    result = RetrievalResult("n1", "c1", "doc", "text", 1.0, 1)
    prediction = RAGAnswer(
        query="Q",
        answer="A",
        contexts=[result],
        citations=[Citation(citation_id="C1", doc_id="doc", chunk_id="c1")],
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
        '"reference_baseline":"bm25","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],"primary_metric":"mrr","minimum_effect":0.0,"coverage":{"min_retrieval_coverage":0.5},'
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
        '"reference_baseline":"bm25","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],"primary_metric":"mrr","minimum_effect":0.0,"coverage":{"min_retrieval_coverage":0.5},'
        '"warmup_queries":1,"concurrency":4,"latency_sample_size":0}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="latency_sample_size must be at least 1 when suite.concurrency > 1"):
        load_suite(suite_path)


def _base_suite_json(fixture, **extra_top_level: object) -> str:
    payload = {
        "id": "s",
        "tier": "claim_eligible",
        "dataset": {"docs": "d", "qa": str(fixture), "fingerprint": "sha256:fixture"},
        "mode": "retrieval_only",
        "top_k": 5,
        "required_baselines": ["bm25"],
        "minimum_queries": 1,
        "reference_baseline": "bm25",
        "cutoffs": [5],
        "bootstrap_samples": 100,
        "primary_metrics": ["mrr"],
        "warmup_queries": 1,
        "concurrency": 1,
        "latency_sample_size": 5,
        "coverage": {"min_retrieval_coverage": 0.5},
    }
    payload.update(extra_top_level)
    return json.dumps(payload)


def test_claim_eligible_suite_requires_primary_metric_without_pareto_mode(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(_base_suite_json(fixture), encoding="utf-8")
    with pytest.raises(ValueError, match="must declare suite.primary_metric"):
        load_suite(suite_path)


def test_claim_eligible_suite_rejects_primary_metric_alongside_pareto_mode(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        _base_suite_json(fixture, pareto_improvement=True, primary_metric="mrr"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_suite(suite_path)


def test_claim_eligible_suite_accepts_pareto_mode_without_primary_metric(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(_base_suite_json(fixture, pareto_improvement=True), encoding="utf-8")
    suite = load_suite(suite_path)
    assert suite["pareto_improvement"] is True


def test_claim_eligible_suite_rejects_malformed_coverage_block(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        _base_suite_json(
            fixture, primary_metric="mrr", minimum_effect=0.0, coverage={"min_retrieval_coverage": 1.5}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_retrieval_coverage must be a ratio between 0 and 1"):
        load_suite(suite_path)


def test_claim_eligible_suite_requires_a_coverage_block(tmp_path) -> None:
    """Regression: coverage was optional, so validate_profile() fell back to
    "at least one qualifying item" — a suite could be claim-eligible while a
    single query in the entire dataset carried a qrel."""
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    payload = json.loads(_base_suite_json(fixture, primary_metric="mrr", minimum_effect=0.0))
    del payload["coverage"]
    suite_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must declare a suite.coverage block"):
        load_suite(suite_path)


def test_coverage_block_is_still_optional_below_claim_eligible_tier(tmp_path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    payload = json.loads(_base_suite_json(fixture))
    payload["tier"] = "exploratory"
    del payload["coverage"]
    suite_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_suite(suite_path)["tier"] == "exploratory"


def _load_suite_with_coverage(tmp_path, coverage, **extra):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text('{"fingerprint":"sha256:fixture"}', encoding="utf-8")
    suite_path = tmp_path / "suite.json"
    payload = json.loads(_base_suite_json(fixture, primary_metric="mrr", minimum_effect=0.0, **extra))
    payload["coverage"] = coverage
    suite_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_suite(suite_path)


def test_claim_eligible_suite_rejects_an_empty_coverage_mapping(tmp_path) -> None:
    """Regression: requiring only "a mapping" let ``coverage: {}`` through. It
    declares nothing, so validate_profile() fell straight back to the "at least
    one qualifying item" floor — the exact hole the coverage requirement was
    added to close."""
    with pytest.raises(ValueError, match="suite.coverage is empty"):
        _load_suite_with_coverage(tmp_path, {})


def test_claim_eligible_suite_rejects_coverage_that_does_not_constrain_the_profile(tmp_path) -> None:
    """A non-empty block is not enough either: declaring an unrelated slice
    leaves the profile's own slice on the weak fallback."""
    with pytest.raises(ValueError, match=r"must declare suite\.coverage\.min_retrieval_coverage"):
        _load_suite_with_coverage(tmp_path, {"min_per_question_type": {"factual": 10}})


@pytest.mark.parametrize(
    ("profile", "mode", "coverage", "missing"),
    [
        ("retrieval", "retrieval_only", {"min_unanswerable_questions": 0}, "min_retrieval_coverage"),
        ("single_hop_rag", "full_rag", {"min_retrieval_coverage": 0.9}, "min_unanswerable_questions"),
        (
            "multi_hop_rag",
            "full_rag",
            {"min_retrieval_coverage": 0.9, "min_unanswerable_questions": 0},
            "min_multi_hop_questions",
        ),
        (
            "citation_rag",
            "full_rag",
            {"min_retrieval_coverage": 0.9, "min_unanswerable_questions": 0},
            "min_citation_coverage",
        ),
    ],
)
def test_each_profile_requires_its_own_coverage_thresholds(tmp_path, profile, mode, coverage, missing) -> None:
    with pytest.raises(ValueError, match=rf"must declare suite\.coverage\.{missing}"):
        _load_suite_with_coverage(tmp_path, coverage, profile=profile, mode=mode)


@pytest.mark.parametrize(
    ("profile", "mode", "coverage"),
    [
        ("retrieval", "retrieval_only", {"min_retrieval_coverage": 0.95}),
        ("single_hop_rag", "full_rag", {"min_retrieval_coverage": 0.95, "min_unanswerable_questions": 10}),
        (
            "multi_hop_rag",
            "full_rag",
            {"min_retrieval_coverage": 0.95, "min_multi_hop_questions": 20, "min_unanswerable_questions": 10},
        ),
        (
            "citation_rag",
            "full_rag",
            {"min_retrieval_coverage": 0.95, "min_citation_coverage": 1.0, "min_unanswerable_questions": 10},
        ),
    ],
)
def test_a_profile_appropriate_coverage_block_is_accepted(tmp_path, profile, mode, coverage) -> None:
    suite = _load_suite_with_coverage(tmp_path, coverage, profile=profile, mode=mode)
    assert suite["coverage"] == coverage


def test_declaring_zero_is_an_accepted_explicit_opt_out(tmp_path) -> None:
    """A suite may legitimately not measure a slice — but it has to say so,
    rather than leaving the threshold absent and inheriting a weak default."""
    suite = _load_suite_with_coverage(
        tmp_path,
        {"min_retrieval_coverage": 0.0, "min_unanswerable_questions": 0},
        profile="single_hop_rag",
        mode="full_rag",
    )
    assert suite["coverage"]["min_retrieval_coverage"] == 0.0
