from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from evaluation.metrics import evaluate_predictions
from evaluation.runner import run_eval
from raglab.benchmarks.experiments import run_experiment_matrix
from raglab.benchmarks.runner import _matching_report, has_failed_runs
from raglab.cli.main import _compare
from raglab.core.base import get_pipeline_spec, load_pipeline, load_pipeline_for_artifact
from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import BuiltContext, EvalItem, RAGAnswer, RetrievalResult
from raglab.datasets.schema import (
    DocumentRecord,
    PreparedDataset,
    QrelRecord,
    QueryRecord,
    sample_processed_dataset,
    validate_processed_dataset,
    write_prepared_dataset,
)
from raglab.inference.generators.chat import ChatGenerator
from raglab.providers.llm_client import ChatCompletionResult


def test_artifact_config_is_source_of_truth(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child", params={"child_size": 77, "top_k": 4})
    assert pipeline is not None
    manifest = pipeline.ingest("datasets/sample/docs", str(artifact))

    assert manifest["artifact_version"] == "3"
    assert manifest["pipeline"]["config"]["child_size"] == 77
    assert manifest["pipeline"]["config_fingerprint"].startswith("sha256:")

    loaded = load_pipeline_for_artifact("parent_child", artifact, {"top_k": 9})
    assert loaded.resolved_config()["child_size"] == 77
    assert loaded.resolved_config()["top_k"] == 9
    with pytest.raises(ValueError, match="persisted pipeline"):
        load_pipeline_for_artifact("parent_child", artifact, {"child_size": 88})
    with pytest.raises(RuntimeError, match="belongs to pipeline"):
        load_pipeline_for_artifact("naive_rag", artifact)


def test_direct_query_rejects_locked_config_drift(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child", params={"child_size": 77})
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))
    mismatched = load_pipeline("parent_child", params={"child_size": 99})
    assert mismatched is not None
    with pytest.raises(RuntimeError, match="configuration does not match"):
        mismatched.query(str(artifact), "question", mode="retrieval_only")


def test_self_rag_retrieval_only_does_not_touch_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import techniques.self_rag_2023.pipeline as self_rag_module

    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("self_rag_2023")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    def unexpected_provider_call(model: str) -> None:
        raise AssertionError(f"provider preflight should not run for {model}")

    monkeypatch.setattr(self_rag_module, "check_provider_ready", unexpected_provider_call)
    answer = pipeline.query(str(artifact), "question", mode="retrieval_only")
    assert answer.metadata["verification"]["status"] == "skipped"
    assert answer.metadata["cost_estimate"]["amount"] == 0.0


def test_metrics_exclude_unanswerable_from_retrieval_denominator() -> None:
    result = RetrievalResult("n1", "c1", "d1", "text", 1.0, 1)
    items = [
        EvalItem("answerable", "Q1", expected_doc_ids=["d1"], expected_citations=["d1"]),
        EvalItem("unanswerable", "Q2", metadata={"is_answerable": False}),
    ]
    predictions = [
        RAGAnswer("Q1", "A", [result], citations=["d1"]),
        RAGAnswer("Q2", "Không đủ bằng chứng", [], abstained=True),
    ]
    metrics = evaluate_predictions(items, predictions)

    assert metrics["retrieval_queries_evaluated"] == 1
    assert metrics["recall_at_5"] == 1.0
    assert metrics["citation_precision"] == 1.0
    assert metrics["citation_recall"] == 1.0
    assert metrics["citation_f1"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0


def test_evaluation_raises_query_depth_to_requested_top_k(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child", params={"top_k": 1, "rerank_top_k": 1})
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    report = run_eval(
        pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(tmp_path / "eval.json"),
        top_k=4,
        cutoffs=[2, 4],
    )

    assert report["run_metadata"]["evaluation_query_overrides"] == {"top_k": 4, "rerank_top_k": 4}
    assert all(len(prediction["contexts"]) == 4 for prediction in report["predictions"])
    assert {"2", "4"} == set(report["metrics_by_cutoff"])
    assert all(len(rows) == 3 for rows in report["query_metrics_by_cutoff"].values())


def test_chat_generator_does_not_invent_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = ChatGenerator(model="fake")
    monkeypatch.setattr(
        generator._client,
        "create_chat_completion",
        lambda **kwargs: ChatCompletionResult("answer without marker", {}, 1.0, 0.0),
    )
    result = RetrievalResult("n1", "c1", "d1", "evidence", 1.0, 1)
    context = BuiltContext("[C1] d1\nevidence", [result], {"C1": result}, 3)
    assert generator.generate("Q", context).citations == []


def test_dataset_overwrite_removes_stale_docs_and_sampling_is_seeded(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    original = PreparedDataset(
        "fixture",
        [DocumentRecord("d1", "one"), DocumentRecord("d2", "two")],
        [QueryRecord("q1", "one?"), QueryRecord("q2", "two?")],
        [QrelRecord("q1", "d1"), QrelRecord("q2", "d2")],
    )
    write_prepared_dataset(original, output)
    replacement = PreparedDataset(
        "fixture2", [DocumentRecord("d1", "one")], [QueryRecord("q1", "one?")], [QrelRecord("q1", "d1")]
    )
    with pytest.raises(FileExistsError):
        write_prepared_dataset(replacement, output)
    write_prepared_dataset(replacement, output, overwrite=True)
    assert {path.stem for path in (output / "docs").glob("*.md")} == {"d1"}

    source = tmp_path / "source"
    write_prepared_dataset(original, source)
    first = sample_processed_dataset(source, tmp_path / "sample1", 1, seed=9)
    second = sample_processed_dataset(source, tmp_path / "sample2", 1, seed=9)
    assert first["fingerprint"] == second["fingerprint"]
    assert validate_processed_dataset(tmp_path / "sample1")["fingerprint"] == first["fingerprint"]


def test_technique_discovery_works_outside_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "raglab.cli.main", "techniques", "list"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    ids = {item["id"] for item in json.loads(completed.stdout)["techniques"]}
    assert "parent_child" in ids


def test_canonical_fingerprint_ignores_mapping_order() -> None:
    assert canonical_fingerprint({"a": 1, "b": 2}) == canonical_fingerprint({"b": 2, "a": 1})


def test_resume_requires_every_run_fingerprint_to_match(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    manifest = pipeline.ingest("datasets/sample/docs", str(artifact))
    dataset_fingerprint = "sha256:dataset"
    report_path = tmp_path / "parent_child_fixture_eval.json"
    report_path.write_text(
        json.dumps(
            {
                "report_schema_version": "2",
                "run_metadata": {
                    "artifact_fingerprint": manifest["corpus"]["fingerprint"],
                    "pipeline_config_fingerprint": manifest["pipeline"]["config_fingerprint"],
                    "dataset_fingerprint": dataset_fingerprint,
                    "mode": "full_rag",
                    "top_k": 5,
                    "evaluation_profile": "citation_rag",
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        _matching_report(tmp_path, "parent_child", manifest, dataset_fingerprint, "full_rag", 5, "citation_rag")
        == report_path
    )
    assert (
        _matching_report(tmp_path, "parent_child", manifest, dataset_fingerprint, "retrieval_only", 5, "retrieval")
        is None
    )
    assert (
        _matching_report(tmp_path, "parent_child", manifest, dataset_fingerprint, "full_rag", 6, "citation_rag")
        is None
    )
    assert _matching_report(tmp_path, "parent_child", manifest, "sha256:other", "full_rag", 5, "citation_rag") is None


def test_compare_keeps_two_artifacts_of_same_technique_separate(tmp_path: Path) -> None:
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    first = tmp_path / "artifact-one"
    second = tmp_path / "artifact-two"
    pipeline.ingest("datasets/sample/docs", str(first))
    docs = tmp_path / "other-docs"
    docs.mkdir()
    (docs / "other.md").write_text("# Tài liệu khác\n\nNội dung độc lập.", encoding="utf-8")
    pipeline.ingest(str(docs), str(second))

    output = tmp_path / "compare.json"
    _compare(
        [f"parent_child={first}", f"parent_child={second}"],
        "datasets/sample/qa.jsonl",
        str(output),
        5,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    reports = {row["report"] for row in payload["runs"]}
    assert len(reports) == 2
    assert all(Path(report).exists() for report in reports)
    assert payload["warnings"]


def test_offline_result_fingerprint_is_reproducible(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))
    query_pipeline = load_pipeline_for_artifact("parent_child", artifact)
    first = run_eval(
        query_pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(tmp_path / "first.json"),
    )
    second = run_eval(
        query_pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(tmp_path / "second.json"),
    )
    assert first["result_fingerprint"] == second["result_fingerprint"]


def test_benchmark_failure_status_is_machine_detectable() -> None:
    assert has_failed_runs({"runs": [{"technique": "parent_child", "status": "ok"}]}) is False
    assert has_failed_runs({"runs": [{"technique": "unknown", "status": "failed"}]}) is True


def test_technique_spec_and_artifact_provenance_are_machine_readable(tmp_path: Path) -> None:
    assert "citation_rag" in get_pipeline_spec("parent_child").evaluation_profiles
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    artifact = tmp_path / "artifact"
    pipeline.ingest("datasets/sample/docs", str(artifact))
    manifest, _ = pipeline.load_artifact(str(artifact))
    assert manifest["runtime"]["source_fingerprint"].startswith("sha256:")
    assert "nodes.json" in manifest["extra"]["artifact_files"]


def test_experiment_matrix_keeps_trials_and_comparisons_separate(tmp_path: Path) -> None:
    matrix = run_experiment_matrix(
        technique_ids=["parent_child", "self_rag_2023"],
        docs="datasets/sample/docs",
        qa="datasets/sample/qa.jsonl",
        output=str(tmp_path / "matrix"),
        trials=2,
        mode="retrieval_only",
        profile="retrieval",
    )
    assert len(matrix["trials"]) == 2
    assert (tmp_path / "matrix" / "matrix.json").exists()
    assert all(item["result"]["comparisons"] for item in matrix["trials"])
