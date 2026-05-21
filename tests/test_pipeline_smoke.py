from __future__ import annotations

from pathlib import Path

from raglab.core.pipeline import ingest, query
from raglab.evaluation.runner import run_eval


def test_ingest_query_eval_smoke(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    report = tmp_path / "eval.json"

    manifest = ingest(
        "configs/pipelines/heading_hybrid.yaml",
        "datasets/sample_docs",
        str(artifact),
    )

    assert manifest["node_count"] > 0
    answer = query(
        "configs/pipelines/heading_hybrid.yaml",
        str(artifact),
        "Điều kiện xét tuyển ngành trí tuệ nhân tạo là gì?",
    )
    assert "tốt nghiệp" in answer.answer.lower()
    assert answer.citations
    assert answer.metadata["verification"]["grounded"] is True
    assert answer.metadata["verification"]["citation_coverage"] == 1.0

    evaluation = run_eval(
        "configs/pipelines/heading_hybrid.yaml",
        str(artifact),
        "datasets/sample_qa/qa.jsonl",
        str(report),
    )
    assert evaluation["metrics"]["queries"] == 3
    assert evaluation["metrics"]["citation_accuracy"] >= 0.6
