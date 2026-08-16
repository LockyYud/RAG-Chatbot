from __future__ import annotations

from pathlib import Path

from evaluation.runner import run_eval
from raglab.core.base import load_pipeline


def test_ingest_query_eval_smoke(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    report = tmp_path / "eval.json"

    pipeline = load_pipeline("parent_child")
    assert pipeline is not None

    manifest = pipeline.ingest("datasets/sample/docs", str(artifact))
    assert manifest["corpus"]["node_count"] > 0

    answer = pipeline.query(str(artifact), "Điều kiện xét tuyển ngành trí tuệ nhân tạo là gì?")
    assert "tốt nghiệp" in answer.answer.lower()
    assert answer.citations
    assert answer.metadata["verification"]["grounded"] is True
    assert answer.metadata["verification"]["citation_coverage"] == 1.0

    evaluation = run_eval(
        pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(report),
    )
    assert evaluation["metrics"]["queries"] == 3
    assert evaluation["metrics"]["citation_f1"] >= 0.6
