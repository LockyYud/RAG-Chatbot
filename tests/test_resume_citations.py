from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import ragbench.evaluation.runner as runner_module
from ragbench.core.base import load_pipeline
from ragbench.core.schema import Citation
from ragbench.evaluation.runner import run_eval


def test_full_rag_resume_reconstructs_citation_objects_not_raw_dicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a checkpointed prediction's citations are serialized as
    plain JSON dicts on disk (RAGAnswer.to_dict()'s shape). Resuming used to
    wrap that raw list in ``list(...)`` — a no-op copy — leaving
    ``RAGAnswer.citations`` as ``list[dict]`` instead of ``list[Citation]``.
    Anything downstream that calls ``citation.doc_id`` or
    ``citation.to_dict()`` (metrics, the result fingerprint, verifiers) would
    then crash on exactly the predictions that were resumed from a checkpoint,
    not the freshly-run ones. Simulates a real crash (an unrelated exception
    escaping mid-run, not BudgetExceededError) partway through a full_rag run
    that actually produces citations end to end (parent_child's
    CitationExtractiveGenerator — no API key required)."""
    artifact = tmp_path / "artifact"
    ingest_pipeline = load_pipeline("parent_child")
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))

    output_path = tmp_path / "eval.json"
    original_run_single_query = runner_module._run_single_query
    call_count = {"n": 0}

    def flaky_run_single_query(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated crash")
        return original_run_single_query(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_run_single_query", flaky_run_single_query)
    crashing_pipeline = load_pipeline("parent_child")
    assert crashing_pipeline is not None

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_eval(crashing_pipeline, str(artifact), "datasets/sample/qa.jsonl", str(output_path))

    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    assert checkpoint_path.exists()
    checkpoint_lines = [
        json.loads(line) for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    prediction_records = [line for line in checkpoint_lines if line.get("type") == "prediction"]
    assert len(prediction_records) == 1  # crashed on the 2nd question; only the 1st got checkpointed
    checkpointed_citations = prediction_records[0]["prediction"]["citations"]
    assert checkpointed_citations  # parent_child's full_rag mode always cites its one best chunk
    assert isinstance(checkpointed_citations[0], dict)  # on-disk shape is plain JSON, not a Citation

    monkeypatch.setattr(runner_module, "_run_single_query", original_run_single_query)
    resuming_pipeline = load_pipeline("parent_child")
    assert resuming_pipeline is not None
    report = run_eval(resuming_pipeline, str(artifact), "datasets/sample/qa.jsonl", str(output_path))

    assert len(report["predictions"]) == 3  # every question present after resume completed the run
    for prediction in report["predictions"]:
        assert prediction["citations"]
        for citation in prediction["citations"]:
            assert set(citation) == {"citation_id", "doc_id", "chunk_id", "start_char", "end_char"}
    # Building metrics/result_fingerprint from the resumed (checkpoint-sourced)
    # predictions must not have crashed on citation.doc_id / citation.to_dict().
    assert report["result_fingerprint"]
    assert report["metrics"]["citation_document_precision"] == 1.0


def test_prediction_from_checkpoint_record_returns_real_citation_instances() -> None:
    """Direct unit check of the deserialization helper itself."""
    record: dict[str, Any] = {
        "query": "Q",
        "answer": "A",
        "contexts": [],
        "citations": [
            {"citation_id": "C1", "doc_id": "d1", "chunk_id": "d1:c1", "start_char": 3, "end_char": 9},
        ],
        "abstained": False,
        "metadata": {},
    }
    prediction = runner_module._prediction_from_checkpoint_record(record)
    assert len(prediction.citations) == 1
    citation = prediction.citations[0]
    assert isinstance(citation, Citation)
    assert citation.doc_id == "d1"
    assert citation.start_char == 3
    assert citation.to_dict() == record["citations"][0]
