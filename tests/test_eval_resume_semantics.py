from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

import ragbench.evaluation.runner as runner_module
from ragbench.core.base import load_pipeline
from ragbench.core.schema import RAGAnswer
from ragbench.evaluation.runner import run_eval
from ragbench.indexing.artifacts import load_manifest


def _counting_fake_run_single_query(call_log: list[str]) -> Any:
    def fake(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
    ) -> RAGAnswer:
        call_log.append(item.question_id)
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "cost_estimate": {"currency": "USD", "amount": 0.0, "status": "estimated"},
                "components": {},
            },
        )

    return fake


def _run(pipeline: Any, artifact: Path, qa_path: str, output_path: Path, *, resume: bool) -> dict:
    return run_eval(pipeline, str(artifact), qa_path, str(output_path), mode="retrieval_only", resume=resume)


@pytest.fixture
def _ingested_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))
    return artifact


def test_resume_false_ignores_an_existing_matching_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _ingested_artifact: Path
) -> None:
    """Regression: ``run_eval()`` used to call ``_open_checkpoint()``
    unconditionally, so a fresh command with no ``--resume`` would silently
    reuse any on-disk checkpoint whose header happened to match — this is
    the exact bug a user hit ("resuming 100/100 already-completed queries"
    on a command with no ``--resume``)."""
    call_log: list[str] = []
    monkeypatch.setattr(runner_module, "_run_single_query", _counting_fake_run_single_query(call_log))
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    output_path = tmp_path / "eval.json"

    _run(pipeline, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)
    first_run_calls = len(call_log)
    assert first_run_calls == 3  # every question actually executed

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    _run(pipeline2, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)

    # The second call must NOT reuse the first call's checkpoint: every
    # question ran again, doubling the call log.
    assert len(call_log) == first_run_calls * 2


def test_resume_true_reuses_a_matching_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _ingested_artifact: Path
) -> None:
    call_log: list[str] = []
    monkeypatch.setattr(runner_module, "_run_single_query", _counting_fake_run_single_query(call_log))
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    output_path = tmp_path / "eval.json"

    _run(pipeline, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)
    assert len(call_log) == 3
    call_log.clear()

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    report = _run(pipeline2, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=True)

    assert call_log == []  # nothing new ran — every prediction came from the checkpoint
    assert len(report["predictions"]) == 3


def test_resume_true_rejects_checkpoint_after_runtime_fingerprint_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _ingested_artifact: Path
) -> None:
    """``runtime_fingerprint`` is what must catch this: the artifact/corpus
    on disk is untouched (no re-ingest), but query-time code (retriever/
    reranker/generator/verifier) changed — old checkpointed predictions must
    not be reused even though ``resume=True`` was requested."""
    call_log: list[str] = []
    monkeypatch.setattr(runner_module, "_run_single_query", _counting_fake_run_single_query(call_log))
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    output_path = tmp_path / "eval.json"

    _run(pipeline, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)
    assert len(call_log) == 3
    call_log.clear()

    real_manifest = load_manifest(_ingested_artifact)
    mutated_manifest = copy.deepcopy(real_manifest)
    mutated_manifest["runtime"]["runtime_fingerprint"] = "sha256:simulated-query-code-change"
    monkeypatch.setattr(runner_module, "load_manifest", lambda path: mutated_manifest)

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    report = _run(pipeline2, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=True)

    assert len(call_log) == 3  # every question re-executed, not reused from the stale checkpoint
    assert len(report["predictions"]) == 3


@pytest.mark.parametrize("field", ["artifact_version", "ingest_fingerprint"])
def test_resume_true_rejects_checkpoint_after_ingest_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _ingested_artifact: Path, field: str
) -> None:
    call_log: list[str] = []
    monkeypatch.setattr(runner_module, "_run_single_query", _counting_fake_run_single_query(call_log))
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    output_path = tmp_path / "eval.json"

    _run(pipeline, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)
    assert len(call_log) == 3
    call_log.clear()

    real_manifest = load_manifest(_ingested_artifact)
    mutated_manifest = copy.deepcopy(real_manifest)
    if field == "artifact_version":
        mutated_manifest["artifact_version"] = "999"
    else:
        mutated_manifest["runtime"]["ingest_fingerprint"] = "sha256:simulated-reingest"
    monkeypatch.setattr(runner_module, "load_manifest", lambda path: mutated_manifest)

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    _run(pipeline2, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=True)

    assert len(call_log) == 3  # rejected — re-executed instead of reusing the stale checkpoint


def test_resume_true_reuses_checkpoint_when_identity_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _ingested_artifact: Path
) -> None:
    """Sanity check paired with the rejection tests above: an unrelated
    ``load_manifest`` monkeypatch that returns an *equal* (deep-copied)
    manifest must not itself defeat resume."""
    call_log: list[str] = []
    monkeypatch.setattr(runner_module, "_run_single_query", _counting_fake_run_single_query(call_log))
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    output_path = tmp_path / "eval.json"

    _run(pipeline, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=False)
    assert len(call_log) == 3
    call_log.clear()

    real_manifest = load_manifest(_ingested_artifact)
    unchanged_manifest = copy.deepcopy(real_manifest)
    monkeypatch.setattr(runner_module, "load_manifest", lambda path: unchanged_manifest)

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    _run(pipeline2, _ingested_artifact, "datasets/sample/qa.jsonl", output_path, resume=True)

    assert call_log == []  # identity unchanged — checkpoint still reused
