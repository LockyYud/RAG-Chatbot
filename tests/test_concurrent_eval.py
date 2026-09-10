from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import ragbench.evaluation.runner as runner_module
from ragbench.core.base import load_pipeline
from ragbench.core.schema import RAGAnswer
from ragbench.evaluation.runner import BudgetExceededError, run_eval

QUESTION_COUNT = 8


def _write_qa_dataset(path: Path, count: int = QUESTION_COUNT) -> None:
    rows = [
        {
            "question_id": f"q{i}",
            "question": f"Question number {i}?",
            "expected_doc_ids": [],
            "expected_chunk_ids": [],
            "expected_citations": [],
            "metadata": {"is_answerable": True},
        }
        for i in range(count)
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_fake_run_single_query(sleep_seconds: dict[str, float], call_log: list[str], log_lock: threading.Lock):
    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
    ) -> RAGAnswer:
        time.sleep(sleep_seconds.get(item.question_id, 0.0))
        with log_lock:
            call_log.append(item.question_id)
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "latency_ms": sleep_seconds.get(item.question_id, 0.0) * 1000,
                "cost_estimate": {
                    "currency": "USD",
                    "amount": 0.1,
                    "embedding_cost": 0.0,
                    "chat_cost": 0.1,
                    "status": "estimated",
                },
                "components": {},
            },
        )

    return fake_run_single_query


def _make_fixed_cost_fake_run_single_query(
    *, cost_per_query: float, sleep_seconds: float, call_log: list[str], log_lock: threading.Lock
):
    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
    ) -> RAGAnswer:
        time.sleep(sleep_seconds)
        with log_lock:
            call_log.append(item.question_id)
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "latency_ms": sleep_seconds * 1000,
                "cost_estimate": {
                    "currency": "USD",
                    "amount": cost_per_query,
                    "embedding_cost": 0.0,
                    "chat_cost": cost_per_query,
                    "status": "estimated",
                },
                "components": {},
            },
        )

    return fake_run_single_query


@pytest.fixture
def parent_child_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))
    return artifact


def test_concurrent_run_matches_sequential_accounting_and_preserves_item_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_child_artifact: Path
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write_qa_dataset(qa_path)
    # Later-submitted items finish first — a real stress test that predictions
    # get reassembled in `items` order rather than completion order.
    sleep_seconds = {f"q{i}": (QUESTION_COUNT - i) * 0.01 for i in range(QUESTION_COUNT)}

    pipeline = load_pipeline("parent_child")
    assert pipeline is not None

    call_log_sequential: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_single_query",
        _make_fake_run_single_query(sleep_seconds, call_log_sequential, threading.Lock()),
    )
    sequential_report = run_eval(
        pipeline, str(parent_child_artifact), str(qa_path), str(tmp_path / "sequential.json"), concurrency=1
    )

    pipeline2 = load_pipeline("parent_child")
    assert pipeline2 is not None
    call_log_concurrent: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_single_query",
        _make_fake_run_single_query(sleep_seconds, call_log_concurrent, threading.Lock()),
    )
    started = time.perf_counter()
    concurrent_report = run_eval(
        pipeline2,
        str(parent_child_artifact),
        str(qa_path),
        str(tmp_path / "concurrent.json"),
        concurrency=4,
        latency_sample_size=2,
    )
    elapsed = time.perf_counter() - started

    # Predictions land back in items order, regardless of completion order.
    assert [p["question_id"] for p in concurrent_report["predictions"]] == [f"q{i}" for i in range(QUESTION_COUNT)]
    # Accounting is concurrency-invariant: same per-item costs, same total.
    assert (
        concurrent_report["cost_summary"]["pipeline_cost"]["total"]
        == sequential_report["cost_summary"]["pipeline_cost"]["total"]
    )
    # Real concurrency happened: completion order differs from submission order,
    # and total wall time is well under the fully-sequential sum of sleeps.
    assert call_log_concurrent != [f"q{i}" for i in range(QUESTION_COUNT)]
    assert elapsed < sum(sleep_seconds.values())

    performance = concurrent_report["performance"]
    assert performance["quality_pass"]["mode"] == "concurrent"
    assert performance["quality_pass"]["workers"] == 4
    assert performance["quality_pass"]["queries"] == QUESTION_COUNT - 2  # 2 went to the sequential/latency sample
    assert performance["quality_pass"]["throughput_qps"] > 0
    assert performance["latency_pass"]["mode"] == "sequential"
    assert performance["latency_pass"]["sampled_queries"] == 2
    # latency_pass numbers come only from the sequential prefix (q0, q1: 8ms, 7ms),
    # not from the concurrent remainder's contended timings.
    assert performance["latency_pass"]["latency_ms_p95"] <= 80.0

    assert "performance" not in sequential_report


def test_concurrent_run_checkpoint_has_no_corrupted_or_duplicate_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_child_artifact: Path
) -> None:
    qa_path = tmp_path / "qa.jsonl"
    _write_qa_dataset(qa_path)
    sleep_seconds = {f"q{i}": 0.01 for i in range(QUESTION_COUNT)}
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    monkeypatch.setattr(
        runner_module, "_run_single_query", _make_fake_run_single_query(sleep_seconds, [], threading.Lock())
    )

    output_path = tmp_path / "eval.json"
    run_eval(pipeline, str(parent_child_artifact), str(qa_path), str(output_path), concurrency=4)

    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]  # raises if any line is malformed/interleaved
    assert records[0]["type"] == "header"
    prediction_records = [record for record in records if record["type"] == "prediction"]
    assert len(prediction_records) == QUESTION_COUNT
    assert sorted(record["question_id"] for record in prediction_records) == [f"q{i}" for i in range(QUESTION_COUNT)]


def test_concurrent_run_headline_latency_metrics_exclude_contended_quality_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_child_artifact: Path
) -> None:
    """metrics.latency_ms_* must reflect only the uncontended sequential sample —
    claim_eligibility() reads this exact field, so a contended aggregate here
    would silently pass off a throughput-degraded number as a real latency claim."""
    qa_path = tmp_path / "qa.jsonl"
    _write_qa_dataset(qa_path)
    # Sequential (latency-pass) items are fast; concurrent (quality-pass) items
    # report an enormous latency, standing in for real contention noise.
    latency_sample_size = 2
    sleep_seconds = {
        f"q{i}": 0.005 if i < latency_sample_size else 5.0  # seconds — never actually slept, see below
        for i in range(QUESTION_COUNT)
    }
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None

    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
    ) -> RAGAnswer:
        # Report the large "latency" without actually sleeping that long, so the test stays fast.
        reported_ms = sleep_seconds[item.question_id] * 1000
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "latency_ms": reported_ms,
                "cost_estimate": {"currency": "USD", "amount": 0.0, "status": "estimated"},
                "components": {},
            },
        )

    monkeypatch.setattr(runner_module, "_run_single_query", fake_run_single_query)

    report = run_eval(
        pipeline,
        str(parent_child_artifact),
        str(qa_path),
        str(tmp_path / "eval.json"),
        concurrency=4,
        latency_sample_size=latency_sample_size,
    )

    # Only the two fast sequential-pass latencies (5ms each) feed the headline metric.
    assert report["metrics"]["latency_ms_p95"] == pytest.approx(5.0, abs=0.001)
    assert report["metrics"]["latency_ms_avg"] == pytest.approx(5.0, abs=0.001)
    # ...even though every cutoff bucket agrees, and the raw per-query rows still
    # show the true (contended) values for anyone who wants to inspect them.
    for bucket in report["metrics_by_cutoff"].values():
        assert bucket["latency_ms_p95"] == pytest.approx(5.0, abs=0.001)
    assert any(row["latency_ms"] > 1000 for row in report["query_metrics"])


def test_concurrent_budget_guard_overshoot_is_bounded_by_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, parent_child_artifact: Path
) -> None:
    """Regression test for the front-loaded-executor bug: submitting every
    remaining item up front let the guard trip only after most of the dataset
    had already been dispatched. A bounded sliding window of in-flight futures
    must cap overshoot at (at most) `concurrency` extra queries."""
    total_queries = 20
    concurrency = 3
    qa_path = tmp_path / "qa.jsonl"
    _write_qa_dataset(qa_path, count=total_queries)
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    call_log: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_run_single_query",
        _make_fixed_cost_fake_run_single_query(
            cost_per_query=1.0, sleep_seconds=0.02, call_log=call_log, log_lock=threading.Lock()
        ),
    )

    with pytest.raises(BudgetExceededError):
        run_eval(
            pipeline,
            str(parent_child_artifact),
            str(qa_path),
            str(tmp_path / "eval.json"),
            concurrency=concurrency,
            latency_sample_size=0,  # force every item straight into the concurrent scheduler
            max_estimated_cost_usd=5.5,  # trips after the 6th $1 query completes
        )

    # Trips at 6 completed queries ($6.0 > $5.5); at most `concurrency` more
    # can already be in flight at that moment. Nowhere near the full dataset.
    assert len(call_log) <= 6 + concurrency
    assert len(call_log) < total_queries
