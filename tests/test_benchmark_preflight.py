from __future__ import annotations

from pathlib import Path

import pytest

from raglab.benchmarks.runner import run_preflight
from raglab.benchmarks.suites import claim_eligibility, load_suite


def _write_suite(tmp_path: Path, *, tier: str = "claim_eligible") -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(
        '{"fingerprint":"sha256:fixture","queries":1,"metadata":{"corpus_policy":"full_upstream_corpus"}}',
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        f'{{"id":"s","tier":"{tier}","dataset":{{"docs":"datasets/sample/docs","qa":"{fixture}",'
        '"fingerprint":"sha256:fixture"},'
        '"mode":"retrieval_only","top_k":5,"required_baselines":["parent_child"],"minimum_queries":1,'
        '"reference_baseline":"parent_child","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],'
        '"warmup_queries":1,"concurrency":1,"latency_sample_size":5}',
        encoding="utf-8",
    )
    return suite_path, fixture


def test_preflight_fails_claim_eligible_suite_without_faiss_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("raglab.benchmarks.runner.importlib.util.find_spec", lambda name: None)
    suite_path, _ = _write_suite(tmp_path, tier="claim_eligible")

    result = run_preflight(technique_ids=["parent_child"], docs=None, qa=None, suite_path=str(suite_path))

    assert any("faiss" in reason for reason in result["reasons"])
    assert result["ready"] is False


def test_preflight_ignores_missing_faiss_for_non_claim_eligible_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("raglab.benchmarks.runner.importlib.util.find_spec", lambda name: None)
    suite_path, _ = _write_suite(tmp_path, tier="smoke_only")

    result = run_preflight(technique_ids=["parent_child"], docs=None, qa=None, suite_path=str(suite_path))

    assert not any("faiss" in reason for reason in result["reasons"])


def test_preflight_passes_faiss_check_when_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("raglab.benchmarks.runner.importlib.util.find_spec", lambda name: object())
    suite_path, _ = _write_suite(tmp_path, tier="claim_eligible")

    result = run_preflight(technique_ids=["parent_child"], docs=None, qa=None, suite_path=str(suite_path))

    assert not any("faiss" in reason for reason in result["reasons"])


def _ok_row(*, node_count: int, store_backend: str) -> dict:
    return {
        "technique": "naive_rag",
        "status": "ok",
        "node_count": node_count,
        "store_backend": store_backend,
        "effective_components": "",
        "cost_status": "estimated",
        "latency_ms_p95": 10.0,
        "index_size_bytes": 1,
        "index_time_ms": 1.0,
    }


def test_claim_eligibility_rejects_run_that_should_have_used_faiss(tmp_path: Path) -> None:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    row = _ok_row(node_count=5000, store_backend="json_memory")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert verdict["eligible"] is False
    assert any("faiss_local" in reason for reason in verdict["reasons"])


def test_claim_eligibility_accepts_faiss_backend_above_threshold(tmp_path: Path) -> None:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    row = _ok_row(node_count=5000, store_backend="faiss_local")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert not any("faiss_local" in reason for reason in verdict["reasons"])


def test_claim_eligibility_ignores_backend_below_threshold(tmp_path: Path) -> None:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    row = _ok_row(node_count=10, store_backend="json_memory")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert not any("faiss_local" in reason for reason in verdict["reasons"])
