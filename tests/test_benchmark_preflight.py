from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.benchmarks.runner import run_preflight
from ragbench.benchmarks.suites import claim_eligibility, load_suite


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
        '"reference_baseline":"parent_child","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],"primary_metric":"mrr","minimum_effect":0.0,'
        '"warmup_queries":1,"concurrency":1,"latency_sample_size":5}',
        encoding="utf-8",
    )
    return suite_path, fixture


def test_preflight_fails_claim_eligible_suite_without_faiss_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ragbench.benchmarks.runner.importlib.util.find_spec", lambda name: None)
    suite_path, _ = _write_suite(tmp_path, tier="claim_eligible")

    result = run_preflight(technique_ids=["parent_child"], docs=None, qa=None, suite_path=str(suite_path))

    assert any("faiss" in reason for reason in result["reasons"])
    assert result["ready"] is False


def test_preflight_ignores_missing_faiss_for_non_claim_eligible_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ragbench.benchmarks.runner.importlib.util.find_spec", lambda name: None)
    suite_path, _ = _write_suite(tmp_path, tier="smoke_only")

    result = run_preflight(technique_ids=["parent_child"], docs=None, qa=None, suite_path=str(suite_path))

    assert not any("faiss" in reason for reason in result["reasons"])


def test_preflight_passes_faiss_check_when_installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragbench.benchmarks.runner.importlib.util.find_spec", lambda name: object())
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


def test_claim_eligibility_flags_mismatched_backends_as_a_production_reason(tmp_path: Path) -> None:
    """Regression: a baseline just under the node-count threshold (json_memory)
    and a candidate just over it (faiss_local) each individually "correctly"
    auto-selected their backend, but a latency/index-size comparison between
    them is then partly a claim about which backend each happened to trigger,
    not about the techniques. This must not be silently claim-eligible."""
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    baseline = {**_ok_row(node_count=1500, store_backend="json_memory"), "technique": "parent_child"}
    candidate = {**_ok_row(node_count=2500, store_backend="faiss_local"), "technique": "naive_rag"}

    verdict = claim_eligibility(suite, [baseline, candidate], str(fixture))

    # Both rows individually respect the threshold, so this is purely a
    # production-claim concern — it must never appear in the base "reasons"
    # that gate protocol/quality eligibility.
    assert not any("backend" in reason for reason in verdict["reasons"])
    assert any("different vector store backends" in reason for reason in verdict["production_reasons"])


def test_claim_eligibility_accepts_matching_backends_across_techniques(tmp_path: Path) -> None:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    baseline = {**_ok_row(node_count=10, store_backend="json_memory"), "technique": "parent_child"}
    candidate = {**_ok_row(node_count=20, store_backend="json_memory"), "technique": "naive_rag"}

    verdict = claim_eligibility(suite, [baseline, candidate], str(fixture))

    assert not any("different vector store backends" in reason for reason in verdict["production_reasons"])


def test_claim_eligibility_rejects_a_dataset_marked_dev(tmp_path: Path) -> None:
    """Regression: config gets tuned against a dev split — evaluating (or
    claiming an improvement) against that same split is exactly the
    benchmark-overfitting risk a held-out test split exists to prevent."""
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    (fixture / "manifest.json").write_text(
        '{"fingerprint":"sha256:fixture","queries":1,'
        '"metadata":{"corpus_policy":"full_upstream_corpus","split":"dev"}}',
        encoding="utf-8",
    )
    suite = load_suite(suite_path)
    row = _ok_row(node_count=10, store_backend="json_memory")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert verdict["eligible"] is False
    assert any("split is 'dev'" in reason for reason in verdict["reasons"])
    assert verdict["dataset_split"] == "dev"


def test_claim_eligibility_allows_an_unmarked_dataset_split(tmp_path: Path) -> None:
    """No existing dataset in this repo has adopted the split convention yet —
    an absent metadata.split must not be treated as though it were "dev"."""
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = load_suite(suite_path)
    row = _ok_row(node_count=10, store_backend="json_memory")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert not any("split" in reason for reason in verdict["reasons"])
    assert verdict["dataset_split"] == "unspecified"


def test_claim_eligibility_surfaces_tuned_on_dataset_and_config_frozen_at(tmp_path: Path) -> None:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible")
    suite = {
        **load_suite(suite_path),
        "tuned_on_dataset": "datasets/processed/vi_wiki_dev",
        "config_frozen_at": "2026-08-01T00:00:00Z",
        "required_baselines": ["naive_rag"],
    }
    row = _ok_row(node_count=10, store_backend="json_memory")

    verdict = claim_eligibility(suite, [row], str(fixture))

    assert verdict["tuned_on_dataset"] == "datasets/processed/vi_wiki_dev"
    assert verdict["config_frozen_at"] == "2026-08-01T00:00:00Z"
