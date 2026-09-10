from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragbench.benchmarks.runner import run_preflight
from ragbench.benchmarks.suites import claim_eligibility, load_suite


def _write_manifest(fixture: Path, **metadata_overrides: Any) -> None:
    metadata: dict[str, Any] = {"corpus_policy": "full_upstream_corpus", "protocol_split": "test"}
    metadata.update(metadata_overrides)
    metadata = {key: value for key, value in metadata.items() if value is not None}
    (fixture / "manifest.json").write_text(
        json.dumps({"fingerprint": "sha256:fixture", "queries": 1, "metadata": metadata}),
        encoding="utf-8",
    )


def _write_suite(tmp_path: Path, *, tier: str = "claim_eligible", **metadata_overrides: Any) -> tuple[Path, Path]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _write_manifest(fixture, **metadata_overrides)
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        f'{{"id":"s","tier":"{tier}","dataset":{{"docs":"datasets/sample/docs","qa":"{fixture}",'
        '"fingerprint":"sha256:fixture"},'
        '"mode":"retrieval_only","top_k":5,"required_baselines":["parent_child"],"minimum_queries":1,'
        '"reference_baseline":"parent_child","cutoffs":[5],"bootstrap_samples":100,"primary_metrics":["mrr"],"primary_metric":"mrr","minimum_effect":0.0,'
        '"coverage":{"min_retrieval_coverage":0.5},'
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


def _split_verdict(tmp_path: Path, **metadata_overrides: Any) -> dict[str, Any]:
    suite_path, fixture = _write_suite(tmp_path, tier="claim_eligible", **metadata_overrides)
    suite = load_suite(suite_path)
    row = _ok_row(node_count=10, store_backend="json_memory")
    return claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))


def test_claim_eligibility_rejects_a_dataset_marked_dev(tmp_path: Path) -> None:
    """Config gets tuned against a dev split — evaluating (or claiming an
    improvement) against that same split is exactly the benchmark-overfitting
    risk a held-out test split exists to prevent."""
    verdict = _split_verdict(tmp_path, protocol_split="dev")

    assert verdict["eligible"] is False
    assert any("protocol_split is 'dev'" in reason for reason in verdict["reasons"])
    assert verdict["dataset_protocol_split"] == "dev"


def test_claim_eligibility_rejects_a_dataset_that_declares_no_protocol_split(tmp_path: Path) -> None:
    """Regression: an absent label used to be read as "fine, not dev". It is
    not an assertion of anything — nobody has said the data was held out, so a
    claim built on it is unsupported by default rather than by exception."""
    verdict = _split_verdict(tmp_path, protocol_split=None)

    assert verdict["eligible"] is False
    assert any("does not declare metadata.protocol_split" in reason for reason in verdict["reasons"])
    assert verdict["dataset_protocol_split"] == "unspecified"


def test_claim_eligibility_does_not_accept_an_upstream_split_name_as_a_held_out_declaration(
    tmp_path: Path,
) -> None:
    """Regression: the committed Vietnamese dataset carries upstream
    ``split: "validation"`` — a dev split under another name. Checking only
    for the literal string "dev" let it through as claim-eligible."""
    verdict = _split_verdict(tmp_path, protocol_split=None, split="validation")

    assert verdict["eligible"] is False
    assert any("upstream metadata.split is 'validation'" in reason for reason in verdict["reasons"])
    # The upstream name is still reported; it just no longer decides anything.
    assert verdict["dataset_split"] == "validation"
    assert verdict["dataset_protocol_split"] == "unspecified"


def test_claim_eligibility_rejects_an_unrecognised_protocol_split_value(tmp_path: Path) -> None:
    """A typo must fail loudly rather than quietly behaving like "not test"."""
    verdict = _split_verdict(tmp_path, protocol_split="Test")

    assert verdict["eligible"] is False
    assert any("must be one of" in reason for reason in verdict["reasons"])


def test_claim_eligibility_accepts_a_dataset_declared_held_out(tmp_path: Path) -> None:
    verdict = _split_verdict(tmp_path, protocol_split="test", split="validation")

    assert not any("split" in reason for reason in verdict["reasons"])
    assert verdict["dataset_protocol_split"] == "test"


def test_protocol_split_is_not_required_below_claim_eligible_tier(tmp_path: Path) -> None:
    """Exploratory and smoke runs are where an unlabelled dataset belongs; the
    held-out contract only gates empirical claims."""
    suite_path, fixture = _write_suite(tmp_path, tier="exploratory", protocol_split=None)
    suite = load_suite(suite_path)
    row = _ok_row(node_count=10, store_backend="json_memory")

    verdict = claim_eligibility({**suite, "required_baselines": ["naive_rag"]}, [row], str(fixture))

    assert not any("protocol_split" in reason for reason in verdict["reasons"])


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
