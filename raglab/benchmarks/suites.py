"""Machine-readable benchmark suite contracts and claim eligibility checks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from raglab.core.config import load_config
from raglab.core.io import read_json
from raglab.core.measure import canonical_fingerprint
from raglab.indexing.artifacts import DEFAULT_FAISS_NODE_THRESHOLD
from raglab.providers.env import env_int


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = load_config(path)
    required = {"id", "tier", "dataset", "mode", "top_k", "required_baselines"}
    missing = sorted(required - set(suite))
    if missing:
        raise ValueError(f"Benchmark suite missing fields: {', '.join(missing)}")
    if suite["tier"] not in {"smoke_only", "exploratory", "claim_eligible"}:
        raise ValueError("suite.tier must be smoke_only, exploratory, or claim_eligible")
    if not isinstance(suite["required_baselines"], list):
        raise ValueError("suite.required_baselines must be a list")
    if suite["tier"] == "claim_eligible":
        frozen_fields = {
            "reference_baseline",
            "cutoffs",
            "bootstrap_samples",
            "primary_metrics",
            "warmup_queries",
            "concurrency",
            "latency_sample_size",
        }
        missing_frozen = sorted(frozen_fields - set(suite))
        if missing_frozen:
            raise ValueError(f"Claim-eligible suite missing fields: {', '.join(missing_frozen)}")
        if str(suite["reference_baseline"]) not in suite["required_baselines"]:
            raise ValueError("suite.reference_baseline must be one of suite.required_baselines")
        if not isinstance(suite["primary_metrics"], list) or not suite["primary_metrics"]:
            raise ValueError("suite.primary_metrics must be a non-empty list")
        if not isinstance(suite["bootstrap_samples"], int) or suite["bootstrap_samples"] < 1:
            raise ValueError("suite.bootstrap_samples must be a positive integer")
        if not isinstance(suite["warmup_queries"], int) or suite["warmup_queries"] < 1:
            # A claim-eligible run's latency must exclude cold-start effects
            # (artifact load, first-call model init); at least one discarded
            # warm-up query is required to make that claim credible.
            raise ValueError("Claim-eligible suite.warmup_queries must be a positive integer")
        if not isinstance(suite["concurrency"], int) or suite["concurrency"] < 1:
            raise ValueError("Claim-eligible suite.concurrency must be a positive integer")
        if not isinstance(suite["latency_sample_size"], int) or suite["latency_sample_size"] < 0:
            raise ValueError("Claim-eligible suite.latency_sample_size must be a non-negative integer")
        if suite["concurrency"] > 1 and suite["latency_sample_size"] < 1:
            # With no sequential sample at all, every prediction's latency is
            # contended — there would be no trustworthy source left for the
            # report's latency_pass / headline latency_ms_p95 at all.
            raise ValueError(
                "Claim-eligible suite.latency_sample_size must be at least 1 when suite.concurrency > 1"
            )
    dataset = suite["dataset"]
    if not isinstance(dataset, dict) or not isinstance(dataset.get("fingerprint"), str):
        raise ValueError("suite.dataset.fingerprint must lock the prepared dataset revision")
    cutoffs = suite.get("cutoffs", [suite["top_k"]])
    if not isinstance(cutoffs, list) or not all(isinstance(value, int) and value > 0 for value in cutoffs):
        raise ValueError("suite.cutoffs must be a positive integer list")
    suite["cutoffs"] = sorted(set(cutoffs))
    fingerprint_payload = {key: value for key, value in suite.items() if key != "suite_fingerprint"}
    suite["suite_fingerprint"] = canonical_fingerprint(fingerprint_payload)
    return suite


def resolve_suite(
    suite: dict[str, Any],
    *,
    docs: str | None,
    qa: str | None,
    mode: str | None,
    top_k: int | None,
    warmup_queries: int | None = None,
    concurrency: int | None = None,
    latency_sample_size: int | None = None,
) -> dict[str, Any]:
    resolved = dict(suite)
    for key, supplied in (("docs", docs), ("qa", qa)):
        expected = suite["dataset"].get(key) if isinstance(suite.get("dataset"), dict) else None
        if supplied and expected and str(Path(supplied)) != str(Path(str(expected))):
            raise ValueError(f"Suite locks {key}={expected}; received {supplied}")
        resolved[key] = supplied or expected
    if not resolved.get("docs") or not resolved.get("qa"):
        raise ValueError("Suite dataset must define docs and qa paths")
    if mode is not None and mode != str(suite["mode"]):
        raise ValueError(f"Suite locks mode={suite['mode']}; received {mode}")
    if top_k is not None and top_k != int(suite["top_k"]):
        raise ValueError(f"Suite locks top_k={suite['top_k']}; received {top_k}")
    if "warmup_queries" in suite and warmup_queries is not None and warmup_queries != int(suite["warmup_queries"]):
        raise ValueError(f"Suite locks warmup_queries={suite['warmup_queries']}; received {warmup_queries}")
    if "concurrency" in suite and concurrency is not None and concurrency != int(suite["concurrency"]):
        raise ValueError(f"Suite locks concurrency={suite['concurrency']}; received {concurrency}")
    if (
        "latency_sample_size" in suite
        and latency_sample_size is not None
        and latency_sample_size != int(suite["latency_sample_size"])
    ):
        raise ValueError(
            f"Suite locks latency_sample_size={suite['latency_sample_size']}; received {latency_sample_size}"
        )
    manifest = dataset_manifest(str(resolved["qa"]))
    if manifest.get("fingerprint") != suite["dataset"]["fingerprint"]:
        raise ValueError(
            "Suite dataset fingerprint does not match the prepared dataset. Re-freeze the suite or restore data."
        )
    resolved["mode"] = suite["mode"]
    resolved["top_k"] = suite["top_k"]
    if "warmup_queries" in suite:
        resolved["warmup_queries"] = suite["warmup_queries"]
    if "concurrency" in suite:
        resolved["concurrency"] = suite["concurrency"]
    if "latency_sample_size" in suite:
        resolved["latency_sample_size"] = suite["latency_sample_size"]
    return resolved


def claim_eligibility(
    suite: dict[str, Any] | None,
    rows: list[dict[str, Any]],
    qa: str,
    comparisons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Separate protocol conformance from evidence that an improvement exists."""
    reasons: list[str] = []
    requested_tier = str((suite or {}).get("tier", "exploratory"))
    if requested_tier != "claim_eligible":
        reasons.append(f"suite tier is {requested_tier}")
    required = set((suite or {}).get("required_baselines", []))
    actual = {str(row.get("technique")) for row in rows if row.get("status") == "ok"}
    missing = sorted(required - actual)
    if missing:
        reasons.append(f"missing required baselines: {', '.join(missing)}")
    if any(row.get("status") != "ok" for row in rows):
        reasons.append("one or more techniques failed")
    if any("fallback" in str(row.get("effective_components", "")) for row in rows):
        reasons.append("a fallback component was used")
    if is_git_dirty():
        reasons.append("git worktree is dirty")
    manifest = dataset_manifest(qa)
    policy = manifest.get("metadata", {}).get("corpus_policy") if isinstance(manifest.get("metadata"), dict) else None
    if policy != "full_upstream_corpus":
        reasons.append(f"dataset corpus policy is {policy or 'unspecified'}")
    minimum = int((suite or {}).get("minimum_queries", 0))
    queries = int(manifest.get("queries", 0))
    if minimum and queries < minimum:
        reasons.append(f"dataset has {queries} queries; suite requires {minimum}")
    if requested_tier == "claim_eligible":
        # Authoritative, post-ingest check: preflight can only warn that faiss
        # isn't installed (it can't know the final node count before chunking
        # happens); this checks the real node count against the real backend
        # that actually ran, so an environment where faiss silently never
        # engaged cannot slip through as claim-eligible.
        threshold = env_int("RAGLAB_FAISS_NODE_THRESHOLD", DEFAULT_FAISS_NODE_THRESHOLD)
        for row in rows:
            if row.get("status") != "ok":
                continue
            node_count = row.get("node_count")
            if isinstance(node_count, int) and node_count >= threshold and row.get("store_backend") != "faiss_local":
                reasons.append(
                    f"{row.get('technique')} has {node_count} nodes (>= {threshold}) but ran on backend "
                    f"'{row.get('store_backend')}', not faiss_local — install faiss-cpu (pip install '.[vector]')"
                )
    cost_reasons = list(reasons)
    if any(row.get("cost_status") != "estimated" for row in rows if row.get("status") == "ok"):
        cost_reasons.append("cost pricing is unknown")
    production_reasons = list(cost_reasons)
    if any(not isinstance(row.get("latency_ms_p95"), int | float) for row in rows if row.get("status") == "ok"):
        production_reasons.append("p95 latency is missing")
    if any(not isinstance(row.get("index_size_bytes"), int) for row in rows if row.get("status") == "ok"):
        production_reasons.append("index size is missing")
    if any(not isinstance(row.get("index_time_ms"), int | float) for row in rows if row.get("status") == "ok"):
        production_reasons.append("index build time is missing")
    protocol_eligible = not reasons
    improvement = _improvement_supported(suite, comparisons or [])
    return {
        "tier": requested_tier,
        "eligible": protocol_eligible,
        "reasons": reasons,
        "protocol_eligible": protocol_eligible,
        # Backward-compatible alias: this establishes only protocol
        # conformance, not that any candidate actually improved.
        "quality_claim_eligible": protocol_eligible,
        "improvement_supported": improvement["supported"],
        "improvement_reasons": improvement["reasons"],
        "cost_claim_eligible": not cost_reasons,
        "production_claim_eligible": not production_reasons,
        "cost_reasons": cost_reasons,
        "production_reasons": production_reasons,
        "dataset_corpus_policy": policy or "unspecified",
    }


def _improvement_supported(suite: dict[str, Any] | None, comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    if not suite or suite.get("tier") != "claim_eligible":
        return {"supported": False, "reasons": ["suite is not claim_eligible"]}
    primary = [str(metric) for metric in suite.get("primary_metrics", [])]
    reference = str(suite.get("reference_baseline", ""))
    candidates = [item for item in comparisons if item.get("baseline") == reference]
    if not candidates:
        return {"supported": False, "reasons": ["no candidate comparison against reference baseline"]}
    supported: list[str] = []
    reasons: list[str] = []
    for comparison in candidates:
        candidate = str(comparison.get("candidate"))
        intervals = comparison.get("paired_ci95", {})
        for metric in primary:
            interval = intervals.get(metric) if isinstance(intervals, dict) else None
            if not isinstance(interval, dict) or int(interval.get("paired_queries") or 0) < 2:
                reasons.append(f"{candidate} {metric} lacks enough paired observations")
                continue
            low, high = interval.get("ci95_low"), interval.get("ci95_high")
            if not isinstance(low, int | float) or not isinstance(high, int | float):
                reasons.append(f"{candidate} {metric} has no confidence interval")
            elif _metric_improves_when_lower(metric):
                if high < 0:
                    supported.append(f"{candidate}:{metric}")
                else:
                    reasons.append(f"{candidate} {metric} CI95 does not show a decrease")
            elif low > 0:
                supported.append(f"{candidate}:{metric}")
            else:
                reasons.append(f"{candidate} {metric} CI95 does not show an increase")
    return {"supported": bool(supported), "reasons": reasons, "supported_metrics": supported}


def _metric_improves_when_lower(metric: str) -> bool:
    return metric in {"latency_ms_avg", "latency_ms_p50", "latency_ms_p95", "estimated_cost_avg"}


def dataset_manifest(qa: str) -> dict[str, Any]:
    path = Path(qa)
    manifest = (path if path.is_dir() else path.parent) / "manifest.json"
    return read_json(manifest) if manifest.exists() else {}


def is_git_dirty() -> bool:
    try:
        command = ["git", "status", "--porcelain"]
        completed = subprocess.run(command, capture_output=True, check=True, text=True, timeout=2)
        return bool(completed.stdout)
    except (OSError, subprocess.SubprocessError):
        return True
