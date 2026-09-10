"""Machine-readable benchmark suite contracts and claim eligibility checks."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ragbench.core.config import load_config
from ragbench.core.io import read_json
from ragbench.core.measure import canonical_fingerprint
from ragbench.datasets.schema import PROTOCOL_SPLITS
from ragbench.evaluation.profiles import resolve_profile
from ragbench.indexing.artifacts import DEFAULT_FAISS_NODE_THRESHOLD
from ragbench.providers.env import env_int


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
        if bool(suite.get("pareto_improvement", False)):
            if "primary_metric" in suite or "minimum_effect" in suite:
                raise ValueError(
                    "suite.pareto_improvement and suite.primary_metric/minimum_effect are mutually exclusive"
                )
        else:
            # Decision-rule fields for improvement_supported (see
            # _improvement_supported): a single named decision metric with an
            # explicit minimum effect size, not "any one of several primary
            # metrics shows a CI that merely clears zero."
            if "primary_metric" not in suite:
                raise ValueError(
                    "Claim-eligible suite must declare suite.primary_metric (the single decision metric "
                    "for improvement_supported), or set suite.pareto_improvement: true"
                )
            if str(suite["primary_metric"]) not in suite["primary_metrics"]:
                raise ValueError("suite.primary_metric must be one of suite.primary_metrics")
            if "minimum_effect" not in suite:
                raise ValueError("Claim-eligible suite must declare suite.minimum_effect")
            if not isinstance(suite["minimum_effect"], int | float) or isinstance(suite["minimum_effect"], bool):
                raise ValueError("suite.minimum_effect must be a number")
            if suite["minimum_effect"] < 0:
                raise ValueError("suite.minimum_effect must be non-negative")
            non_inferiority = suite.get("non_inferiority", {})
            if not isinstance(non_inferiority, dict):
                raise ValueError("suite.non_inferiority must be a mapping of metric name -> allowed regression")
            for metric, allowed in non_inferiority.items():
                if not isinstance(allowed, int | float) or isinstance(allowed, bool) or allowed > 0:
                    raise ValueError(
                        f"suite.non_inferiority[{metric!r}] must be a number <= 0 "
                        "(the allowed regression, e.g. -0.01)"
                    )
        coverage = suite.get("coverage")
        if coverage is None:
            # Without a declared coverage block, validate_profile() falls back
            # to "at least one qualifying item" — so a suite could be called
            # claim-eligible while a single query in the whole dataset carried
            # a qrel. A per-slice claim needs the suite to state, up front,
            # what fraction of the dataset it actually expects to measure.
            raise ValueError(
                "Claim-eligible suite must declare a suite.coverage block stating the minimum "
                "measurable fraction/counts per slice (e.g. min_retrieval_coverage). Derive the "
                "numbers from the dataset's real composition, not from what the current run happens "
                "to pass — see docs/benchmark_protocol_v1.md"
            )
        _validate_coverage_block(coverage)
        # Resolving the profile here also validates suite.profile against
        # suite.mode at load time instead of at run time.
        _require_profile_coverage(coverage, resolve_profile(str(suite.get("profile", "auto")), str(suite["mode"])))
        # Optional provenance fields — not required (most suites tune config
        # informally), but recorded in claim_eligibility()'s verdict when
        # present so a report reader can see whether a claim's config was
        # frozen before evaluation, and against what dataset it was tuned.
        if "tuned_on_dataset" in suite and not isinstance(suite["tuned_on_dataset"], str):
            raise ValueError("suite.tuned_on_dataset must be a string (a dataset path or id)")
        if "config_frozen_at" in suite and not isinstance(suite["config_frozen_at"], str):
            raise ValueError("suite.config_frozen_at must be an ISO 8601 timestamp string")
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


# Which coverage thresholds a claim-eligible suite must declare, per resolved
# evaluation profile. Each key listed here is one ``validate_profile`` actually
# enforces for that profile — requiring a threshold that nothing consumes would
# just be paperwork, and leaving one out reopens the "at least one qualifying
# item" fallback for that slice.
#
# ``min_retrieval_coverage`` is required everywhere because every profile's
# report publishes recall/nDCG/MRR. ``min_unanswerable_questions`` is required
# for the RAG profiles because they publish abstention metrics, which are
# vacuous on a dataset with no unanswerable items. Declaring ``0`` (or ``0.0``)
# is a legitimate answer — "this suite deliberately does not measure that
# slice" — but it has to be said out loud rather than left to a default.
_REQUIRED_COVERAGE_KEYS: dict[str, tuple[str, ...]] = {
    "retrieval": ("min_retrieval_coverage",),
    "single_hop_rag": ("min_retrieval_coverage", "min_unanswerable_questions"),
    "multi_hop_rag": ("min_retrieval_coverage", "min_multi_hop_questions", "min_unanswerable_questions"),
    "citation_rag": ("min_retrieval_coverage", "min_citation_coverage", "min_unanswerable_questions"),
}


def _require_profile_coverage(coverage: dict[str, Any], profile: str) -> None:
    """Reject a coverage block that does not constrain the profile being run.

    Requiring merely "a mapping" was not enough: ``coverage: {}`` satisfied
    that while declaring nothing, so ``validate_profile`` fell straight back to
    its "at least one qualifying item" floor — the exact hole the coverage
    requirement was added to close. The same is true of a block that declares
    unrelated slices (e.g. only ``min_per_question_type`` for a retrieval
    suite), so the check is per-profile rather than merely non-empty.
    """
    if not coverage:
        raise ValueError(
            "suite.coverage is empty. An empty mapping declares nothing, so profile validation "
            "falls back to the 'at least one qualifying item' floor that a claim-eligible suite "
            "exists to rule out — see docs/benchmark_protocol_v1.md"
        )
    required = _REQUIRED_COVERAGE_KEYS[profile]
    missing = [key for key in required if coverage.get(key) is None]
    if missing:
        listed = ", ".join(f"suite.coverage.{key}" for key in missing)
        raise ValueError(
            f"Claim-eligible suite with evaluation profile '{profile}' must declare {listed}. "
            "Derive the values from the dataset's real composition; declare 0 to state explicitly "
            "that a slice is not measured — see docs/benchmark_protocol_v1.md"
        )


def _validate_coverage_block(coverage: Any) -> None:
    """Shape-check a suite's per-slice coverage minimums (see
    ``evaluation.profiles.validate_profile``).

    Required for claim-eligible suites, optional elsewhere. Individual keys
    stay optional — not every suite measures every slice — but each one that
    *is* present is validated so a typo becomes a load-time error instead of
    silently turning into a no-op requirement.
    """
    if not isinstance(coverage, dict):
        raise ValueError("suite.coverage must be a mapping")
    ratio_keys = ("min_retrieval_coverage", "min_citation_coverage")
    for key in ratio_keys:
        if key in coverage and (
            not isinstance(coverage[key], int | float) or isinstance(coverage[key], bool) or not 0 <= coverage[key] <= 1
        ):
            raise ValueError(f"suite.coverage.{key} must be a ratio between 0 and 1")
    count_keys = ("min_multi_hop_questions", "min_unanswerable_questions")
    for key in count_keys:
        value = coverage.get(key)
        invalid = value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0)
        if invalid:
            raise ValueError(f"suite.coverage.{key} must be a non-negative integer")
    per_type = coverage.get("min_per_question_type")
    if per_type is not None:
        if not isinstance(per_type, dict):
            raise ValueError("suite.coverage.min_per_question_type must be a mapping of question_type -> count")
        for question_type, minimum in per_type.items():
            if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
                raise ValueError(
                    f"suite.coverage.min_per_question_type[{question_type!r}] must be a non-negative integer"
                )


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
    dataset_metadata = manifest.get("metadata", {})
    dataset_metadata = dataset_metadata if isinstance(dataset_metadata, dict) else {}
    policy = dataset_metadata.get("corpus_policy")
    if policy != "full_upstream_corpus":
        reasons.append(f"dataset corpus policy is {policy or 'unspecified'}")
    upstream_split = dataset_metadata.get("split")
    protocol_split = dataset_metadata.get("protocol_split")
    if requested_tier == "claim_eligible":
        reason = _protocol_split_reason(protocol_split, upstream_split)
        if reason:
            reasons.append(reason)
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
    if requested_tier == "claim_eligible":
        # The per-row threshold check above only confirms each technique
        # individually respected the threshold — it does not catch a
        # baseline and candidate that both "correctly" auto-selected
        # *different* backends (e.g. one just under, one just over the
        # node-count threshold). Exact-search backends (json_memory,
        # faiss_local) rank identically (see JsonMemoryVectorStore's
        # docstring), so this never affects a quality claim — it only taints
        # latency/index-size (production) claims, where comparing two
        # techniques on different search backends is partly a claim about
        # which backend each happened to trigger, not about the techniques.
        backends_used = {
            str(row.get("store_backend"))
            for row in rows
            if row.get("status") == "ok" and row.get("store_backend") is not None
        }
        if len(backends_used) > 1:
            production_reasons.append(
                f"techniques ran on different vector store backends ({', '.join(sorted(backends_used))}); "
                "a production/latency claim requires every technique on the same backend"
            )
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
        # Two different facts that used to share one field. ``dataset_split``
        # is the upstream split name the data came from ("validation",
        # "test", ...); ``dataset_protocol_split`` is this protocol's own
        # dev/test role for it, which only a human can assert.
        "dataset_split": upstream_split or "unspecified",
        "dataset_protocol_split": protocol_split or "unspecified",
        "tuned_on_dataset": (suite or {}).get("tuned_on_dataset"),
        "config_frozen_at": (suite or {}).get("config_frozen_at"),
    }


def _protocol_split_reason(protocol_split: Any, upstream_split: Any) -> str | None:
    """Require an explicit held-out declaration before any empirical claim.

    ``metadata.split`` records the *upstream* split the data was taken from,
    which says nothing about whether this project has been tuning against it:
    the committed Vietnamese retrieval dataset is upstream "validation" — a
    dev split under another name — and the previous rule (reject only the
    literal string "dev") let it through. Nor can an absent label be read as
    "held out"; it means nobody has asserted anything.

    So the assertion has to be its own field, and it has to be positive: only
    ``protocol_split: "test"`` clears a claim. Whoever prepared the dataset is
    the only party who knows whether config, prompts, or chunk sizes were ever
    tuned against it — this contract makes them say so on the record.
    """
    if protocol_split == "test":
        return None
    tail = "claim-eligible runs require a held-out dataset labelled metadata.protocol_split: 'test'"
    if protocol_split is None:
        upstream = upstream_split or "unspecified"
        return (
            f"dataset does not declare metadata.protocol_split (upstream metadata.split is "
            f"'{upstream}', which does not imply a held-out split); {tail}"
        )
    if protocol_split not in PROTOCOL_SPLITS:
        allowed = ", ".join(repr(value) for value in PROTOCOL_SPLITS)
        return f"dataset metadata.protocol_split is '{protocol_split}'; must be one of {allowed} — {tail}"
    return f"dataset metadata.protocol_split is 'dev' (config is tuned against it); {tail}"


def _improvement_supported(suite: dict[str, Any] | None, comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Decide whether any candidate has a *supported* improvement over the reference baseline.

    Two decision rules, chosen by the suite (see ``load_suite``):

    - Default — a single ``primary_metric`` must show a paired CI95
      improvement of at least ``minimum_effect``, AND every metric listed in
      ``non_inferiority`` must not have regressed by more than its allowed
      amount. Previously this checked *any one* of possibly several
      ``primary_metrics`` with no effect-size floor and no guard against the
      others regressing — a candidate that raised Recall@10 while quietly
      cratering NDCG/MRR was reported "improvement_supported": true.
    - ``pareto_improvement: true`` — every metric in ``primary_metrics`` must
      not regress, and at least one must show a real (CI95-clears-zero)
      improvement. This is the appropriate rule only when there is genuinely
      no single decision metric to elevate above the others.
    """
    if not suite or suite.get("tier") != "claim_eligible":
        return {"supported": False, "reasons": ["suite is not claim_eligible"]}
    reference = str(suite.get("reference_baseline", ""))
    candidates = [item for item in comparisons if item.get("baseline") == reference]
    if not candidates:
        return {"supported": False, "reasons": ["no candidate comparison against reference baseline"]}
    if bool(suite.get("pareto_improvement", False)):
        return _pareto_improvement_supported(suite, candidates)
    return _primary_metric_improvement_supported(suite, candidates)


def _metric_improves_when_lower(metric: str) -> bool:
    return metric in {"latency_ms_avg", "latency_ms_p50", "latency_ms_p95", "estimated_cost_avg"}


def _ci_verdict(metric: str, interval: Any, threshold: float = 0.0) -> str:
    """Classify a paired CI95 as "improved" / "regressed" / "inconclusive" / "insufficient_data".

    ``threshold`` is a one-sided margin in the metric's *improving* direction
    — 0.0 asks "does the CI clear zero"; a positive ``minimum_effect`` asks
    "does the CI clear a real effect size, not just statistical noise around
    zero." Delta is assumed to be ``candidate - baseline`` (matching
    ``paired_bootstrap_delta``'s convention).
    """
    if not isinstance(interval, dict) or int(interval.get("paired_queries") or 0) < 2:
        return "insufficient_data"
    low, high = interval.get("ci95_low"), interval.get("ci95_high")
    if not isinstance(low, int | float) or not isinstance(high, int | float):
        return "insufficient_data"
    if _metric_improves_when_lower(metric):
        if high < -threshold:
            return "improved"
        if low > threshold:
            return "regressed"
        return "inconclusive"
    if low > threshold:
        return "improved"
    if high < -threshold:
        return "regressed"
    return "inconclusive"


def _primary_metric_improvement_supported(suite: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    primary_metric = str(suite["primary_metric"])
    minimum_effect = float(suite.get("minimum_effect", 0.0))
    non_inferiority = suite.get("non_inferiority") or {}
    supported: list[str] = []
    reasons: list[str] = []
    for comparison in candidates:
        candidate = str(comparison.get("candidate"))
        intervals = comparison.get("paired_ci95", {})
        intervals = intervals if isinstance(intervals, dict) else {}
        verdict = _ci_verdict(primary_metric, intervals.get(primary_metric), minimum_effect)
        if verdict != "improved":
            direction = "decrease" if _metric_improves_when_lower(primary_metric) else "increase"
            reasons.append(
                f"{candidate} {primary_metric} does not show at least a {minimum_effect} {direction} "
                f"({verdict})"
            )
            continue
        guards_ok = True
        for metric, allowed_regression in non_inferiority.items():
            guard_verdict = _ci_verdict(str(metric), intervals.get(str(metric)), 0.0)
            # Non-inferiority tolerates a bounded regression, not just "no
            # regression" — recompute against the allowed margin specifically
            # rather than reusing the zero-margin verdict above.
            allowed = abs(float(allowed_regression))
            margin_verdict = _ci_verdict(str(metric), intervals.get(str(metric)), -allowed)
            if margin_verdict == "regressed" or (margin_verdict == "insufficient_data" and guard_verdict != "improved"):
                guards_ok = False
                reasons.append(
                    f"{candidate} {metric} regressed beyond the allowed {allowed_regression} "
                    "non-inferiority margin (or lacks a usable CI)"
                )
        if guards_ok:
            supported.append(f"{candidate}:{primary_metric}")
    return {"supported": bool(supported), "reasons": reasons, "supported_metrics": supported}


def _pareto_improvement_supported(suite: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    primary_metrics = [str(metric) for metric in suite.get("primary_metrics", [])]
    supported: list[str] = []
    reasons: list[str] = []
    for comparison in candidates:
        candidate = str(comparison.get("candidate"))
        intervals = comparison.get("paired_ci95", {})
        intervals = intervals if isinstance(intervals, dict) else {}
        any_improved = False
        candidate_reasons: list[str] = []
        for metric in primary_metrics:
            verdict = _ci_verdict(metric, intervals.get(metric))
            if verdict == "improved":
                any_improved = True
            elif verdict == "regressed":
                candidate_reasons.append(f"{metric} regressed")
            elif verdict == "insufficient_data":
                candidate_reasons.append(f"{metric} lacks a usable confidence interval")
        if any_improved and not candidate_reasons:
            supported.append(f"{candidate}:pareto")
        else:
            reasons.append(f"{candidate}: " + ("; ".join(candidate_reasons) or "no primary metric improved"))
    return {"supported": bool(supported), "reasons": reasons, "supported_metrics": supported}


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
