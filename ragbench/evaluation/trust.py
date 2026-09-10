"""Trust verdicts for benchmarks whose labels came from a model.

A synthetic benchmark can be perfectly usable for *ranking* techniques while
being worthless for absolute numbers, and the difference is not visible in the
results table — a bad label set and a good one both render as a tidy grid of
floats. This module turns the generation manifest (see
``ragbench.datasets.synthetic``) plus an optional audit (see
``ragbench.evaluation.audit``) into a verdict that a report has to carry.

The deliberate asymmetry: no verdict here ever licenses an absolute number.
The best available outcome is ``synthetic_trusted_for_ranking``, and the name
is the claim — "technique A beat technique B on this corpus", never
"recall@10 was 0.82".
"""

from __future__ import annotations

from typing import Any

#: Initial thresholds, chosen from judgement rather than data — there is no
#: corpus to calibrate against until the first real audit runs. Recalibrate
#: after three real corpora and record the history in
#: ``docs/synthetic_benchmark_v2.md`` §5.4.
TRUST_THRESHOLDS: dict[str, float] = {
    "min_kendall_tau": 0.5,
    "min_label_precision": 0.8,
    "max_answerability_reject_rate": 0.4,
    "warn_lexical_overlap_p90": 0.75,
}

UNTRUSTED = "synthetic_untrusted"
TRUSTED_FOR_RANKING = "synthetic_trusted_for_ranking"


def build_trust_block(
    manifest: dict[str, Any] | None,
    audit: dict[str, Any] | None = None,
    *,
    judge_model: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    """Return the ``trust`` block for a report, or ``None`` for a human-labelled set.

    ``manifest`` is the synthetic dataset's ``manifest.json``; ``None`` means
    the dataset was not generated and no trust block belongs in the report.
    """
    if manifest is None:
        return None
    limits = {**TRUST_THRESHOLDS, **(thresholds or {})}
    stages = manifest.get("stages", {}) if isinstance(manifest.get("stages"), dict) else {}
    models = manifest.get("models", {}) if isinstance(manifest.get("models"), dict) else {}

    generation = {
        "answerability_reject_rate": _number(stages.get("answerability_reject_rate")),
        "lexical_overlap_p90": _number(stages.get("lexical_overlap_p90")),
        "extra_relevant_mean": _number(stages.get("extra_relevant_mean")),
        "final_questions": int(stages.get("final", 0) or 0),
    }

    reasons: list[str] = []
    warnings: list[str] = []

    # Role collisions recorded at generation time, plus the one that can only
    # be known now: a judge that also wrote the questions.
    same_roles = list(manifest.get("same_model_roles", []) or [])
    if judge_model:
        for role in ("generator", "verifier"):
            if models.get(role) and models[role] == judge_model:
                label = f"{role}==judge"
                if label not in same_roles:
                    same_roles.append(label)
    if same_roles:
        reasons.append("circular_judge")

    if generation["answerability_reject_rate"] is not None and (
        generation["answerability_reject_rate"] > limits["max_answerability_reject_rate"]
    ):
        reasons.append("weak_generator")
    if generation["lexical_overlap_p90"] is not None and (
        generation["lexical_overlap_p90"] > limits["warn_lexical_overlap_p90"]
    ):
        warnings.append("lexical_bias")
    if list(manifest.get("pool_retrievers", []) or []) == ["bm25"]:
        # A pool built by one retriever family inherits that family's blind
        # spots, so passages only a dense retriever would surface never get
        # graded and stay unlabelled — the same false-negative problem pooling
        # exists to fix, just narrower.
        warnings.append("single_retriever_pool")

    block: dict[str, Any] = {
        "audited": audit is not None,
        "generation": generation,
        "same_model_roles": same_roles,
    }

    if audit is None:
        reasons.append("not_audited")
        block.update({"verdict": UNTRUSTED, "reasons": sorted(set(reasons)), "warnings": sorted(set(warnings))})
        return block

    block["audit_path"] = audit.get("audit_path")
    block["golden_queries"] = audit.get("golden_queries")
    block["rank_agreement"] = audit.get("rank_agreement", {})
    block["label_precision"] = audit.get("label_precision")

    overall = audit.get("overall_kendall_tau")
    if overall is None:
        reasons.append("rank_disagreement")
    else:
        tau = _number(overall.get("kendall_tau")) if isinstance(overall, dict) else _number(overall)
        ci_low = _number(overall.get("ci95_low")) if isinstance(overall, dict) else None
        if tau is None or tau < limits["min_kendall_tau"]:
            reasons.append("rank_disagreement")
        elif ci_low is not None and ci_low <= 0:
            # A tau above the bar whose interval still spans zero has not
            # separated agreement from chance.
            reasons.append("rank_disagreement")

    precision = audit.get("label_precision")
    if isinstance(precision, dict):
        pair_value = _number(precision.get("pair_precision"))
        if pair_value is None or pair_value < limits["min_label_precision"]:
            reasons.append("label_noise")
    else:
        reasons.append("label_noise")

    block["verdict"] = UNTRUSTED if reasons else TRUSTED_FOR_RANKING
    block["reasons"] = sorted(set(reasons))
    block["warnings"] = sorted(set(warnings))
    block["thresholds"] = limits
    return block


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)
