"""Audit a synthetic benchmark against a human-labelled golden set.

The question this answers is the only one that matters about a model-labelled
benchmark: *does it rank techniques the way a human-labelled set does?* If it
does, it is usable for choosing between techniques even though its absolute
numbers mean nothing. If it does not, the tidy results table it produces is
decoration.

Two independent measurements, deliberately kept separate:

* **Rank agreement** — Kendall tau-b between the technique ordering on the
  synthetic set and on the golden set, per metric, with a bootstrap interval.
  Resampling is done *independently* on each side because the two sets contain
  different questions; there is no pairing to preserve.
* **Label precision** — a human reviews a sample of (question, labelled
  document) pairs. Nothing here can be automated: a model checking model-made
  labels reproduces the error it is meant to detect.

Audit consumes eval reports rather than re-running anything, so it is cheap
and stays honest about what was actually measured.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragbench.core.io import read_json, read_jsonl, write_json, write_jsonl

DEFAULT_METRICS = ("ndcg_at_10", "recall_at_10", "mrr")


# ─── Rank correlation ────────────────────────────────────────────────────────


def kendall_tau_b(left: list[float], right: list[float]) -> float | None:
    """Kendall tau-b between two equal-length score vectors, tie-corrected.

    Returns ``None`` when fewer than two comparable pairs exist, rather than a
    misleading 0.0 — with one or two techniques there is no ordering to agree
    about.
    """
    if len(left) != len(right):
        raise ValueError("kendall_tau_b needs two equal-length sequences")
    size = len(left)
    if size < 2:
        return None
    total_pairs = size * (size - 1) // 2
    concordant = discordant = left_ties = right_ties = 0
    for i in range(size):
        for j in range(i + 1, size):
            left_delta = left[i] - left[j]
            right_delta = right[i] - right[j]
            if left_delta == 0:
                left_ties += 1
            if right_delta == 0:
                right_ties += 1
            product = left_delta * right_delta
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
    # tau-b: (C - D) / sqrt((n0 - n1)(n0 - n2)), where n1/n2 are the pairs tied
    # on each side. A side with every pair tied zeroes the denominator, and the
    # correlation is then undefined rather than zero — reporting 0.0 there
    # would read as "measured no agreement" when nothing was measurable.
    denominator = math.sqrt((total_pairs - left_ties) * (total_pairs - right_ties))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def row_metric_key(metric: str) -> str:
    """Map an aggregate metric name (``ndcg_at_10``) to its per-query row key (``ndcg``)."""
    head, separator, tail = metric.rpartition("_at_")
    if separator and tail.isdigit():
        return head
    return metric


def _mean_by_technique(
    rows_by_technique: dict[str, list[dict[str, Any]]], key: str, question_ids: list[str] | None = None
) -> dict[str, float]:
    means: dict[str, float] = {}
    for technique, rows in rows_by_technique.items():
        if question_ids is None:
            values = [row[key] for row in rows if isinstance(row.get(key), int | float)]
        else:
            by_id = {str(row.get("question_id")): row for row in rows}
            values = [
                by_id[qid][key] for qid in question_ids if qid in by_id and isinstance(by_id[qid].get(key), int | float)
            ]
        means[technique] = sum(float(value) for value in values) / len(values) if values else 0.0
    return means


def rank_agreement(
    synthetic_rows: dict[str, list[dict[str, Any]]],
    golden_rows: dict[str, list[dict[str, Any]]],
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    seed: int = 42,
    samples: int = 1000,
) -> dict[str, Any]:
    """Per-metric Kendall tau-b with a bootstrap 95% interval."""
    techniques = sorted(set(synthetic_rows) & set(golden_rows))
    if len(techniques) < 2:
        raise ValueError(
            f"Rank agreement needs at least two techniques present in both runs; got {techniques or 'none'}"
        )
    synthetic_ids = sorted({str(row.get("question_id")) for rows in synthetic_rows.values() for row in rows})
    golden_ids = sorted({str(row.get("question_id")) for rows in golden_rows.values() for row in rows})
    rng = random.Random(seed)

    per_metric: dict[str, Any] = {}
    for metric in metrics:
        key = row_metric_key(metric)
        synthetic_means = _mean_by_technique(synthetic_rows, key)
        golden_means = _mean_by_technique(golden_rows, key)
        point = kendall_tau_b(
            [synthetic_means[name] for name in techniques], [golden_means[name] for name in techniques]
        )
        replicates: list[float] = []
        for _ in range(samples):
            synthetic_sample = [rng.choice(synthetic_ids) for _ in synthetic_ids]
            golden_sample = [rng.choice(golden_ids) for _ in golden_ids]
            tau = kendall_tau_b(
                [_mean_by_technique(synthetic_rows, key, synthetic_sample)[name] for name in techniques],
                [_mean_by_technique(golden_rows, key, golden_sample)[name] for name in techniques],
            )
            if tau is not None:
                replicates.append(tau)
        per_metric[metric] = {
            "kendall_tau": round(point, 6) if point is not None else None,
            "ci95_low": _percentile(replicates, 2.5),
            "ci95_high": _percentile(replicates, 97.5),
            "bootstrap_samples": len(replicates),
            "synthetic_means": {name: round(synthetic_means[name], 6) for name in techniques},
            "golden_means": {name: round(golden_means[name], 6) for name in techniques},
        }
    return {"techniques": techniques, "per_metric": per_metric}


def overall_agreement(per_metric: dict[str, Any]) -> dict[str, Any] | None:
    """Summarise the per-metric taus as the weakest one.

    The minimum, not the mean: a benchmark that orders techniques correctly on
    nDCG while disagreeing on recall has not earned a single trustworthy
    headline, and averaging would hide exactly that.
    """
    entries = [value for value in per_metric.values() if value.get("kendall_tau") is not None]
    if not entries:
        return None
    weakest = min(entries, key=lambda value: value["kendall_tau"])
    return {
        "kendall_tau": weakest["kendall_tau"],
        "ci95_low": weakest.get("ci95_low"),
        "ci95_high": weakest.get("ci95_high"),
        "basis": "weakest metric",
    }


# ─── Label review ────────────────────────────────────────────────────────────


def emit_label_sample(qa_rows: list[dict[str, Any]], *, sample_size: int, seed: int = 42) -> list[dict[str, Any]]:
    """Build a review file: one row per (question, labelled document) pair.

    Reviewed at pair level rather than question level because a question whose
    source document is right but whose three pooled documents are wrong is a
    label problem the question-level view cannot see.
    """
    pairs: list[dict[str, Any]] = []
    for row in qa_rows:
        for doc_id in row.get("expected_doc_ids", []) or []:
            pairs.append(
                {
                    "question_id": row.get("question_id"),
                    "question": row.get("question"),
                    "doc_id": doc_id,
                    "grade": (row.get("metadata", {}).get("relevance_by_doc_id", {}) or {}).get(doc_id),
                    "is_source": doc_id in (row.get("expected_citations") or []),
                    "correct": None,
                    "reviewer": "",
                    "note": "",
                }
            )
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:sample_size]


def label_precision(review_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a filled review file at pair and question level.

    Rows whose ``correct`` is still ``None`` are unreviewed and excluded; a
    file nobody filled in yields zero reviewed pairs rather than a perfect
    score.
    """
    reviewed = [row for row in review_rows if isinstance(row.get("correct"), bool)]
    pair_correct = sum(1 for row in reviewed if row["correct"])
    by_question: dict[str, list[bool]] = {}
    for row in reviewed:
        by_question.setdefault(str(row.get("question_id")), []).append(bool(row["correct"]))
    question_correct = sum(1 for verdicts in by_question.values() if all(verdicts))
    return {
        "sampled_pairs": len(review_rows),
        "reviewed_pairs": len(reviewed),
        "correct_pairs": pair_correct,
        "pair_precision": round(pair_correct / len(reviewed), 6) if reviewed else None,
        "reviewed_questions": len(by_question),
        "question_precision": round(question_correct / len(by_question), 6) if by_question else None,
    }


# ─── Orchestration ───────────────────────────────────────────────────────────


def load_report_rows(runs: Mapping[str, str | Path]) -> dict[str, list[dict[str, Any]]]:
    """Read ``query_metrics`` out of one eval report per technique."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for technique, path in runs.items():
        report = read_json(path)
        query_metrics = report.get("query_metrics")
        if not isinstance(query_metrics, list) or not query_metrics:
            raise ValueError(f"Report for '{technique}' has no query_metrics: {path}")
        rows[technique] = query_metrics
    return rows


def run_audit(
    *,
    synthetic_runs: Mapping[str, str | Path],
    golden_runs: Mapping[str, str | Path],
    output_dir: str | Path,
    synthetic_qa_path: str | Path | None = None,
    label_review_path: str | Path | None = None,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    seed: int = 42,
    samples: int = 1000,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    synthetic_rows = load_report_rows(synthetic_runs)
    golden_rows = load_report_rows(golden_runs)
    agreement = rank_agreement(synthetic_rows, golden_rows, metrics=metrics, seed=seed, samples=samples)
    golden_question_ids = {str(row.get("question_id")) for rows in golden_rows.values() for row in rows}

    precision: dict[str, Any] | None = None
    if label_review_path is not None:
        precision = label_precision(read_jsonl(label_review_path))

    audit = {
        "audit_path": str(target / "audit.json"),
        "golden_queries": len(golden_question_ids),
        "techniques": agreement["techniques"],
        "rank_agreement": agreement["per_metric"],
        "overall_kendall_tau": overall_agreement(agreement["per_metric"]),
        "label_precision": precision,
        "label_review_path": str(label_review_path) if label_review_path else None,
        "synthetic_qa_path": str(synthetic_qa_path) if synthetic_qa_path else None,
        "seed": seed,
        "bootstrap_samples": samples,
    }
    write_json(target / "audit.json", audit)
    if synthetic_qa_path is not None:
        # Record the audit on the dataset itself, so any later eval of that
        # dataset picks it up without the caller having to remember to pass it.
        _record_audit_on_dataset(synthetic_qa_path, audit["audit_path"])
    return audit


def _record_audit_on_dataset(qa_path: str | Path, audit_path: str) -> None:
    source = Path(qa_path)
    manifest_path = (source.parent if source.is_file() else source) / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = read_json(manifest_path)
    except (OSError, ValueError):
        return
    if not isinstance(manifest, dict):
        return
    manifest["audit_path"] = audit_path
    write_json(manifest_path, manifest)


def load_recorded_audit(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Read the audit a dataset's manifest points at, if any.

    Missing or unreadable means "not audited", never an error: an un-audited
    dataset is a legitimate state that the trust verdict already reports.
    """
    audit_path = manifest.get("audit_path")
    if not isinstance(audit_path, str) or not Path(audit_path).exists():
        return None
    try:
        audit = read_json(audit_path)
    except (OSError, ValueError):
        return None
    return audit if isinstance(audit, dict) else None


def write_label_sample(
    qa_path: str | Path, output_path: str | Path, *, sample_size: int, seed: int = 42
) -> dict[str, Any]:
    rows = emit_label_sample(read_jsonl(qa_path), sample_size=sample_size, seed=seed)
    write_jsonl(output_path, rows)
    return {
        "output": str(output_path),
        "pairs": len(rows),
        "instructions": (
            'Set "correct" to true or false on every row, fill "reviewer", then pass this file back '
            "via --label-review. Rows left at null are counted as unreviewed."
        ),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return round(ordered[lower], 6)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)
