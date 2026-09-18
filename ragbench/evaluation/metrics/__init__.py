from __future__ import annotations

import math
from typing import Any

from ragbench.core.schema import EvalItem, RAGAnswer
from ragbench.evaluation.metrics.qa import exact_match, token_f1


def evaluate_prediction_rows(
    items: list[EvalItem], predictions: list[RAGAnswer], k: int = 5, include_citation_metrics: bool = True
) -> list[dict[str, Any]]:
    """Return query-level measurements so aggregate claims remain auditable.

    Retrieval scores are intentionally calculated per query first. Benchmark
    comparisons can therefore use paired bootstrap confidence intervals instead
    of presenting a fragile difference between two rounded means.
    """
    if len(items) != len(predictions):
        raise ValueError("items and predictions must have the same length")
    rows: list[dict[str, Any]] = []
    for item, prediction in zip(items, predictions, strict=True):
        answerable = bool(item.metadata.get("is_answerable", True))
        contexts = prediction.contexts[:k]
        retrieved_chunks = [context.chunk_id for context in contexts]
        retrieved_docs = [context.doc_id for context in contexts]
        expected_chunks = set(item.expected_chunk_ids)
        expected_docs = set(item.expected_doc_ids)
        expected = expected_chunks or expected_docs
        retrieved = retrieved_chunks if expected_chunks else _unique_ranked(retrieved_docs)
        relevance = _relevance(item)
        retrieval_evaluated = bool(expected)
        found = expected & set(retrieved)
        row: dict[str, Any] = {
            "question_id": item.question_id,
            "question_type": item.metadata.get("question_type", "unspecified"),
            "is_answerable": answerable,
            "retrieval_evaluated": retrieval_evaluated,
            "recall": len(found) / len(expected) if expected else None,
            "hit_rate": 1.0 if found else (0.0 if expected else None),
            "mrr": _rr(retrieved, expected) if expected else None,
            "ndcg": _ndcg(retrieved, relevance, k) if expected else None,
            "map": _average_precision(retrieved, expected, k) if expected else None,
            "context_precision": len(found) / len(contexts) if contexts and expected else (0.0 if expected else None),
            "evidence_complete": 1.0 if expected and found == expected else (0.0 if expected else None),
            "evidence_partial": 1.0 if expected and found and found != expected else (0.0 if expected else None),
            "evidence_zero": 1.0 if expected and not found else (0.0 if expected else None),
            "abstention_correct": 1.0 if prediction.abstained != answerable else 0.0,
            "latency_ms": float(prediction.metadata.get("latency_ms", 0.0)),
            "context_tokens": int(prediction.metadata.get("context_token_count", 0)),
            "estimated_cost": _cost(prediction),
        }
        _attach_raw_retrieval_scores(row, prediction, expected, expected_chunks, relevance, k)
        if item.ground_truth_answer is not None:
            # Only extractive-QA style items carry a ground_truth_answer;
            # unanswerable questions leave it None, so this naturally scopes
            # EM/F1 to the same population the LLM judge's correctness score
            # covers, giving a deterministic primary signal alongside it.
            row["exact_match"] = exact_match(prediction.answer, item.ground_truth_answer)
            row["token_f1"] = token_f1(prediction.answer, item.ground_truth_answer)
        if include_citation_metrics and answerable and item.expected_citations:
            predicted_doc_ids = {citation.doc_id for citation in prediction.citations}
            precision, recall, f1 = _citation_scores(set(item.expected_citations), predicted_doc_ids)
            # Existing citation metrics are document identity matches; do not
            # imply they prove that a claim is entailed by the cited span.
            row.update(
                {
                    "citation_document_precision": precision,
                    "citation_document_recall": recall,
                    "citation_document_f1": f1,
                }
            )
        judge = prediction.metadata.get("judge")
        if isinstance(judge, dict):
            _attach_judge_scores(row, judge)
        rows.append(row)
    return rows


def _attach_raw_retrieval_scores(
    row: dict[str, Any],
    prediction: RAGAnswer,
    expected: set[str],
    expected_chunks: set[str],
    relevance: dict[str, int],
    k: int,
) -> None:
    """Score the retriever's own ranking, not what survived context building.

    ``recall``/``mrr``/``ndcg``/``map`` above are computed on
    ``prediction.contexts`` — the result of ``CitationContextBuilder``, which
    stops adding results once ``max_context_tokens`` is spent. A technique
    whose results are individually large (e.g. ``parent_child`` returning a
    full parent section per hit) can fit far fewer than ``k`` of them, making
    its "recall@k" look worse for a reason that has nothing to do with how
    well it ranked candidates. ``retrieved_chunk_ids``/``retrieved_doc_ids``
    (set by ``build_query_metadata``) capture the ranking before that
    truncation, so ``raw_recall`` etc. isolate retrieval-ranking quality from
    the context budget's effect on it. Absent on older checkpoints/fixtures
    written before this field existed — silently skipped rather than raising,
    like the judge/citation metrics above.
    """
    if "retrieved_chunk_ids" not in prediction.metadata:
        return
    # `run_eval()` runs one query at `max(cutoffs)` depth and reuses that
    # single prediction for every cutoff (see `_ensure_evaluation_depth` /
    # `metrics_by_cutoff`), so the metadata ranking here can be deeper than
    # this call's `k` — must slice, exactly like `contexts = prediction.
    # contexts[:k]` above, or a smaller cutoff sees hits that only rank
    # within the larger one.
    raw_chunks = list(prediction.metadata.get("retrieved_chunk_ids") or [])[:k]
    raw_docs = list(prediction.metadata.get("retrieved_doc_ids") or [])[:k]
    raw_retrieved = raw_chunks if expected_chunks else _unique_ranked(raw_docs)
    raw_found = expected & set(raw_retrieved)
    row.update(
        {
            "raw_recall": len(raw_found) / len(expected) if expected else None,
            "raw_hit_rate": 1.0 if raw_found else (0.0 if expected else None),
            "raw_mrr": _rr(raw_retrieved, expected) if expected else None,
            "raw_ndcg": _ndcg(raw_retrieved, relevance, k) if expected else None,
            "raw_map": _average_precision(raw_retrieved, expected, k) if expected else None,
        }
    )


# Which sub-judge produces which score. LLMJudge issues two independent calls
# (correctness sees the ground truth; faithfulness sees only the retrieved
# evidence) — see evaluation.judge.LLMJudge.
_CORRECTNESS_METRICS = ("answer_correctness", "abstention_correctness")
_FAITHFULNESS_METRICS = ("faithfulness", "citation_support")


def _attach_judge_scores(row: dict[str, Any], judge: dict[str, Any]) -> None:
    """Keep each sub-judge's scores independent of the *other* one's failure.

    The judge makes two separate provider calls, so one can fail while the
    other returns a perfectly valid score. Gating all four fields on the
    blanket ``status`` (which is "not ok" if *either* half failed) threw away
    a usable answer_correctness whenever the faithfulness call happened to
    return malformed JSON — a provider hiccup on one prompt silently shrank
    the sample size of an unrelated metric.

    Older judge payloads carry only ``status``; they fall back to it for both
    halves, which reproduces the previous behaviour exactly for reports and
    fixtures written before the split.
    """
    blanket_status = str(judge.get("status", "ok"))
    correctness_status = str(judge.get("correctness_status", blanket_status))
    faithfulness_status = str(judge.get("faithfulness_status", blanket_status))
    row["judge_status"] = blanket_status
    row["judge_correctness_status"] = correctness_status
    row["judge_faithfulness_status"] = faithfulness_status
    usable = (
        _CORRECTNESS_METRICS if correctness_status == "ok" else (),
        _FAITHFULNESS_METRICS if faithfulness_status == "ok" else (),
    )
    for key in (*usable[0], *usable[1]):
        if key in judge:
            row[key] = float(judge[key])


def evaluate_predictions(
    items: list[EvalItem],
    predictions: list[RAGAnswer],
    k: int = 5,
    include_citation_metrics: bool = True,
    include_citation_accuracy: bool | None = None,
) -> dict[str, Any]:
    """Aggregate auditable retrieval, RAG, operational, and judge metrics."""
    if include_citation_accuracy is not None:
        include_citation_metrics = include_citation_accuracy
    rows = evaluate_prediction_rows(items, predictions, k=k, include_citation_metrics=include_citation_metrics)
    retrieval = [row for row in rows if row["retrieval_evaluated"]]
    metrics: dict[str, Any] = {
        "retrieval_label_level": retrieval_label_level(items),
        f"recall_at_{k}": _mean(row["recall"] for row in retrieval),
        "hit_rate": _mean(row["hit_rate"] for row in retrieval),
        "mrr": _mean(row["mrr"] for row in retrieval),
        f"ndcg_at_{k}": _mean(row["ndcg"] for row in retrieval),
        f"map_at_{k}": _mean(row["map"] for row in retrieval),
        f"context_precision_at_{k}": _mean(row["context_precision"] for row in retrieval),
        "evidence_complete_rate": _mean(row["evidence_complete"] for row in retrieval),
        "partial_evidence_rate": _mean(row["evidence_partial"] for row in retrieval),
        "zero_evidence_rate": _mean(row["evidence_zero"] for row in retrieval),
        "retrieval_queries_evaluated": len(retrieval),
        **_raw_retrieval_aggregate(retrieval, k),
        "answerable_queries": sum(1 for row in rows if row["is_answerable"]),
        "unanswerable_queries": sum(1 for row in rows if not row["is_answerable"]),
        "abstention_accuracy": _mean(row["abstention_correct"] for row in rows),
        "latency_ms_avg": _mean(row["latency_ms"] for row in rows),
        "latency_ms_p50": _percentile([row["latency_ms"] for row in rows], 50),
        "latency_ms_p95": _percentile([row["latency_ms"] for row in rows], 95),
        "context_tokens_avg": _mean(row["context_tokens"] for row in rows),
        "estimated_cost_avg": _mean(row["estimated_cost"] for row in rows),
        "queries": len(rows),
    }
    if include_citation_metrics:
        citation_rows = [row for row in rows if "citation_document_f1" in row]
        metrics.update(
            {
                "citation_document_precision": _mean(row["citation_document_precision"] for row in citation_rows),
                "citation_document_recall": _mean(row["citation_document_recall"] for row in citation_rows),
                "citation_document_f1": _mean(row["citation_document_f1"] for row in citation_rows),
                "citation_queries_evaluated": len(citation_rows),
            }
        )
        # Backward-compatible aliases, deprecated in reports/documentation.
        metrics["citation_precision"] = metrics["citation_document_precision"]
        metrics["citation_recall"] = metrics["citation_document_recall"]
        metrics["citation_f1"] = metrics["citation_document_f1"]
    qa_rows = [row for row in rows if "exact_match" in row]
    if qa_rows:
        metrics["exact_match"] = _mean(row["exact_match"] for row in qa_rows)
        metrics["token_f1"] = _mean(row["token_f1"] for row in qa_rows)
        metrics["qa_queries_evaluated"] = len(qa_rows)
    _add_judge_metrics(metrics, rows)
    return metrics


def _add_judge_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Report each sub-judge's yield separately, and average each metric over
    the rows whose *own* judge call succeeded.

    A single blanket ``judge_failure_rate`` cannot say whether the correctness
    judge or the faithfulness judge is the one drifting off-schema, and the
    blanket-gated means made a faithfulness outage look like missing
    correctness data. Both are reported per sub-judge; the blanket figures are
    kept as the "at least one half failed" summary they always were.
    """
    judge_attempts = [row for row in rows if "judge_status" in row]
    if not judge_attempts:
        return
    judged = [row for row in judge_attempts if row.get("judge_status") == "ok"]
    metrics["judge_queries_evaluated"] = len(judged)
    metrics["judge_failure_rate"] = round(1 - len(judged) / len(judge_attempts), 6)
    for label, status_key, keys in (
        ("correctness", "judge_correctness_status", _CORRECTNESS_METRICS),
        ("faithfulness", "judge_faithfulness_status", _FAITHFULNESS_METRICS),
    ):
        usable = [row for row in judge_attempts if row.get(status_key) == "ok"]
        metrics[f"{label}_judge_queries_evaluated"] = len(usable)
        metrics[f"{label}_judge_failure_rate"] = round(1 - len(usable) / len(judge_attempts), 6)
        for key in keys:
            values = [row[key] for row in usable if key in row]
            if values:
                metrics[key if key != "abstention_correctness" else "judge_abstention_correctness"] = _mean(values)


def _raw_retrieval_aggregate(retrieval: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Aggregate the raw-ranking metrics ``_attach_raw_retrieval_scores`` attached.

    Omitted entirely (not zero-filled) when no row carries them, e.g. reports
    built from checkpoints/fixtures predating this field — keeping an absent
    metric distinguishable from a genuinely measured 0.0.
    """
    scored = [row for row in retrieval if "raw_recall" in row]
    if not scored:
        return {}
    return {
        f"raw_recall_at_{k}": _mean(row["raw_recall"] for row in scored),
        "raw_hit_rate": _mean(row["raw_hit_rate"] for row in scored),
        "raw_mrr": _mean(row["raw_mrr"] for row in scored),
        f"raw_ndcg_at_{k}": _mean(row["raw_ndcg"] for row in scored),
        f"raw_map_at_{k}": _mean(row["raw_map"] for row in scored),
    }


def retrieval_label_level(items: list[EvalItem]) -> str:
    """Which identifier the qrels are expressed in: ``chunk``, ``doc``, or ``none``.

    Reported alongside the metrics because chunk labels only mean anything for
    a technique that chunks exactly the way the labels were built. A dataset
    labelled with one chunker's ids scores every differently-chunking technique
    at zero — and, since ``retrieval_evaluated`` stays true, does so without
    any other signal that the run was meaningless. Seeing ``"chunk"`` here next
    to a ``zero_evidence_rate`` of 1.0 identifies that failure immediately.
    """
    if any(item.expected_chunk_ids for item in items):
        return "chunk"
    return "doc" if any(item.expected_doc_ids for item in items) else "none"


def _relevance(item: EvalItem) -> dict[str, int]:
    raw = item.metadata.get("relevance_by_doc_id", {})
    if isinstance(raw, dict):
        parsed = {str(key): max(int(value), 0) for key, value in raw.items()}
        if parsed:
            return parsed
    return {key: 1 for key in (item.expected_chunk_ids or item.expected_doc_ids)}


def _rr(values: list[str], expected: set[str]) -> float:
    for index, value in enumerate(values, start=1):
        if value in expected:
            return 1.0 / index
    return 0.0


def _unique_ranked(values: list[str]) -> list[str]:
    """Document qrels must not earn multiple gains from sibling chunks."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _ndcg(values: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum((2 ** relevance.get(value, 0) - 1) / math.log2(index + 1) for index, value in enumerate(values[:k], 1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(index + 1) for index, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def _average_precision(values: list[str], expected: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for index, value in enumerate(values[:k], 1):
        if value in expected:
            hits += 1
            score += hits / index
    return score / min(len(expected), k) if expected else 0.0


def _citation_scores(expected: set[str], predicted: set[str]) -> tuple[float, float, float]:
    overlap = expected & predicted
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _cost(prediction: RAGAnswer) -> float:
    cost = prediction.metadata.get("cost_estimate", {})
    return float(cost.get("amount", 0.0)) if isinstance(cost, dict) else 0.0


def _mean(values: Any) -> float:
    values = [float(value) for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower, upper = int(position), math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 6)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)
