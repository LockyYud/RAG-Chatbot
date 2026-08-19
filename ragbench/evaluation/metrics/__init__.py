from __future__ import annotations

import math
from typing import Any

from ragbench.core.schema import EvalItem, RAGAnswer


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
            row["judge_status"] = str(judge.get("status", "ok"))
            if row["judge_status"] == "ok":
                for key in ("answer_correctness", "faithfulness", "citation_support", "abstention_correctness"):
                    if key in judge:
                        row[key] = float(judge[key])
        rows.append(row)
    return rows


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
    judged = [row for row in rows if row.get("judge_status") == "ok"]
    judge_attempts = [row for row in rows if "judge_status" in row]
    if judge_attempts:
        metrics["judge_queries_evaluated"] = len(judged)
        metrics["judge_failure_rate"] = round(1 - len(judged) / len(judge_attempts), 6)
    if judged:
        for key in ("answer_correctness", "faithfulness", "citation_support", "abstention_correctness"):
            values = [row[key] for row in judged if key in row]
            if values:
                metrics[key if key != "abstention_correctness" else "judge_abstention_correctness"] = _mean(values)
    return metrics


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
    dcg = sum((2**relevance.get(value, 0) - 1) / math.log2(index + 1) for index, value in enumerate(values[:k], 1))
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
