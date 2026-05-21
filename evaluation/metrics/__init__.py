from __future__ import annotations

from raglab.core.schema import EvalItem, RAGAnswer


def evaluate_predictions(items: list[EvalItem], predictions: list[RAGAnswer], k: int = 5) -> dict:
    if len(items) != len(predictions):
        raise ValueError("items and predictions must have the same length")
    recall_values: list[float] = []
    hit_values: list[float] = []
    reciprocal_ranks: list[float] = []
    citation_values: list[float] = []
    latencies: list[float] = []
    context_tokens: list[int] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    estimated_costs: list[float] = []

    for item, prediction in zip(items, predictions, strict=True):
        contexts = prediction.contexts[:k]
        retrieved_chunks = [context.chunk_id for context in contexts]
        retrieved_docs = [context.doc_id for context in contexts]
        expected_chunks = set(item.expected_chunk_ids)
        expected_docs = set(item.expected_doc_ids)

        if expected_chunks:
            found = expected_chunks & set(retrieved_chunks)
            recall_values.append(len(found) / len(expected_chunks))
            hit_values.append(1.0 if found else 0.0)
            reciprocal_ranks.append(_rr(retrieved_chunks, expected_chunks))
        elif expected_docs:
            found_docs = expected_docs & set(retrieved_docs)
            recall_values.append(len(found_docs) / len(expected_docs))
            hit_values.append(1.0 if found_docs else 0.0)
            reciprocal_ranks.append(_rr(retrieved_docs, expected_docs))
        else:
            recall_values.append(0.0)
            hit_values.append(0.0)
            reciprocal_ranks.append(0.0)

        citation_values.append(_citation_score(item, prediction))
        latencies.append(float(prediction.metadata.get("latency_ms", 0.0)))
        context_tokens.append(int(prediction.metadata.get("context_token_count", 0)))
        usage = prediction.metadata.get("usage", {})
        if isinstance(usage, dict):
            prompt_tokens.append(int(usage.get("prompt_tokens", 0)))
            completion_tokens.append(int(usage.get("completion_tokens", 0)))
        cost = prediction.metadata.get("cost_estimate", {})
        if isinstance(cost, dict):
            estimated_costs.append(float(cost.get("amount", 0.0)))

    return {
        f"recall_at_{k}": _mean(recall_values),
        "hit_rate": _mean(hit_values),
        "mrr": _mean(reciprocal_ranks),
        "citation_accuracy": _mean(citation_values),
        "latency_ms_avg": _mean(latencies),
        "context_tokens_avg": _mean(context_tokens),
        "prompt_tokens_avg": _mean(prompt_tokens),
        "completion_tokens_avg": _mean(completion_tokens),
        "estimated_cost_avg": _mean(estimated_costs),
        "queries": len(items),
    }


def _rr(values: list[str], expected: set[str]) -> float:
    for index, value in enumerate(values, start=1):
        if value in expected:
            return 1.0 / index
    return 0.0


def _citation_score(item: EvalItem, prediction: RAGAnswer) -> float:
    if item.expected_citations:
        return 1.0 if set(item.expected_citations) & set(prediction.citations) else 0.0
    if item.expected_doc_ids:
        cited = " ".join(prediction.citations)
        return 1.0 if any(doc_id in cited for doc_id in item.expected_doc_ids) else 0.0
    return 0.0


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
