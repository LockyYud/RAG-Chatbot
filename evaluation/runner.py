from __future__ import annotations

from evaluation.judge import create_judge
from evaluation.metrics import evaluate_predictions
from raglab.core.io import read_jsonl, write_json
from raglab.core.pipeline import query
from raglab.core.schema import EvalItem


def run_eval(
    config_path: str,
    artifact_path: str,
    dataset_path: str,
    output_path: str,
    top_k: int = 5,
    mode: str = "full_rag",
    judge_spec: dict | None = None,
) -> dict:
    items = [EvalItem.from_dict(row) for row in read_jsonl(dataset_path)]
    judge = create_judge(judge_spec)
    predictions = []
    for item in items:
        prediction = query(config_path, artifact_path, item.question, mode=mode)
        if judge is not None and mode == "full_rag":
            judge_result = judge.judge(item, prediction)
            prediction.metadata["judge"] = judge_result.to_dict()
        predictions.append(prediction)
    metrics = evaluate_predictions(items, predictions, k=top_k)
    failures = _failures(items, predictions, top_k)
    report = {
        "run_metadata": {
            "config_path": config_path,
            "artifact_path": artifact_path,
            "dataset_path": dataset_path,
            "mode": mode,
            "top_k": top_k,
            "judge_enabled": judge is not None,
        },
        "metrics": metrics,
        "cost_summary": _cost_summary(predictions),
        "failures": failures,
        "predictions": [
            {
                "question_id": item.question_id,
                "question": item.question,
                "answer": prediction.answer,
                "citations": prediction.citations,
                "contexts": [context.to_dict() for context in prediction.contexts],
                "metadata": prediction.metadata,
            }
            for item, prediction in zip(items, predictions, strict=True)
        ],
    }
    write_json(output_path, report)
    return report


def _cost_summary(predictions: list) -> dict:
    costs = []
    for prediction in predictions:
        cost = prediction.metadata.get("cost_estimate", {})
        if isinstance(cost, dict):
            costs.append(float(cost.get("amount", 0.0)))
        judge = prediction.metadata.get("judge", {})
        if isinstance(judge, dict):
            costs.append(float(judge.get("estimated_cost", 0.0)))
    return {
        "currency": "USD",
        "total_estimated_cost": round(sum(costs), 8),
        "avg_estimated_cost": round(sum(costs) / len(predictions), 8) if predictions else 0.0,
    }


def _failures(items: list[EvalItem], predictions: list, top_k: int) -> list[dict]:
    rows = []
    for item, prediction in zip(items, predictions, strict=True):
        contexts = prediction.contexts[:top_k]
        retrieved_docs = {context.doc_id for context in contexts}
        retrieved_chunks = {context.chunk_id for context in contexts}
        missing_docs = sorted(set(item.expected_doc_ids) - retrieved_docs)
        missing_chunks = sorted(set(item.expected_chunk_ids) - retrieved_chunks)
        citation_ok = not item.expected_citations or bool(set(item.expected_citations) & set(prediction.citations))
        judge = prediction.metadata.get("judge", {})
        answer_score = float(judge.get("answer_correctness", 1.0)) if isinstance(judge, dict) else 1.0
        if missing_docs or missing_chunks or not citation_ok or answer_score < 0.5:
            rows.append(
                {
                    "question_id": item.question_id,
                    "question": item.question,
                    "severity": _severity(missing_docs, missing_chunks, citation_ok, answer_score),
                    "missing_doc_ids": missing_docs,
                    "missing_chunk_ids": missing_chunks,
                    "citation_ok": citation_ok,
                    "answer_correctness": answer_score,
                }
            )
    return sorted(rows, key=lambda row: row["severity"], reverse=True)


def _severity(missing_docs: list[str], missing_chunks: list[str], citation_ok: bool, answer_score: float) -> int:
    score = 0
    if missing_docs:
        score += 3
    if missing_chunks:
        score += 3
    if not citation_ok:
        score += 2
    if answer_score < 0.5:
        score += 2
    return score
