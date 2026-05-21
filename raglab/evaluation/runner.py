from __future__ import annotations

from raglab.core.io import read_jsonl, write_json
from raglab.core.pipeline import query
from raglab.core.schema import EvalItem
from raglab.evaluation.metrics import evaluate_predictions


def run_eval(config_path: str, artifact_path: str, dataset_path: str, output_path: str, top_k: int = 5) -> dict:
    items = [EvalItem.from_dict(row) for row in read_jsonl(dataset_path)]
    predictions = [query(config_path, artifact_path, item.question) for item in items]
    metrics = evaluate_predictions(items, predictions, k=top_k)
    report = {
        "metrics": metrics,
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
