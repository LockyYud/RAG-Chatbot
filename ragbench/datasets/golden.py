"""Validation for manually curated end-to-end RAG golden sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.core.io import read_jsonl

REQUIRED_FIELDS = frozenset({"question_id", "question", "reference_answer", "is_answerable", "required_claims"})


def validate_golden_dataset(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Golden dataset is empty")
    ids: set[str] = set()
    errors: list[str] = []
    answerable = 0
    evidence_spans = 0
    for index, row in enumerate(rows, 1):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"row {index}: missing {', '.join(sorted(missing))}")
            continue
        question_id = str(row["question_id"])
        if question_id in ids:
            errors.append(f"row {index}: duplicate question_id {question_id!r}")
        ids.add(question_id)
        if not isinstance(row["is_answerable"], bool):
            errors.append(f"row {index}: is_answerable must be boolean")
        claims = row["required_claims"]
        if not isinstance(claims, list) or not all(isinstance(value, str) and value for value in claims):
            errors.append(f"row {index}: required_claims must be a non-empty string list")
        spans = row.get("evidence_spans", [])
        if not isinstance(spans, list):
            errors.append(f"row {index}: evidence_spans must be a list")
        else:
            evidence_spans += len(spans)
            for span in spans:
                if not isinstance(span, dict) or not isinstance(span.get("doc_id"), str):
                    errors.append(f"row {index}: every evidence span requires doc_id")
        answerable += int(bool(row["is_answerable"]))
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "path": str(path),
        "queries": len(rows),
        "answerable_queries": answerable,
        "unanswerable_queries": len(rows) - answerable,
        "evidence_spans": evidence_spans,
    }
