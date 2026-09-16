from __future__ import annotations

import hashlib
from typing import Any

from ragbench.datasets.adapters.common import answer_text, first_present, limit_rows, load_hf_dataset, rows_from_split
from ragbench.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "taidng/UIT-ViQuAD2.0"
# Pinned so this benchmark reproduces a year from now even if upstream
# reshapes or relabels the repo.
REVISION = "406f09a45cc106a8f7b7fd0c25078883fe58cb1f"


def prepare_uit_viquad(
    split: str | None = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    selected_split = split or "validation"
    rows = limit_rows(_load_uit_viquad_rows(selected_split), limit, seed)
    docs_by_id: dict[str, DocumentRecord] = {}
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []

    for index, row in enumerate(rows, start=1):
        context = str(first_present(row, ["context", "text"], ""))
        doc_id = _context_id(row, context)
        if context and doc_id not in docs_by_id:
            docs_by_id[doc_id] = DocumentRecord(
                doc_id=doc_id,
                title=str(row.get("title", doc_id)) or None,
                text=context,
                metadata={"dataset": "uit_viquad", "language": "vi", "domain": "wikipedia"},
            )
        query_id = str(first_present(row, ["id", "uit_id", "question_id"], f"uit_viquad_{index:06d}"))
        is_answerable = not bool(row.get("is_impossible", False))
        queries.append(
            QueryRecord(
                query_id=query_id,
                question=str(first_present(row, ["question"], "")),
                ground_truth_answer=answer_text(row.get("answers")),
                is_answerable=is_answerable,
                metadata={
                    "dataset": "uit_viquad",
                    "language": "vi",
                    "domain": "wikipedia",
                    "question_type": "unanswerable" if not is_answerable else "extractive",
                },
            )
        )
        if is_answerable and context:
            qrels.append(
                QrelRecord(
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=2,
                    evidence_span=answer_text(row.get("answers")),
                    metadata={"dataset": "uit_viquad"},
                )
            )

    return PreparedDataset(
        dataset_id="vi_mrc_abstention",
        documents=list(docs_by_id.values()),
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "source_revision": REVISION,
            "license": "check upstream Hugging Face dataset card",
            "adapter": "uit_viquad",
            "split": selected_split,
            "upstream_split": selected_split,
            "annotation_type": "human",
            "task": "extractive_qa",
            "supports_unanswerable": True,
            "language": "vi",
            "domain": "wikipedia",
        },
    )


def _load_uit_viquad_rows(split: str) -> list[dict[str, Any]]:
    return rows_from_split(load_hf_dataset(REPO_ID, revision=REVISION), split)


def _context_id(row: dict[str, Any], context: str) -> str:
    source_id = first_present(row, ["uit_id", "title"])
    digest = hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
    return f"uit_{source_id}_{digest}" if source_id else f"uit_context_{digest}"
