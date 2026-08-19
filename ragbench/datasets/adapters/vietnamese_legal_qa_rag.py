from __future__ import annotations

import hashlib

from ragbench.datasets.adapters.common import as_list, first_present, limit_rows, load_hf_dataset, rows_from_split
from ragbench.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "NamSyntax/Vietnamese-Legal-QA-RAG"


def prepare_vietnamese_legal_qa_rag(
    split: str | None = "train",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    rows = limit_rows(rows_from_split(load_hf_dataset(REPO_ID), split), limit, seed)
    documents_by_id: dict[str, DocumentRecord] = {}
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []

    for index, row in enumerate(rows, start=1):
        query_id = str(first_present(row, ["question_id", "id"], f"legal_rag_{index:04d}"))
        question_type = str(first_present(row, ["question_type", "type"], "factoid"))
        is_answerable = question_type != "unanswerable"
        queries.append(
            QueryRecord(
                query_id=query_id,
                question=str(first_present(row, ["question", "query"], "")),
                ground_truth_answer=str(first_present(row, ["ground_truth_answer", "answer"], "")),
                is_answerable=is_answerable,
                metadata={
                    "dataset": "vietnamese_legal_qa_rag",
                    "language": "vi",
                    "domain": "legal",
                    "question_type": question_type,
                },
            )
        )
        if not is_answerable:
            continue
        contexts = as_list(first_present(row, ["ground_truth_context", "contexts"]))
        for context_index, context in enumerate(contexts, start=1):
            text = str(context)
            doc_id = _context_id(query_id, context_index, text)
            documents_by_id.setdefault(
                doc_id,
                DocumentRecord(
                    doc_id=doc_id,
                    title=f"Vietnamese legal QA evidence {doc_id}",
                    text=text,
                    metadata={"dataset": "vietnamese_legal_qa_rag", "language": "vi", "domain": "legal"},
                ),
            )
            qrels.append(
                QrelRecord(
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=2,
                    evidence_span=text[:500],
                    metadata={"dataset": "vietnamese_legal_qa_rag"},
                )
            )

    return PreparedDataset(
        dataset_id="vi_legal_rag_small",
        documents=list(documents_by_id.values()),
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "license": "check upstream Hugging Face dataset card",
            "adapter": "vietnamese_legal_qa_rag",
        },
    )


def _context_id(query_id: str, context_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"legal_rag_{query_id}_{context_index}_{digest}"
