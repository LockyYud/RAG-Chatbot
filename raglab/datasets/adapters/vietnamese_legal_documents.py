from __future__ import annotations

from typing import Any

from raglab.datasets.adapters.common import as_list, first_present, limit_rows, load_hf_dataset, rows_from_split
from raglab.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "YuITC/Vietnamese-Legal-Documents"


def prepare_vietnamese_legal_documents(
    split: str | None = "test",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    dataset = load_hf_dataset(REPO_ID)
    corpus_rows = _load_corpus(dataset)
    query_rows = limit_rows(rows_from_split(dataset, split), limit, seed)
    relevant_doc_ids = {str(cid) for row in query_rows for cid in as_list(row.get("cid"))}
    corpus_by_id = {_legal_doc_id(row): row for row in corpus_rows}

    documents = [
        DocumentRecord(
            doc_id=doc_id,
            title=f"Vietnamese legal document {doc_id}",
            text=str(first_present(row, ["text", "context", "document"], "")),
            metadata={"dataset": "vietnamese_legal_documents", "language": "vi", "domain": "legal"},
        )
        for doc_id, row in corpus_by_id.items()
        if doc_id in relevant_doc_ids
    ]
    missing_doc_ids = relevant_doc_ids - {doc.doc_id for doc in documents}
    for row in query_rows:
        for cid, context in zip(as_list(row.get("cid")), as_list(row.get("context_list")), strict=False):
            doc_id = str(cid)
            if doc_id in missing_doc_ids:
                documents.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        title=f"Vietnamese legal document {doc_id}",
                        text=str(context),
                        metadata={
                            "dataset": "vietnamese_legal_documents",
                            "language": "vi",
                            "domain": "legal",
                            "source": "context_list_fallback",
                        },
                    )
                )
                missing_doc_ids.remove(doc_id)

    queries = [
        QueryRecord(
            query_id=str(first_present(row, ["qid", "query_id", "id"])),
            question=str(first_present(row, ["question", "query"], "")),
            metadata={"dataset": "vietnamese_legal_documents", "language": "vi", "domain": "legal"},
        )
        for row in query_rows
    ]
    qrels = []
    for row in query_rows:
        query_id = str(first_present(row, ["qid", "query_id", "id"]))
        for cid, context in zip(as_list(row.get("cid")), as_list(row.get("context_list")), strict=False):
            qrels.append(
                QrelRecord(
                    query_id=query_id,
                    doc_id=str(cid),
                    relevance=2,
                    evidence_span=str(context)[:500] if context is not None else None,
                    metadata={"dataset": "vietnamese_legal_documents"},
                )
            )
    return PreparedDataset(
        dataset_id="vi_legal_retrieval",
        documents=documents,
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "license": "MIT according to Hugging Face dataset card",
            "adapter": "vietnamese_legal_documents",
        },
    )


def _load_corpus(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, dict):
        for split_name in ("corpus", "documents", "all"):
            if split_name in dataset:
                return [dict(row) for row in dataset[split_name]]
    try:
        return rows_from_split(load_hf_dataset(REPO_ID, "corpus"), "corpus")
    except Exception:  # pragma: no cover - depends on upstream HF packaging
        return []


def _legal_doc_id(row: dict[str, Any]) -> str:
    return str(first_present(row, ["cid", "doc_id", "id"]))
