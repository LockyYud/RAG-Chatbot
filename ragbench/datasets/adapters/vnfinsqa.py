from __future__ import annotations

import hashlib

from ragbench.datasets.adapters.common import as_list, first_present, limit_rows, load_hf_dataset, rows_from_split
from ragbench.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "duykhangh/VNFinsQA"


def prepare_vnfinsqa(
    split: str | None = "test",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    rows = limit_rows(rows_from_split(load_hf_dataset(REPO_ID), split), limit, seed)
    documents_by_id: dict[str, DocumentRecord] = {}
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []

    for index, row in enumerate(rows, start=1):
        query_id = str(first_present(row, ["question_id", "qid", "id"], f"vnfinsqa_{index:04d}"))
        question = str(first_present(row, ["question", "query"], ""))
        answer = first_present(row, ["answer", "ground_truth_answer", "response"])
        queries.append(
            QueryRecord(
                query_id=query_id,
                question=question,
                ground_truth_answer=str(answer) if answer is not None else None,
                metadata={"dataset": "vnfinsqa", "language": "vi", "domain": "finance"},
            )
        )
        contexts = as_list(first_present(row, ["contexts", "context", "documents", "evidence", "ground_truth_context"]))
        if not contexts and answer:
            contexts = [str(answer)]
        for context_index, context in enumerate(contexts, start=1):
            text = str(context)
            doc_id = _context_id(query_id, context_index, text)
            documents_by_id.setdefault(
                doc_id,
                DocumentRecord(
                    doc_id=doc_id,
                    title=str(first_present(row, ["ticker", "symbol", "company"], f"VNFinsQA evidence {doc_id}")),
                    text=text,
                    metadata={
                        "dataset": "vnfinsqa",
                        "language": "vi",
                        "domain": "finance",
                        "ticker": first_present(row, ["ticker", "symbol"]),
                    },
                ),
            )
            qrels.append(
                QrelRecord(
                    query_id=query_id,
                    doc_id=doc_id,
                    relevance=2,
                    evidence_span=text[:500],
                    metadata={"dataset": "vnfinsqa"},
                )
            )

    return PreparedDataset(
        dataset_id="vi_finance_qa",
        documents=list(documents_by_id.values()),
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "license": "check upstream Hugging Face dataset card",
            "adapter": "vnfinsqa",
        },
    )


def _context_id(query_id: str, context_index: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"vnfinsqa_{query_id}_{context_index}_{digest}"
