from __future__ import annotations

from raglab.datasets.adapters.common import (
    as_list,
    first_present,
    flatten_text,
    limit_rows,
    load_hf_dataset,
    rows_from_split,
)
from raglab.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "vimqa/vimqa"


def prepare_vimqa(
    split: str | None = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    rows = limit_rows(rows_from_split(load_hf_dataset(REPO_ID), split), limit, seed)
    documents_by_id: dict[str, DocumentRecord] = {}
    queries: list[QueryRecord] = []
    qrels: list[QrelRecord] = []

    for index, row in enumerate(rows, start=1):
        query_id = str(first_present(row, ["_id", "id", "question_id"], f"vimqa_{index:06d}"))
        queries.append(
            QueryRecord(
                query_id=query_id,
                question=str(first_present(row, ["question"], "")),
                ground_truth_answer=str(first_present(row, ["answer"], "")) or None,
                metadata={"dataset": "vimqa", "language": "vi", "domain": "wikipedia", "question_type": "multi-hop"},
            )
        )
        support_titles = {
            str(item[0]) for item in as_list(row.get("supporting_facts")) if isinstance(item, list) and item
        }
        for context in as_list(row.get("context")):
            if not isinstance(context, list) or len(context) < 2:
                continue
            title = str(context[0])
            sentences = as_list(context[1])
            doc_id = f"vimqa_{title}"
            text = flatten_text(sentences)
            documents_by_id.setdefault(
                doc_id,
                DocumentRecord(
                    doc_id=doc_id,
                    title=title,
                    text=text,
                    metadata={"dataset": "vimqa", "language": "vi", "domain": "wikipedia"},
                ),
            )
            if not support_titles or title in support_titles:
                qrels.append(
                    QrelRecord(
                        query_id=query_id,
                        doc_id=doc_id,
                        relevance=2 if title in support_titles else 1,
                        evidence_span=_supporting_sentences(row, title, sentences),
                        metadata={"dataset": "vimqa"},
                    )
                )

    return PreparedDataset(
        dataset_id="vi_multihop_reasoning",
        documents=list(documents_by_id.values()),
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "license": "requires upstream user agreement for full dataset",
            "adapter": "vimqa",
        },
    )


def _supporting_sentences(row: dict, title: str, sentences: list) -> str | None:
    selected = []
    for item in as_list(row.get("supporting_facts")):
        if isinstance(item, list) and len(item) >= 2 and str(item[0]) == title:
            index = int(item[1])
            if 0 <= index < len(sentences):
                selected.append(str(sentences[index]))
    return " ".join(selected) if selected else None
