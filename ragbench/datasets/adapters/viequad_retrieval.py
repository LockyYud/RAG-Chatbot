from __future__ import annotations

from typing import Any

from ragbench.datasets.adapters.common import first_present, limit_rows, load_hf_dataset, rows_from_split
from ragbench.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "mteb/VieQuADRetrieval"


def prepare_viequad_retrieval(
    split: str | None = "validation",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    corpus_rows, query_rows, qrel_rows = _load_mteb_triplet(split)
    query_rows = limit_rows(query_rows, limit, seed)
    query_ids = {_row_id(row) for row in query_rows}
    qrel_rows = [row for row in qrel_rows if str(first_present(row, ["query-id", "query_id", "qid"])) in query_ids]
    # Keep the complete corpus even when selecting a smaller query fixture.
    # Restricting it to positive documents removes hard negatives and produces
    # retrieval scores that cannot be compared with the upstream MTEB task.
    source_corpus_size = len(corpus_rows)

    documents = [
        DocumentRecord(
            doc_id=_row_id(row),
            title=str(row.get("title", "")) or None,
            text=str(first_present(row, ["text", "document", "contents"], "")),
            metadata={"dataset": "viequad_retrieval", "language": "vi", "domain": "wikipedia"},
        )
        for row in corpus_rows
    ]
    queries = [
        QueryRecord(
            query_id=_row_id(row),
            question=str(first_present(row, ["text", "query", "question"], "")),
            metadata={"dataset": "viequad_retrieval", "language": "vi", "domain": "wikipedia"},
        )
        for row in query_rows
    ]
    qrels = [
        QrelRecord(
            query_id=str(first_present(row, ["query-id", "query_id", "qid"])),
            doc_id=str(first_present(row, ["corpus-id", "corpus_id", "doc_id", "cid"])),
            relevance=int(first_present(row, ["score", "relevance"], 1)),
            metadata={"dataset": "viequad_retrieval"},
        )
        for row in qrel_rows
    ]
    return PreparedDataset(
        dataset_id="vi_wiki_retrieval",
        documents=documents,
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "license": "MIT according to Hugging Face dataset card",
            "adapter": "viequad_retrieval",
            "split": split or "validation",
            "sampling_seed": seed,
            "sampled_query_count": len(query_rows),
            "source_corpus_size": source_corpus_size,
            "corpus_policy": "full_upstream_corpus",
        },
    )


def _load_mteb_triplet(split: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected_split = split or "validation"
    try:
        # This MTEB repo stores the three retrieval tables as Parquet files
        # with different schemas. Loading the repo as one dataset makes
        # Hugging Face attempt to cast qrels to the corpus schema.
        corpus = rows_from_split(
            load_hf_dataset(REPO_ID, data_files=f"corpus/{selected_split}-*.parquet", split="train")
        )
        queries = rows_from_split(
            load_hf_dataset(REPO_ID, data_files=f"queries/{selected_split}-*.parquet", split="train")
        )
        qrels = rows_from_split(load_hf_dataset(REPO_ID, data_files=f"qrels/{selected_split}-*.parquet", split="train"))
        return corpus, queries, qrels
    except Exception as exc:  # pragma: no cover - depends on upstream HF packaging
        raise RuntimeError(f"Could not load {REPO_ID} retrieval tables for split '{selected_split}': {exc}") from exc


def _row_id(row: dict[str, Any]) -> str:
    return str(first_present(row, ["_id", "id", "doc_id", "query_id", "qid", "cid"]))
