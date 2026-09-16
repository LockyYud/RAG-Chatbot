from __future__ import annotations

from typing import Any

from ragbench.datasets.adapters.common import first_present, limit_rows, load_hf_dataset, rows_from_split
from ragbench.datasets.schema import DocumentRecord, PreparedDataset, QrelRecord, QueryRecord

REPO_ID = "minhnguyent546/zalo-ai-legal-text-retrieval-2021"
# Pinned so this benchmark reproduces a year from now even if upstream
# reshapes or relabels the repo. See datasets/adapters/uit_viquad.py for the
# same convention.
REVISION = "5fb3df4300b3a81455fe09af444ec159afa8d212"


def prepare_zalo_legal_retrieval(
    split: str | None = "test",
    limit: int | None = None,
    seed: int = 42,
) -> PreparedDataset:
    """Zalo AI Challenge 2021 legal text retrieval, as a retrieval-only benchmark.

    Upstream ships one query pool (3,196 questions) against one corpus
    (61,425 legal documents), with qrels split into ``train`` (2,556) and
    ``test`` (640) subsets. ``split`` selects which qrels subset to evaluate;
    the query set is narrowed to only the queries judged in that subset, since
    a query with no qrel in the chosen split would otherwise look like an
    unanswerable question with a missing positive qrel.
    """
    qrel_split = split or "test"
    corpus_rows, query_rows, qrel_rows = _load_zalo_triplet(qrel_split)

    judged_query_ids = {str(first_present(row, ["query_id", "query-id"])) for row in qrel_rows}
    evaluated_queries = [row for row in query_rows if str(first_present(row, ["query_id", "_id"])) in judged_query_ids]
    evaluated_queries = limit_rows(evaluated_queries, limit, seed)
    query_ids = {str(first_present(row, ["query_id", "_id"])) for row in evaluated_queries}
    qrel_rows = [row for row in qrel_rows if str(first_present(row, ["query_id", "query-id"])) in query_ids]

    # Keep the complete corpus even when selecting a smaller query fixture.
    # Restricting it to positive documents would remove hard negatives and
    # produce retrieval scores that cannot be compared with the upstream task.
    source_corpus_size = len(corpus_rows)

    documents = [
        DocumentRecord(
            doc_id=str(first_present(row, ["id", "_id", "corpus_id"])),
            title=str(row.get("title", "")) or None,
            text=str(first_present(row, ["text", "document", "contents"], "")),
            metadata={"dataset": "zalo_legal_retrieval", "language": "vi", "domain": "legal"},
        )
        for row in corpus_rows
    ]
    queries = [
        QueryRecord(
            query_id=str(first_present(row, ["query_id", "_id"])),
            question=str(first_present(row, ["question", "text", "query"], "")),
            metadata={"dataset": "zalo_legal_retrieval", "language": "vi", "domain": "legal"},
        )
        for row in evaluated_queries
    ]
    qrels = [
        QrelRecord(
            query_id=str(first_present(row, ["query_id", "query-id"])),
            doc_id=str(first_present(row, ["corpus_id", "corpus-id", "doc_id"])),
            relevance=int(first_present(row, ["score", "relevance"], 1)),
            metadata={"dataset": "zalo_legal_retrieval"},
        )
        for row in qrel_rows
    ]
    return PreparedDataset(
        dataset_id="vi_legal_retrieval",
        documents=documents,
        queries=queries,
        qrels=qrels,
        metadata={
            "source": REPO_ID,
            "source_revision": REVISION,
            "license": "check upstream Hugging Face dataset card (Zalo AI Challenge 2021 terms)",
            "adapter": "zalo_legal_retrieval",
            "split": qrel_split,
            "upstream_split": qrel_split,
            "annotation_type": "human",
            "task": "retrieval",
            "language": "vi",
            "domain": "legal",
            "sampling_seed": seed,
            "sampled_query_count": len(evaluated_queries),
            "source_corpus_size": source_corpus_size,
            "corpus_policy": "full_upstream_corpus",
        },
    )


def _load_zalo_triplet(qrel_split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        # This repo ships corpus/queries/qrels as three distinct HF configs
        # (not just three parquet files under one default config), each with
        # its own schema. Loading via data_files= without a config name pulls
        # in the default config's schema (qrels: corpus_id/query_id/score)
        # and casts every file to it, which corrupts the corpus/queries rows
        # instead of raising — the config name must be passed explicitly.
        corpus = rows_from_split(load_hf_dataset(REPO_ID, "corpus", split="train", revision=REVISION))
        queries = rows_from_split(load_hf_dataset(REPO_ID, "queries", split="train", revision=REVISION))
        qrels = rows_from_split(load_hf_dataset(REPO_ID, "qrels", split=qrel_split, revision=REVISION))
        return corpus, queries, qrels
    except Exception as exc:  # pragma: no cover - depends on upstream HF packaging
        raise RuntimeError(
            f"Could not load {REPO_ID} legal retrieval tables for qrels split '{qrel_split}': {exc}"
        ) from exc
