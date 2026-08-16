from __future__ import annotations

import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from raglab.core.io import read_json, read_jsonl, write_json, write_jsonl
from raglab.core.measure import canonical_fingerprint


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    text: str
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentRecord:
        return cls(
            doc_id=str(data["doc_id"]),
            text=str(data["text"]),
            title=str(data["title"]) if data.get("title") is not None else None,
            source=str(data["source"]) if data.get("source") is not None else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    question: str
    ground_truth_answer: str | None = None
    is_answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueryRecord:
        return cls(
            query_id=str(data["query_id"]),
            question=str(data["question"]),
            ground_truth_answer=str(data["ground_truth_answer"])
            if data.get("ground_truth_answer") is not None
            else None,
            is_answerable=bool(data.get("is_answerable", True)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class QrelRecord:
    query_id: str
    doc_id: str
    relevance: int = 1
    evidence_span: str | None = None
    citation: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QrelRecord:
        return cls(
            query_id=str(data["query_id"]),
            doc_id=str(data["doc_id"]),
            relevance=int(data.get("relevance", 1)),
            evidence_span=str(data["evidence_span"]) if data.get("evidence_span") is not None else None,
            citation=dict(data["citation"]) if isinstance(data.get("citation"), dict) else None,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class PreparedDataset:
    dataset_id: str
    documents: list[DocumentRecord]
    queries: list[QueryRecord]
    qrels: list[QrelRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


def write_prepared_dataset(
    dataset: PreparedDataset, output_dir: str | Path, *, overwrite: bool = False
) -> dict[str, Any]:
    target = Path(output_dir)
    if target.exists() and not target.is_dir():
        raise NotADirectoryError(f"Dataset output path is not a directory: {target}")
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {target}. Pass --overwrite to replace it.")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
        backup: Path | None = None
        try:
            summary = _write_prepared_dataset(dataset, staging)
            validate_processed_dataset(staging)
            backup = target.with_name(f".{target.name}.backup")
            if backup.exists():
                shutil.rmtree(backup)
            target.replace(backup)
            staging.replace(target)
            shutil.rmtree(backup)
            manifest = read_json(target / "manifest.json")
            manifest.update(
                {
                    "output_dir": str(target),
                    "docs_dir": str(target / "docs"),
                    "qa_path": str(target / "qa.jsonl"),
                }
            )
            write_json(target / "manifest.json", manifest)
            return {
                **summary,
                "output_dir": str(target),
                "docs_dir": str(target / "docs"),
                "qa_path": str(target / "qa.jsonl"),
            }
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            if backup is not None and backup.exists() and not target.exists():
                backup.replace(target)
            raise
    return _write_prepared_dataset(dataset, target)


def _write_prepared_dataset(dataset: PreparedDataset, target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    documents, queries, qrels, id_map = _normalize_ids(dataset.documents, dataset.queries, dataset.qrels)
    document_rows = [item.to_dict() for item in documents]
    query_rows = [item.to_dict() for item in queries]
    qrel_rows = [item.to_dict() for item in qrels]
    fingerprint = canonical_fingerprint({"documents": document_rows, "queries": query_rows, "qrels": qrel_rows})
    write_jsonl(target / "documents.jsonl", document_rows)
    write_jsonl(target / "queries.jsonl", query_rows)
    write_jsonl(target / "qrels.jsonl", qrel_rows)
    _write_docs_dir(target / "docs", documents)
    write_jsonl(target / "qa.jsonl", _eval_rows(queries, qrels))
    (target / "dataset_card.md").write_text(
        _dataset_card(dataset.dataset_id, dataset.metadata, documents, queries, qrels), encoding="utf-8"
    )
    summary = {
        "dataset_id": dataset.dataset_id,
        "output_dir": str(target),
        "documents": len(documents),
        "queries": len(queries),
        "qrels": len(qrels),
        "docs_dir": str(target / "docs"),
        "qa_path": str(target / "qa.jsonl"),
        "id_map_size": len(id_map),
        "fingerprint": fingerprint,
    }
    write_json(target / "manifest.json", {**summary, "metadata": dataset.metadata})
    return summary


def validate_processed_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    source = Path(dataset_dir)
    required_files = ["documents.jsonl", "queries.jsonl", "qrels.jsonl"]
    missing_files = [name for name in required_files if not (source / name).exists()]
    if missing_files:
        raise ValueError(f"{source} is not a processed evaluation dataset; missing: {', '.join(missing_files)}")
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(source / "documents.jsonl")]
    queries = [QueryRecord.from_dict(row) for row in read_jsonl(source / "queries.jsonl")]
    qrels = [QrelRecord.from_dict(row) for row in read_jsonl(source / "qrels.jsonl")]
    errors: list[str] = []
    doc_ids = [item.doc_id for item in documents]
    query_ids = [item.query_id for item in queries]
    doc_id_set, query_id_set = set(doc_ids), set(query_ids)
    if len(doc_ids) != len(doc_id_set):
        errors.append("documents.jsonl contains duplicate doc_id values")
    if len(query_ids) != len(query_id_set):
        errors.append("queries.jsonl contains duplicate query_id values")
    missing_docs = sorted({qrel.doc_id for qrel in qrels if qrel.doc_id not in doc_id_set})
    missing_queries = sorted({qrel.query_id for qrel in qrels if qrel.query_id not in query_id_set})
    if missing_docs:
        errors.append(f"qrels reference missing doc_id values: {missing_docs[:10]}")
    if missing_queries:
        errors.append(f"qrels reference missing query_id values: {missing_queries[:10]}")
    positives = {query_id: 0 for query_id in query_id_set}
    for qrel in qrels:
        if qrel.relevance > 0:
            positives[qrel.query_id] = positives.get(qrel.query_id, 0) + 1
    for query in queries:
        if query.is_answerable and positives.get(query.query_id, 0) == 0:
            errors.append(f"answerable query has no positive qrel: {query.query_id}")
        if not query.is_answerable and positives.get(query.query_id, 0) > 0:
            errors.append(f"unanswerable query has positive qrels: {query.query_id}")
    if not documents:
        errors.append("documents.jsonl is empty")
    if not queries:
        errors.append("queries.jsonl is empty")
    qa_path, docs_dir = source / "qa.jsonl", source / "docs"
    if not qa_path.exists():
        errors.append("qa.jsonl is missing")
    elif read_jsonl(qa_path) != _eval_rows(queries, qrels):
        errors.append("qa.jsonl is inconsistent with qrels")
    if not docs_dir.exists():
        errors.append("docs/ directory is missing")
    else:
        actual_docs = {path.stem for path in docs_dir.glob("*.md")}
        if actual_docs != doc_id_set:
            missing_exported_docs = sorted(doc_id_set - actual_docs)[:10]
            extra_exported_docs = sorted(actual_docs - doc_id_set)[:10]
            errors.append(f"docs/ IDs mismatch: missing={missing_exported_docs}, extra={extra_exported_docs}")
    if errors:
        raise ValueError("; ".join(errors))
    fingerprint = canonical_fingerprint(
        {
            "documents": [x.to_dict() for x in documents],
            "queries": [x.to_dict() for x in queries],
            "qrels": [x.to_dict() for x in qrels],
        }
    )
    return {
        "dataset_dir": str(source),
        "documents": len(documents),
        "queries": len(queries),
        "qrels": len(qrels),
        "answerable_queries": sum(1 for query in queries if query.is_answerable),
        "corpus_policy": _corpus_policy(documents, qrels, source / "manifest.json"),
        "fingerprint": fingerprint,
    }


def sample_processed_dataset(
    dataset_dir: str | Path, output_dir: str | Path, limit: int, *, seed: int = 42, overwrite: bool = False
) -> dict[str, Any]:
    if limit <= 0:
        raise ValueError("sample limit must be greater than zero")
    source = Path(dataset_dir)
    validation = validate_processed_dataset(source)
    queries = sorted(
        (QueryRecord.from_dict(row) for row in read_jsonl(source / "queries.jsonl")), key=lambda item: item.query_id
    )
    random.Random(seed).shuffle(queries)
    queries = queries[:limit]
    query_ids = {item.query_id for item in queries}
    qrels = [QrelRecord.from_dict(row) for row in read_jsonl(source / "qrels.jsonl") if row["query_id"] in query_ids]
    # Sampling selects evaluation queries, not a smaller retrieval corpus. Keeping
    # the corpus fixed preserves hard negatives and makes scores comparable.
    documents = [DocumentRecord.from_dict(row) for row in read_jsonl(source / "documents.jsonl")]
    dataset = PreparedDataset(
        dataset_id=f"{source.name}_sample_{limit}",
        documents=documents,
        queries=queries,
        qrels=qrels,
        metadata={
            "source_dataset_dir": str(source),
            "source_fingerprint": validation["fingerprint"],
            "sample_limit": limit,
            "seed": seed,
        },
    )
    return write_prepared_dataset(dataset, output_dir, overwrite=overwrite)


def resolve_eval_dataset_path(dataset_path: str | Path) -> str:
    source = Path(dataset_path)
    if not source.exists():
        raise FileNotFoundError(f"Evaluation dataset path does not exist: {source}")
    if source.is_dir():
        qa_path = source / "qa.jsonl"
        if not qa_path.exists():
            raise FileNotFoundError(f"Processed dataset directory is missing qa.jsonl: {source}")
        return str(qa_path)
    return str(source)


def _normalize_ids(
    documents: list[DocumentRecord], queries: list[QueryRecord], qrels: list[QrelRecord]
) -> tuple[list[DocumentRecord], list[QueryRecord], list[QrelRecord], dict[str, str]]:
    id_map: dict[str, str] = {}
    used: set[str] = set()
    normalized_docs: list[DocumentRecord] = []
    for doc in documents:
        safe_id = _safe_id(doc.doc_id)
        candidate = safe_id
        suffix = 2
        while candidate in used:
            candidate = f"{safe_id}_{suffix}"
            suffix += 1
        used.add(candidate)
        id_map[doc.doc_id] = candidate
        metadata = dict(doc.metadata)
        if candidate != doc.doc_id:
            metadata["original_doc_id"] = doc.doc_id
        normalized_docs.append(DocumentRecord(candidate, doc.text, doc.title, doc.source, metadata))
    normalized_qrels = [
        QrelRecord(
            qrel.query_id,
            id_map.get(qrel.doc_id, _safe_id(qrel.doc_id)),
            qrel.relevance,
            qrel.evidence_span,
            qrel.citation,
            qrel.metadata,
        )
        for qrel in qrels
    ]
    return normalized_docs, queries, normalized_qrels, id_map


def _write_docs_dir(docs_dir: Path, documents: list[DocumentRecord]) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        title = doc.title or doc.doc_id
        metadata_lines = [
            f"dataset_doc_id: {doc.doc_id}",
            *(f"{k}: {v}" for k, v in sorted(doc.metadata.items()) if isinstance(v, str | int | float)),
        ]
        (docs_dir / f"{doc.doc_id}.md").write_text(
            f"# {title}\n\n<!--\n" + "\n".join(metadata_lines) + f"\n-->\n\n{doc.text.strip()}\n", encoding="utf-8"
        )


def _eval_rows(queries: list[QueryRecord], qrels: list[QrelRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[QrelRecord]] = {}
    for qrel in qrels:
        if qrel.relevance > 0:
            grouped.setdefault(qrel.query_id, []).append(qrel)
    rows = []
    for query in queries:
        relevant = grouped.get(query.query_id, []) if query.is_answerable else []
        expected_doc_ids = sorted({qrel.doc_id for qrel in relevant})
        relevance_by_doc_id: dict[str, int] = {}
        for qrel in relevant:
            relevance_by_doc_id[qrel.doc_id] = max(relevance_by_doc_id.get(qrel.doc_id, 0), int(qrel.relevance), 0)
        rows.append(
            {
                "question_id": query.query_id,
                "question": query.question,
                "ground_truth_answer": query.ground_truth_answer,
                "expected_doc_ids": expected_doc_ids,
                "expected_chunk_ids": [],
                "expected_citations": expected_doc_ids,
                "metadata": {
                    **query.metadata,
                    "is_answerable": query.is_answerable,
                    "dataset_query_id": query.query_id,
                    "relevance_by_doc_id": relevance_by_doc_id,
                },
            }
        )
    return rows


def _dataset_card(
    dataset_id: str,
    metadata: dict[str, Any],
    documents: list[DocumentRecord],
    queries: list[QueryRecord],
    qrels: list[QrelRecord],
) -> str:
    return "\n".join(
        [
            f"# {dataset_id}",
            "",
            "This processed dataset is an evaluation fixture for rag-pipeline-lab research benchmarks.",
            "It is not part of the user document ingestion path.",
            "",
            "## Summary",
            "",
            f"- Documents: {len(documents)}",
            f"- Queries: {len(queries)}",
            f"- Qrels: {len(qrels)}",
            f"- Source: {metadata.get('source', 'unknown')}",
            f"- License: {metadata.get('license', 'check upstream dataset card')}",
            f"- Corpus policy: {metadata.get('corpus_policy', 'dataset-defined')}",
            "",
            "## Files",
            "",
            "- `documents.jsonl`: canonical retrievable documents.",
            "- `queries.jsonl`: canonical evaluation questions.",
            "- `qrels.jsonl`: query-to-document relevance labels.",
            "- `docs/`: markdown export used by ingest.",
            "- `qa.jsonl`: evaluation export used by eval.",
            "",
        ]
    )


def _corpus_policy(documents: list[DocumentRecord], qrels: list[QrelRecord], manifest_path: Path) -> str:
    """Expose a warning-friendly policy label without guessing corpus validity.

    A small corpus can be perfectly legitimate for an in-domain benchmark, so
    validation does not reject it. Upstream adapters instead declare their
    policy explicitly and benchmark reports carry that declaration forward.
    """
    if manifest_path.exists():
        try:
            metadata = read_json(manifest_path).get("metadata", {})
            if isinstance(metadata, dict) and isinstance(metadata.get("corpus_policy"), str):
                return metadata["corpus_policy"]
        except (OSError, ValueError):
            pass
    positives = {item.doc_id for item in qrels if item.relevance > 0}
    return "positive_only_suspected" if documents and {doc.doc_id for doc in documents} <= positives else "unspecified"


def _safe_id(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return compact[:180] or "doc"
