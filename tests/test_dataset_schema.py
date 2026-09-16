from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.core.io import iter_input_files
from ragbench.datasets.adapters import common, viequad_retrieval, zalo_legal_retrieval
from ragbench.datasets.golden import validate_golden_dataset
from ragbench.datasets.schema import (
    DocumentRecord,
    PreparedDataset,
    QrelRecord,
    QueryRecord,
    resolve_eval_dataset_path,
    sample_processed_dataset,
    validate_processed_dataset,
    write_prepared_dataset,
)


def test_write_validate_and_sample_processed_dataset(tmp_path: Path) -> None:
    output = tmp_path / "processed"
    summary = write_prepared_dataset(
        PreparedDataset(
            dataset_id="unit_vi",
            documents=[
                DocumentRecord(
                    doc_id="law/123 article:1",
                    title="Điều 1",
                    text="Người lao động được nghỉ hằng năm theo quy định.",
                )
            ],
            queries=[
                QueryRecord(
                    query_id="q1",
                    question="Người lao động được nghỉ gì?",
                    ground_truth_answer="Nghỉ hằng năm.",
                )
            ],
            qrels=[QrelRecord(query_id="q1", doc_id="law/123 article:1", relevance=2)],
        ),
        output,
    )

    assert summary["documents"] == 1
    assert (output / "documents.jsonl").exists()
    assert (output / "queries.jsonl").exists()
    assert (output / "qrels.jsonl").exists()
    assert (output / "qa.jsonl").exists()
    assert len(list((output / "docs").glob("*.md"))) == 1
    assert resolve_eval_dataset_path(output) == str(output / "qa.jsonl")

    validation = validate_processed_dataset(output)
    assert validation["queries"] == 1
    assert validation["qrels"] == 1

    sample = tmp_path / "sample"
    sample_summary = sample_processed_dataset(output, sample, 1)
    assert sample_summary["queries"] == 1
    assert validate_processed_dataset(sample)["documents"] == 1


def test_golden_dataset_validation_requires_claims_and_spans(tmp_path: Path) -> None:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        '{"question_id":"g1","question":"Q","reference_answer":"A","is_answerable":true,'
        '"required_claims":["A"],"evidence_spans":[{"doc_id":"d1","text":"A"}]}\n',
        encoding="utf-8",
    )
    assert validate_golden_dataset(golden)["evidence_spans"] == 1


def test_viequad_query_sampling_keeps_full_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = [{"_id": "d1", "text": "one"}, {"_id": "d2", "text": "two"}, {"_id": "d3", "text": "hard negative"}]
    queries = [{"_id": "q1", "text": "one?"}, {"_id": "q2", "text": "two?"}]
    qrels = [{"query-id": "q1", "corpus-id": "d1", "score": 1}, {"query-id": "q2", "corpus-id": "d2", "score": 1}]
    monkeypatch.setattr(viequad_retrieval, "_load_mteb_triplet", lambda split: (corpus, queries, qrels))
    prepared = viequad_retrieval.prepare_viequad_retrieval(limit=1, seed=3)
    assert len(prepared.documents) == 3
    assert prepared.metadata["corpus_policy"] == "full_upstream_corpus"


def test_zalo_adapter_scopes_queries_to_split_qrels_and_keeps_full_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = [
        {"id": "d1", "title": "T1", "text": "Điều khoản một"},
        {"id": "d2", "title": "T2", "text": "Điều khoản hai"},
        {"id": "d3", "title": "T3", "text": "hard negative"},
    ]
    queries = [
        {"query_id": "q1", "question": "Câu hỏi một?"},
        {"query_id": "q2", "question": "Câu hỏi hai?"},
        {"query_id": "q3", "question": "Câu hỏi chưa có qrel ở split này?"},
    ]
    qrels = {
        "train": [{"query_id": "q1", "corpus_id": "d1", "score": 1}, {"query_id": "q3", "corpus_id": "d3", "score": 1}],
        "test": [{"query_id": "q2", "corpus_id": "d2", "score": 1}],
    }

    def load_fixture(repo_id: str, *args: object, **kwargs: object) -> list[dict[str, object]]:
        assert repo_id == zalo_legal_retrieval.REPO_ID
        assert kwargs["revision"] == zalo_legal_retrieval.REVISION
        data_files = kwargs["data_files"]
        assert isinstance(data_files, str)
        if data_files.startswith("corpus/"):
            return corpus
        if data_files.startswith("queries/"):
            return queries
        return qrels[data_files.split("/")[1].split("-")[0]]

    monkeypatch.setattr(zalo_legal_retrieval, "load_hf_dataset", load_fixture)
    prepared = zalo_legal_retrieval.prepare_zalo_legal_retrieval(split="test")

    assert len(prepared.documents) == 3
    assert [query.query_id for query in prepared.queries] == ["q2"]
    assert [qrel.doc_id for qrel in prepared.qrels] == ["d2"]
    assert prepared.metadata["corpus_policy"] == "full_upstream_corpus"
    assert prepared.metadata["source_revision"] == zalo_legal_retrieval.REVISION
    assert prepared.metadata["annotation_type"] == "human"
    assert prepared.metadata["upstream_split"] == "test"


def test_zalo_prepared_dataset_passes_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = [{"id": "d1", "title": "T1", "text": "Điều khoản một"}]
    queries = [{"query_id": "q1", "question": "Câu hỏi một?"}]
    qrels = [{"query_id": "q1", "corpus_id": "d1", "score": 1}]

    def load_fixture(repo_id: str, *args: object, **kwargs: object) -> list[dict[str, object]]:
        data_files = str(kwargs["data_files"])
        if data_files.startswith("corpus/"):
            return corpus
        if data_files.startswith("queries/"):
            return queries
        return qrels

    monkeypatch.setattr(zalo_legal_retrieval, "load_hf_dataset", load_fixture)
    prepared = zalo_legal_retrieval.prepare_zalo_legal_retrieval(split="test")
    summary = write_prepared_dataset(prepared, tmp_path / "zalo")
    assert summary["queries"] == 1

    validation = validate_processed_dataset(tmp_path / "zalo")
    assert validation["annotation_type"] == "human"
    assert validation["provenance_warnings"] == []


def test_require_datasets_reports_local_namespace_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    class LocalDatasetsNamespace:
        pass

    monkeypatch.setattr(common.importlib, "import_module", lambda name: LocalDatasetsNamespace())

    with pytest.raises(RuntimeError, match=r"local `datasets/` fixture directory"):
        common.require_datasets()


def test_iter_input_files_rejects_missing_or_empty_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Input document path does not exist"):
        iter_input_files(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No text or Markdown documents"):
        iter_input_files(empty)


def test_resolve_eval_dataset_path_rejects_missing_fixture(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Evaluation dataset path does not exist"):
        resolve_eval_dataset_path(tmp_path / "not-prepared")


def test_viequad_adapter_loads_separate_retrieval_tables(monkeypatch: pytest.MonkeyPatch) -> None:
    tables: dict[str, list[dict[str, object]]] = {
        "corpus/validation-*.parquet": [{"_id": "d1", "title": "T", "text": "Evidence"}],
        "queries/validation-*.parquet": [{"_id": "q1", "text": "Question?"}],
        "qrels/validation-*.parquet": [{"query-id": "q1", "corpus-id": "d1", "score": 1}],
    }

    def load_fixture(repo_id: str, *args: object, **kwargs: object) -> list[dict[str, object]]:
        assert repo_id == viequad_retrieval.REPO_ID
        assert kwargs["split"] == "train"
        data_files = kwargs["data_files"]
        assert isinstance(data_files, str)
        return tables[data_files]

    monkeypatch.setattr(viequad_retrieval, "load_hf_dataset", load_fixture)
    prepared = viequad_retrieval.prepare_viequad_retrieval()

    assert [document.doc_id for document in prepared.documents] == ["d1"]
    assert [query.query_id for query in prepared.queries] == ["q1"]
    assert [qrel.doc_id for qrel in prepared.qrels] == ["d1"]
