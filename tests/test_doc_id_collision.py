from __future__ import annotations

from pathlib import Path

from ragbench.core.io import iter_input_files, relative_doc_id
from ragbench.datasets.synthetic import _load_text_chunks
from ragbench.processing.parsers.text_parser import TextParser


def _write(root: Path, relative: str, text: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_relative_doc_id_disambiguates_same_named_files_in_different_subdirs(tmp_path: Path) -> None:
    _write(tmp_path, "legal/report.md", "legal text")
    _write(tmp_path, "finance/report.md", "finance text")

    ids = {relative_doc_id(path, tmp_path) for path in iter_input_files(tmp_path)}
    assert ids == {"legal/report", "finance/report"}


def test_relative_doc_id_matches_bare_stem_for_a_flat_corpus(tmp_path: Path) -> None:
    """Existing flat corpora (no subdirectories) keep exactly the same doc_id
    as before this fix — only the nested-subdirectory collision case changes."""
    _write(tmp_path, "faq.md", "hello")
    (path,) = iter_input_files(tmp_path)
    assert relative_doc_id(path, tmp_path) == "faq" == path.stem


def test_relative_doc_id_falls_back_to_stem_for_a_single_file_root(tmp_path: Path) -> None:
    path = _write(tmp_path, "faq.md", "hello")
    assert relative_doc_id(path, path) == "faq"


def test_text_parser_produces_distinct_doc_ids_for_same_named_files(tmp_path: Path) -> None:
    _write(tmp_path, "legal/report.md", "legal text")
    _write(tmp_path, "finance/report.md", "finance text")

    parser = TextParser()
    blocks = []
    for path in iter_input_files(tmp_path):
        blocks.extend(parser.parse(str(path), root=tmp_path))

    doc_ids = {block.doc_id for block in blocks}
    block_ids = [block.block_id for block in blocks]
    assert doc_ids == {"legal/report", "finance/report"}
    assert len(block_ids) == len(set(block_ids))  # no block_id collision either


def test_text_parser_without_root_keeps_pre_fix_bare_stem_behavior(tmp_path: Path) -> None:
    """Callers that don't pass root (single-file use, or code not yet updated)
    keep the exact old behavior rather than raising or silently misbehaving."""
    path = _write(tmp_path, "faq.md", "hello world")
    blocks = TextParser().parse(str(path))
    assert blocks[0].doc_id == "faq"


def test_synthetic_dataset_chunking_disambiguates_same_named_files(tmp_path: Path) -> None:
    _write(tmp_path, "legal/report.md", "legal " * 50)
    _write(tmp_path, "finance/report.md", "finance " * 50)

    chunks = _load_text_chunks(tmp_path, max_tokens=1000)
    doc_ids = {chunk["doc_id"] for chunk in chunks}
    assert doc_ids == {"legal/report", "finance/report"}
