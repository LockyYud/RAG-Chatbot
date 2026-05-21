from __future__ import annotations

import hashlib

from raglab.core.schema import Chunk, DocumentBlock
from raglab.core.text import tokenize


def chunk_id(doc_id: str, text: str, index: int) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}:c{index + 1}:{digest}"


def make_chunk(
    doc_id: str,
    text: str,
    index: int,
    blocks: list[DocumentBlock],
    parent_id: str | None = None,
    metadata: dict | None = None,
) -> Chunk:
    merged_metadata = dict(metadata or {})
    if blocks:
        headings = [block.text.lstrip("# ").strip() for block in blocks if block.type == "heading"]
        if headings:
            merged_metadata["section_title"] = headings[-1]
        merged_metadata["source_path"] = blocks[0].metadata.get("source_path")
    return Chunk(
        chunk_id=chunk_id(doc_id, text, index),
        doc_id=doc_id,
        text=text.strip(),
        parent_id=parent_id,
        block_ids=[block.block_id for block in blocks],
        metadata=merged_metadata,
    )


def split_tokens(text: str, size: int, overlap: int) -> list[str]:
    tokens = tokenize(text)
    if not tokens:
        return []
    windows = []
    start = 0
    step = max(1, size - overlap)
    while start < len(tokens):
        windows.append(" ".join(tokens[start : start + size]))
        start += step
    return windows
