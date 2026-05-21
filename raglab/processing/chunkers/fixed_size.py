from __future__ import annotations

from raglab.core.interfaces import BaseChunker
from raglab.core.schema import Chunk, DocumentBlock
from raglab.processing.chunkers.common import make_chunk, split_tokens


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 250, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        by_doc: dict[str, list[DocumentBlock]] = {}
        for block in blocks:
            by_doc.setdefault(block.doc_id, []).append(block)
        for doc_id, doc_blocks in by_doc.items():
            text = "\n\n".join(block.text for block in doc_blocks)
            for window in split_tokens(text, self.chunk_size, self.overlap):
                chunks.append(make_chunk(doc_id, window, len(chunks), doc_blocks))
        return chunks
