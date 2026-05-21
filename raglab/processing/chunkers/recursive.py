from __future__ import annotations

from raglab.core.interfaces import BaseChunker
from raglab.core.schema import Chunk, DocumentBlock
from raglab.core.text import token_count
from raglab.processing.chunkers.common import make_chunk, split_tokens


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size: int = 300, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        current_blocks: list[DocumentBlock] = []
        current_text: list[str] = []
        for block in blocks:
            candidate = "\n\n".join([*current_text, block.text])
            if current_text and token_count(candidate) > self.chunk_size:
                self._flush(chunks, current_blocks, "\n\n".join(current_text))
                current_blocks = []
                current_text = []
            current_blocks.append(block)
            current_text.append(block.text)
        if current_text:
            self._flush(chunks, current_blocks, "\n\n".join(current_text))
        return chunks

    def _flush(self, chunks: list[Chunk], blocks: list[DocumentBlock], text: str) -> None:
        if token_count(text) <= self.chunk_size:
            chunks.append(make_chunk(blocks[0].doc_id, text, len(chunks), blocks))
            return
        for window in split_tokens(text, self.chunk_size, self.overlap):
            chunks.append(make_chunk(blocks[0].doc_id, window, len(chunks), blocks))
