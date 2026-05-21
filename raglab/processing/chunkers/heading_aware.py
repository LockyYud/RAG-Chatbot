from __future__ import annotations

from raglab.core.interfaces import BaseChunker
from raglab.core.schema import Chunk, DocumentBlock
from raglab.core.text import token_count
from raglab.processing.chunkers.common import make_chunk, split_tokens


class HeadingAwareChunker(BaseChunker):
    def __init__(self, chunk_size: int = 350, overlap: int = 40) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        section: list[DocumentBlock] = []
        for block in blocks:
            if block.type == "heading" and section:
                self._emit_section(chunks, section)
                section = []
            section.append(block)
        if section:
            self._emit_section(chunks, section)
        return chunks

    def _emit_section(self, chunks: list[Chunk], section: list[DocumentBlock]) -> None:
        text = "\n\n".join(block.text for block in section)
        title = next((block.text.lstrip("# ").strip() for block in section if block.type == "heading"), None)
        metadata = {"section_title": title} if title else {}
        if token_count(text) <= self.chunk_size:
            chunks.append(make_chunk(section[0].doc_id, text, len(chunks), section, metadata=metadata))
            return
        for window in split_tokens(text, self.chunk_size, self.overlap):
            chunks.append(make_chunk(section[0].doc_id, window, len(chunks), section, metadata=metadata))
