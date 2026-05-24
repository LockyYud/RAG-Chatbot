from __future__ import annotations

from raglab.core.interfaces import BaseChunker
from raglab.core.schema import Chunk, DocumentBlock
from raglab.core.text import token_count
from raglab.processing.chunkers.common import make_chunk, split_tokens


class ParentChildChunker(BaseChunker):
    def __init__(self, child_size: int = 220, child_overlap: int = 35) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        section: list[DocumentBlock] = []
        for block in blocks:
            if block.type == "heading" and section:
                self._emit_children(chunks, section)
                section = []
            section.append(block)
        if section:
            self._emit_children(chunks, section)
        return chunks

    def _emit_children(self, chunks: list[Chunk], section: list[DocumentBlock]) -> None:
        text = "\n\n".join(block.text for block in section)
        parent_id = f"{section[0].doc_id}:p{len(chunks) + 1}"
        title = next((block.text.lstrip("# ").strip() for block in section if block.type == "heading"), None)
        metadata = {
            "parent_text": text,
            "section_title": title,
            "parent_token_count": token_count(text),
        }
        for window in split_tokens(text, self.child_size, self.child_overlap):
            chunks.append(
                make_chunk(section[0].doc_id, window, len(chunks), section, parent_id=parent_id, metadata=metadata)
            )
