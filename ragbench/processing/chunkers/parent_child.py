from __future__ import annotations

from typing import Any

from ragbench.core.interfaces import BaseChunker
from ragbench.core.schema import Chunk, DocumentBlock
from ragbench.core.text import token_count
from ragbench.processing.chunkers.common import make_chunk, split_tokens


class ParentChildChunker(BaseChunker):
    """Split sections into small "child" chunks, keeping each parent's full
    text out of every child.

    ``parent_text`` used to be copied into every child's metadata (and, via
    ``SectionTitleEnricher``/``NoEnricher``, into ``text_for_generation`` too
    — a *second* copy). A section with N children then stored its own text
    N times over, in a corpus of ~260k children that inflated ``nodes.json``
    to ~10x a comparable non-parent-child artifact and OOM'd the process that
    tried to load it back (``json.load()`` gives every duplicate its own
    fresh string — no reference sharing survives a JSON round trip). Each
    child now only carries ``parent_id``; ``self.parents`` holds the one
    canonical copy of each parent's text, for the pipeline to persist
    alongside the nodes and resolve at query time (see
    ``ParentChildPipeline``).
    """

    def __init__(self, child_size: int = 220, child_overlap: int = 35) -> None:
        self.child_size = child_size
        self.child_overlap = child_overlap
        self.parents: dict[str, dict[str, Any]] = {}

    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        chunks: list[Chunk] = []
        self.parents = {}
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
        self.parents[parent_id] = {
            "text": text,
            "section_title": title,
            "token_count": token_count(text),
        }
        metadata = {"section_title": title}
        for window in split_tokens(text, self.child_size, self.child_overlap):
            chunks.append(
                make_chunk(section[0].doc_id, window, len(chunks), section, parent_id=parent_id, metadata=metadata)
            )
