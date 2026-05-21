from __future__ import annotations

from pathlib import Path

from raglab.core.interfaces import BaseParser
from raglab.core.schema import DocumentBlock


class TextParser(BaseParser):
    def parse(self, path: str) -> list[DocumentBlock]:
        source = Path(path)
        text = source.read_text(encoding="utf-8")
        doc_id = source.stem
        blocks: list[DocumentBlock] = []
        paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
        for index, paragraph in enumerate(paragraphs):
            block_type = "heading" if _looks_like_heading(paragraph) else "paragraph"
            blocks.append(
                DocumentBlock(
                    block_id=f"{doc_id}:b{index + 1}",
                    doc_id=doc_id,
                    type=block_type,
                    text=paragraph,
                    metadata={"source_path": str(source), "order": index},
                )
            )
        return blocks


def _looks_like_heading(text: str) -> bool:
    compact = text.strip()
    if "\n" in compact:
        return False
    if compact.startswith("#"):
        return True
    if len(compact) <= 80 and compact[:1].isdigit() and "." in compact[:8]:
        return True
    return len(compact) <= 70 and compact.isupper()
