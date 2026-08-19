from __future__ import annotations

import re

from ragbench.core.interfaces import BaseCleaner
from ragbench.core.schema import DocumentBlock
from ragbench.core.text import normalize_text


class WhitespaceCleaner(BaseCleaner):
    def clean(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        cleaned: list[DocumentBlock] = []
        for block in blocks:
            block.text = normalize_text(block.text)
            if block.text:
                cleaned.append(block)
        return cleaned


class VietnameseNormalizer(BaseCleaner):
    def clean(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        for block in blocks:
            block.text = normalize_text(block.text)
            block.text = re.sub(r"\s+([,.;:!?])", r"\1", block.text)
            block.text = re.sub(r"([({\[])\s+", r"\1", block.text)
            block.text = re.sub(r"\s+([)}\]])", r"\1", block.text)
        return blocks
