from __future__ import annotations

from raglab.core.interfaces import BaseEnricher
from raglab.core.schema import Chunk, IndexedNode


class NoEnricher(BaseEnricher):
    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        return [
            IndexedNode(
                node_id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text_for_embedding=chunk.text,
                text_for_generation=chunk.metadata.get("parent_text", chunk.text),
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]


class SectionTitleEnricher(BaseEnricher):
    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        nodes: list[IndexedNode] = []
        for chunk in chunks:
            title = chunk.metadata.get("section_title")
            embedding_text = f"{title}\n\n{chunk.text}" if title else chunk.text
            generation_text = chunk.metadata.get("parent_text", chunk.text)
            nodes.append(
                IndexedNode(
                    node_id=chunk.chunk_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text_for_embedding=embedding_text,
                    text_for_generation=generation_text,
                    metadata=dict(chunk.metadata),
                )
            )
        return nodes
