from __future__ import annotations

from ragbench.core.interfaces import BaseEnricher
from ragbench.core.schema import Chunk, IndexedNode


def _node_metadata(chunk: Chunk) -> dict:
    metadata = dict(chunk.metadata)
    if chunk.parent_id is not None:
        metadata["parent_id"] = chunk.parent_id
    return metadata


class NoEnricher(BaseEnricher):
    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        return [
            IndexedNode(
                node_id=chunk.chunk_id,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text_for_embedding=chunk.text,
                text_for_generation=chunk.text,
                metadata=_node_metadata(chunk),
            )
            for chunk in chunks
        ]


class SectionTitleEnricher(BaseEnricher):
    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        nodes: list[IndexedNode] = []
        for chunk in chunks:
            title = chunk.metadata.get("section_title")
            embedding_text = f"{title}\n\n{chunk.text}" if title else chunk.text
            nodes.append(
                IndexedNode(
                    node_id=chunk.chunk_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text_for_embedding=embedding_text,
                    text_for_generation=chunk.text,
                    metadata=_node_metadata(chunk),
                )
            )
        return nodes
