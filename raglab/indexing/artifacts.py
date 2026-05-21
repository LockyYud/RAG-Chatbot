from __future__ import annotations

from pathlib import Path

from raglab.core.io import read_json, write_json
from raglab.core.schema import IndexedNode


def save_nodes(path: str | Path, nodes: list[IndexedNode], manifest: dict) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "nodes.json", [node.to_dict() for node in nodes])
    write_json(target / "index_manifest.json", manifest)


def load_nodes(path: str | Path) -> list[IndexedNode]:
    rows = read_json(Path(path) / "nodes.json")
    return [
        IndexedNode(
            node_id=row["node_id"],
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text_for_embedding=row["text_for_embedding"],
            text_for_generation=row["text_for_generation"],
            embedding=row.get("embedding"),
            metadata=dict(row.get("metadata", {})),
        )
        for row in rows
    ]
