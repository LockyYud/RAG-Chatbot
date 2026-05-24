from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.core.io import read_json, write_json
from raglab.core.schema import IndexedNode
from raglab.indexing.vector_stores import create_vector_store

ARTIFACT_VERSION = "2"


def save_nodes(
    path: str | Path,
    nodes: list[IndexedNode],
    manifest: dict,
    store_spec: dict[str, Any] | str | None = None,
) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "nodes.json", [node.to_dict() for node in nodes])
    if store_spec is not None:
        store = create_vector_store(store_spec)
        if store is not None:
            store.build(nodes)
            store.save(target)
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


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path) / "index_manifest.json"
    if not manifest_path.exists():
        return {}
    return dict(read_json(manifest_path))


def load_vector_store(path: str | Path, nodes: list[IndexedNode]):
    manifest = load_manifest(path)
    store_backend = manifest.get("store_backend")
    if not store_backend:
        return None
    store = create_vector_store({"type": store_backend})
    if store is None:
        return None
    store.load(path, nodes)
    return store


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    manifest = load_manifest(artifact_path)
    nodes = load_nodes(artifact_path)
    return {
        "path": str(artifact_path),
        "manifest": manifest,
        "node_count": len(nodes),
        "has_embeddings": any(node.embedding is not None for node in nodes),
        "files": sorted(item.name for item in artifact_path.iterdir() if item.is_file()),
    }
