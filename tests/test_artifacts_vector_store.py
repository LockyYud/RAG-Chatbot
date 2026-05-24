from __future__ import annotations

from pathlib import Path

from raglab.core.schema import IndexedNode
from raglab.indexing.artifacts import inspect_artifact, load_vector_store, save_nodes


def test_json_memory_vector_store_roundtrip(tmp_path: Path) -> None:
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    manifest = {
        "artifact_version": "2",
        "node_count": 2,
        "store_backend": "json_memory",
        "embedding_model": "test-embedding",
    }
    save_nodes(tmp_path, nodes, manifest, store_spec={"type": "json_memory"})

    info = inspect_artifact(tmp_path)
    assert info["manifest"]["store_backend"] == "json_memory"
    assert info["has_embeddings"] is True

    store = load_vector_store(tmp_path, nodes)
    assert store is not None
    results = store.search([1.0, 0.0], 1)
    assert results[0][0].node_id == "n1"
