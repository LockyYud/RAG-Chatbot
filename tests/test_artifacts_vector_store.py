from __future__ import annotations

from pathlib import Path

import pytest

from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import IndexedNode
from raglab.indexing.artifacts import (
    _source_fingerprint,
    inspect_artifact,
    load_manifest,
    load_vector_store,
    save_nodes,
)


def test_json_memory_vector_store_roundtrip(tmp_path: Path) -> None:
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    manifest = {
        "artifact_version": "3",
        "pipeline": {
            "id": "test",
            "implementation_level": "test",
            "config": {},
            "config_fingerprint": canonical_fingerprint({"id": "test", "config": {}}),
        },
        "embedding": {"type": "dense", "model": "test-embedding", "dimension": 2},
        "store": {"backend": "json_memory"},
        "corpus": {
            "fingerprint": "sha256:corpus",
            "documents": ["d1", "d2"],
            "document_count": 2,
            "block_count": 2,
            "chunk_count": 2,
            "node_count": 2,
        },
        "runtime": {"created_at": "now", "raglab_version": "0.2.0", "input_path": "fixture"},
    }
    save_nodes(tmp_path, nodes, manifest, store_spec={"type": "json_memory"})

    info = inspect_artifact(tmp_path)
    assert info["manifest"]["store"]["backend"] == "json_memory"
    assert info["has_embeddings"] is True

    store = load_vector_store(tmp_path, nodes)
    assert store is not None
    results = store.search([1.0, 0.0], 1)
    assert results[0][0].node_id == "n1"


def test_artifact_v2_requires_reingest(tmp_path: Path) -> None:
    (tmp_path / "index_manifest.json").write_text('{"artifact_version":"2"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Artifact v2 không được hỗ trợ"):
        load_manifest(tmp_path)


def test_source_fingerprint_covers_shared_engine_code() -> None:
    # External test technique has no pipeline module; the shared raglab source
    # inventory is still non-empty and stable enough to detect engine drift.
    assert _source_fingerprint("external_test").startswith("sha256:")
