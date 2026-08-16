from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import IndexedNode
from raglab.core.text import dense_cosine
from raglab.indexing.artifacts import (
    DEFAULT_FAISS_NODE_THRESHOLD,
    _source_fingerprint,
    default_store_backend,
    inspect_artifact,
    load_manifest,
    load_nodes,
    load_vector_store,
    save_nodes,
)
from raglab.indexing.vector_stores import JsonMemoryVectorStore


def _fixture_manifest() -> dict:
    return {
        "artifact_version": "4",
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


def test_json_memory_vector_store_roundtrip(tmp_path: Path) -> None:
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    save_nodes(tmp_path, nodes, _fixture_manifest(), store_spec={"type": "json_memory"})

    info = inspect_artifact(tmp_path)
    assert info["manifest"]["store"]["backend"] == "json_memory"
    assert info["has_embeddings"] is True

    store = load_vector_store(tmp_path, nodes)
    assert store is not None
    results = store.search([1.0, 0.0], 1)
    assert results[0][0].node_id == "n1"


def test_artifact_v4_moves_embeddings_out_of_nodes_json(tmp_path: Path) -> None:
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0, 0.5]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0, 0.5]),
    ]
    manifest = _fixture_manifest()
    manifest["embedding"]["dimension"] = 3
    save_nodes(tmp_path, nodes, manifest)

    raw_rows = json.loads((tmp_path / "nodes.json").read_text(encoding="utf-8"))
    assert all("embedding" not in row for row in raw_rows)
    assert (tmp_path / "embeddings.npy").exists()

    saved_manifest = load_manifest(tmp_path)
    assert saved_manifest["store"]["embeddings_path"] == "embeddings.npy"
    assert saved_manifest["store"]["embeddings_dtype"] == "float32"

    reloaded = load_nodes(tmp_path)
    assert [node.embedding for node in reloaded] == [
        pytest.approx([1.0, 0.0, 0.5]),
        pytest.approx([0.0, 1.0, 0.5]),
    ]


def test_save_nodes_rejects_partial_embeddings(tmp_path: Path) -> None:
    """validate_manifest() (called before embeddings are ever written) already
    enforces all-or-nothing embeddings — this is what makes it safe for
    _save_embeddings() to assume that invariant rather than re-check it."""
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", None),
    ]
    manifest = _fixture_manifest()
    with pytest.raises(RuntimeError, match="one or more nodes have no embedding"):
        save_nodes(tmp_path, nodes, manifest)
    assert not (tmp_path / "embeddings.npy").exists()


def test_sparse_technique_writes_no_embeddings_file(tmp_path: Path) -> None:
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", None),
        IndexedNode("n2", "c2", "d2", "beta", "beta", None),
    ]
    manifest = _fixture_manifest()
    manifest["embedding"] = {"type": "none", "model": None, "dimension": None}
    save_nodes(tmp_path, nodes, manifest)

    assert not (tmp_path / "embeddings.npy").exists()
    saved_manifest = load_manifest(tmp_path)
    assert saved_manifest["store"]["embeddings_path"] is None
    assert all(node.embedding is None for node in load_nodes(tmp_path))


def test_artifact_v2_requires_reingest(tmp_path: Path) -> None:
    (tmp_path / "index_manifest.json").write_text('{"artifact_version":"2"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Artifact v2 không được hỗ trợ"):
        load_manifest(tmp_path)


def test_artifact_v3_requires_reingest(tmp_path: Path) -> None:
    (tmp_path / "index_manifest.json").write_text('{"artifact_version":"3"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Artifact v3 không được hỗ trợ"):
        load_manifest(tmp_path)


def test_faiss_local_vector_store_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    manifest = _fixture_manifest()
    manifest["store"] = {"backend": "faiss_local"}
    save_nodes(tmp_path, nodes, manifest, store_spec={"type": "faiss_local"})

    assert (tmp_path / "faiss.index").exists()
    store = load_vector_store(tmp_path, nodes)
    assert store is not None
    results = store.search([1.0, 0.0], 1)
    assert results[0][0].node_id == "n1"


def test_json_memory_vector_store_matches_bruteforce_ranking() -> None:
    """Vectorizing search must not change which nodes rank where — only how fast."""
    random.seed(7)
    nodes = [
        IndexedNode(f"n{i}", f"c{i}", f"d{i}", "text", "text", [random.random() for _ in range(16)])
        for i in range(50)
    ]
    query = [random.random() for _ in range(16)]

    store = JsonMemoryVectorStore()
    store.build(nodes)
    vectorized = [node.node_id for node, _ in store.search(query, top_k=10)]

    reference = sorted(nodes, key=lambda node: dense_cosine(query, node.embedding or []), reverse=True)[:10]
    assert vectorized == [node.node_id for node in reference]


def test_json_memory_vector_store_zero_query_keeps_original_node_order() -> None:
    """An all-zero query ties every node's score at 0.0 — argpartition/argsort
    without a stable tie-break could return an arbitrary subset in arbitrary
    order; the old per-node Python sort always kept original node order."""
    nodes = [IndexedNode(f"n{i}", f"c{i}", f"d{i}", "text", "text", [float(i), 1.0]) for i in range(20)]
    store = JsonMemoryVectorStore()
    store.build(nodes)

    results = store.search([0.0, 0.0], top_k=5)

    assert [node.node_id for node, _ in results] == ["n0", "n1", "n2", "n3", "n4"]
    assert all(score == 0.0 for _, score in results)


def test_json_memory_vector_store_ties_break_by_original_node_index() -> None:
    """Duplicate embeddings must rank in original node order, not FAISS/argpartition
    implementation-defined order."""
    nodes = [
        IndexedNode("n0", "c0", "d0", "text", "text", [1.0, 0.0]),
        IndexedNode("n1", "c1", "d1", "text", "text", [1.0, 0.0]),  # tied with n0
        IndexedNode("n2", "c2", "d2", "text", "text", [1.0, 0.0]),  # tied with n0, n1
        IndexedNode("n3", "c3", "d3", "text", "text", [0.0, 1.0]),  # not tied — orthogonal
    ]
    store = JsonMemoryVectorStore()
    store.build(nodes)

    results = store.search([1.0, 0.0], top_k=3)

    assert [node.node_id for node, _ in results] == ["n0", "n1", "n2"]


def test_save_nodes_removes_stale_embeddings_file_when_reingesting_sparse(tmp_path: Path) -> None:
    """Reusing an output directory: a prior dense ingest's embeddings.npy must not
    survive (and get silently reattached) once the same path is re-ingested sparse."""
    dense_nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    save_nodes(tmp_path, dense_nodes, _fixture_manifest(), store_spec={"type": "json_memory"})
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "vector_store.json").exists()

    sparse_nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", None),
        IndexedNode("n2", "c2", "d2", "beta", "beta", None),
    ]
    sparse_manifest = _fixture_manifest()
    sparse_manifest["embedding"] = {"type": "none", "model": None, "dimension": None}
    save_nodes(tmp_path, sparse_nodes, sparse_manifest)

    assert not (tmp_path / "embeddings.npy").exists()
    assert not (tmp_path / "vector_store.json").exists()
    reloaded = load_nodes(tmp_path)
    assert all(node.embedding is None for node in reloaded)


def test_save_nodes_removes_stale_faiss_index_when_backend_changes(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    nodes = [
        IndexedNode("n1", "c1", "d1", "alpha", "alpha", [1.0, 0.0]),
        IndexedNode("n2", "c2", "d2", "beta", "beta", [0.0, 1.0]),
    ]
    faiss_manifest = _fixture_manifest()
    faiss_manifest["store"] = {"backend": "faiss_local"}
    save_nodes(tmp_path, nodes, faiss_manifest, store_spec={"type": "faiss_local"})
    assert (tmp_path / "faiss.index").exists()

    json_manifest = _fixture_manifest()
    save_nodes(tmp_path, nodes, json_manifest, store_spec={"type": "json_memory"})

    assert not (tmp_path / "faiss.index").exists()
    assert (tmp_path / "vector_store.json").exists()


def test_default_store_backend() -> None:
    assert default_store_backend(10, has_embeddings=False) is None
    assert default_store_backend(10, has_embeddings=True) == "json_memory"
    assert default_store_backend(DEFAULT_FAISS_NODE_THRESHOLD, has_embeddings=True) in {"json_memory", "faiss_local"}


def test_default_store_backend_falls_back_without_faiss_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert default_store_backend(DEFAULT_FAISS_NODE_THRESHOLD + 1, has_embeddings=True) == "json_memory"


def test_default_store_backend_threshold_is_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGLAB_FAISS_NODE_THRESHOLD", "5")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert default_store_backend(4, has_embeddings=True) == "json_memory"
    assert default_store_backend(5, has_embeddings=True) == "faiss_local"


def test_source_fingerprint_covers_shared_engine_code() -> None:
    # External test technique has no pipeline module; the shared raglab source
    # inventory is still non-empty and stable enough to detect engine drift.
    assert _source_fingerprint("external_test").startswith("sha256:")
