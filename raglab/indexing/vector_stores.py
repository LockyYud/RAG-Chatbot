from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from raglab.core.interfaces import BaseVectorStore
from raglab.core.io import read_json, write_json
from raglab.core.schema import IndexedNode


class JsonMemoryVectorStore(BaseVectorStore):
    """Exact cosine search over every node's embedding, vectorized with numpy.

    Same ranking as a naive per-node Python cosine loop — verified equal in
    ``tests/test_artifacts_vector_store.py`` — just computed as one matrix
    multiply instead of N Python-level function calls. This is the fallback
    used below the FAISS node-count threshold (or when faiss isn't
    installed), not an approximation: latency changes, results do not.
    """

    backend = "json_memory"

    def __init__(self, **_: Any) -> None:
        self.nodes: list[IndexedNode] = []
        self._matrix: np.ndarray | None = None

    def build(self, nodes: list[IndexedNode]) -> None:
        _require_embeddings(nodes, self.backend)
        self.nodes = list(nodes)
        self._matrix = _normalized_matrix(self.nodes)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        write_json(
            target / "vector_store.json",
            {"backend": self.backend, "node_ids": [node.node_id for node in self.nodes]},
        )

    def load(self, path: str | Path, nodes: list[IndexedNode]) -> None:
        metadata_path = Path(path) / "vector_store.json"
        if metadata_path.exists():
            read_json(metadata_path)
        _require_embeddings(nodes, self.backend)
        self.nodes = list(nodes)
        self._matrix = _normalized_matrix(self.nodes)

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[IndexedNode, float]]:
        if self._matrix is None or not len(self.nodes):
            return []
        query = np.asarray(query_embedding, dtype="float32")
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        scores = self._matrix @ query
        k = min(top_k, len(self.nodes))
        # kind="stable" is required for correctness, not just style: on a tie
        # (e.g. an all-zero query against any corpus, or duplicate embeddings)
        # an unstable partition/sort may pick an arbitrary subset of the tied
        # nodes into the top-k, and in a different order run to run. A stable
        # sort matches the old per-node Python `sorted(..., reverse=True)`
        # exactly — ties keep their original node order — so ranking is
        # identical to before, not just "usually the same."
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.nodes[index], float(scores[index])) for index in order]


def _normalized_matrix(nodes: list[IndexedNode]) -> np.ndarray:
    matrix = np.array([node.embedding for node in nodes], dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # a zero vector's direction is undefined; leave it as all-zero rather than divide by 0
    return matrix / norms


class FaissLocalVectorStore(BaseVectorStore):
    backend = "faiss_local"

    def __init__(self, **_: Any) -> None:
        self.nodes_by_id: dict[str, IndexedNode] = {}
        self.node_ids: list[str] = []
        self.index: Any = None

    def build(self, nodes: list[IndexedNode]) -> None:
        _require_embeddings(nodes, self.backend)
        faiss, np = _faiss_modules()
        vectors = np.array([node.embedding for node in nodes], dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        self.index = index
        self.node_ids = [node.node_id for node in nodes]
        self.nodes_by_id = {node.node_id: node for node in nodes}

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save faiss_local store before build()")
        faiss, _ = _faiss_modules()
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target / "faiss.index"))
        write_json(target / "vector_store.json", {"backend": self.backend, "node_ids": self.node_ids})

    def load(self, path: str | Path, nodes: list[IndexedNode]) -> None:
        target = Path(path)
        metadata = read_json(target / "vector_store.json")
        if metadata.get("backend") != self.backend:
            raise RuntimeError(f"Artifact vector store is {metadata.get('backend')}, not {self.backend}")
        faiss, _ = _faiss_modules()
        self.index = faiss.read_index(str(target / "faiss.index"))
        self.node_ids = [str(node_id) for node_id in metadata.get("node_ids", [])]
        self.nodes_by_id = {node.node_id: node for node in nodes}

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[IndexedNode, float]]:
        if self.index is None:
            raise RuntimeError("faiss_local store is not loaded")
        faiss, np = _faiss_modules()
        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indexes = self.index.search(query, top_k)
        results: list[tuple[IndexedNode, float]] = []
        for score, index in zip(scores[0], indexes[0], strict=True):
            if index < 0:
                continue
            node_id = self.node_ids[int(index)]
            results.append((self.nodes_by_id[node_id], float(score)))
        return results


def create_vector_store(spec: dict[str, Any] | str | None) -> BaseVectorStore | None:
    if spec is None:
        return None
    if isinstance(spec, str):
        backend = spec
        params: dict[str, Any] = {}
    else:
        backend = str(spec.get("type", "json_memory"))
        params = dict(spec.get("params", {}))
    if backend == "json_memory":
        return JsonMemoryVectorStore(**params)
    if backend == "faiss_local":
        return FaissLocalVectorStore(**params)
    raise KeyError(f"Unknown vector store '{backend}'. Known: faiss_local, json_memory")


def _require_embeddings(nodes: list[IndexedNode], backend: str) -> None:
    missing = [node.node_id for node in nodes if node.embedding is None]
    if missing:
        raise RuntimeError(f"{backend} requires node embeddings. Missing embeddings for {len(missing)} nodes.")


def _faiss_modules() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("faiss_local requires optional dependencies: pip install '.[vector]'") from exc
    return faiss, np
