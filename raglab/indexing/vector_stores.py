from __future__ import annotations

from pathlib import Path
from typing import Any

from raglab.core.interfaces import BaseVectorStore
from raglab.core.io import read_json, write_json
from raglab.core.schema import IndexedNode
from raglab.core.text import dense_cosine


class JsonMemoryVectorStore(BaseVectorStore):
    backend = "json_memory"

    def __init__(self, **_: Any) -> None:
        self.nodes: list[IndexedNode] = []

    def build(self, nodes: list[IndexedNode]) -> None:
        _require_embeddings(nodes, self.backend)
        self.nodes = list(nodes)

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

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[IndexedNode, float]]:
        scored = [(node, dense_cosine(query_embedding, node.embedding or [])) for node in self.nodes]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


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
