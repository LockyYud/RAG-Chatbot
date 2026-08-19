from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import numpy as np

from ragbench.core.io import read_json, write_json
from ragbench.core.measure import ARTIFACT_VERSION, canonical_fingerprint
from ragbench.core.schema import ArtifactManifest, IndexedNode, JSONValue
from ragbench.indexing.vector_stores import create_vector_store
from ragbench.providers.env import env_int

EMBEDDINGS_FILE = "embeddings.npy"

# Below this many nodes, FAISS's C++ vectorized exact search isn't worth the
# extra artifact complexity over the plain numpy-vectorized json_memory path;
# above it, the same exact-search recall gets meaningfully faster. Tunable
# without a code change since the "right" corpus size varies per deployment.
DEFAULT_FAISS_NODE_THRESHOLD = 2000


def default_store_backend(node_count: int, *, has_embeddings: bool) -> str | None:
    """Pick the vector store backend a technique's ``ingest()`` should request.

    Returns ``None`` for sparse-only techniques (no embeddings to index at
    all — matches every BM25-only technique's existing behavior). Otherwise
    ``"faiss_local"`` once the corpus is large enough that vectorized C++
    search meaningfully beats numpy, unless faiss isn't installed (it is an
    optional dependency — see the ``vector`` extra), in which case this
    degrades to ``"json_memory"`` instead of failing.
    """
    if not has_embeddings:
        return None
    threshold = env_int("RAGLAB_FAISS_NODE_THRESHOLD", DEFAULT_FAISS_NODE_THRESHOLD)
    if node_count >= threshold and importlib.util.find_spec("faiss") is not None:
        return "faiss_local"
    return "json_memory"


_SIDECAR_FILES = (EMBEDDINGS_FILE, "vector_store.json", "faiss.index")


def _clear_stale_sidecars(target: Path) -> None:
    """Remove every sidecar this or a prior save_nodes() call could have written.

    A technique may reuse an output directory across ingests with a different
    embedding/backend config (sparse this time, dense last time; json_memory
    this time, faiss_local last time). Without this, a leftover
    ``embeddings.npy`` from a previous dense run would get silently reattached
    by ``load_nodes()`` to nodes that were never actually embedded this run,
    and a leftover ``faiss.index``/``vector_store.json`` from a different
    backend would sit in the artifact directory unused but still checksummed
    as though intentional.
    """
    for name in _SIDECAR_FILES:
        path = target / name
        if path.exists():
            path.unlink()


def save_nodes(
    path: str | Path,
    nodes: list[IndexedNode],
    manifest: ArtifactManifest,
    store_spec: dict[str, Any] | str | None = None,
) -> None:
    validate_manifest(manifest, nodes)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    _clear_stale_sidecars(target)
    write_json(target / "nodes.json", [_node_row_without_embedding(node) for node in nodes])
    _save_embeddings(target, nodes, manifest)
    if store_spec is not None:
        store = create_vector_store(store_spec)
        if store is not None:
            store.build(nodes)
            store.save(target)
    _record_provenance(manifest, target)
    write_json(target / "index_manifest.json", manifest)


def _node_row_without_embedding(node: IndexedNode) -> dict[str, Any]:
    row = node.to_dict()
    row.pop("embedding", None)
    return row


def _save_embeddings(target: Path, nodes: list[IndexedNode], manifest: ArtifactManifest) -> None:
    """Persist embeddings as one binary ``.npy`` array instead of inline JSON floats.

    All-or-nothing: ``validate_manifest()`` (called earlier in ``save_nodes()``)
    already rejects a partially-embedded node list via ``embedding.model`` vs.
    per-node embedding presence, so by the time this runs every node either has
    an embedding or none do — a partial set would make the node-order-aligned
    array ambiguous.
    """
    store = manifest.setdefault("store", {})
    embedded = [node for node in nodes if node.embedding is not None]
    if not embedded:
        store["embeddings_path"] = None
        store["embeddings_dtype"] = None
        return
    array = np.array([node.embedding for node in nodes], dtype="float32")
    np.save(target / EMBEDDINGS_FILE, array)
    store["embeddings_path"] = EMBEDDINGS_FILE
    store["embeddings_dtype"] = "float32"


def load_nodes(path: str | Path) -> list[IndexedNode]:
    rows = read_json(Path(path) / "nodes.json")
    embeddings_path = Path(path) / EMBEDDINGS_FILE
    # mmap avoids buffering the whole array through a plain Python read before
    # numpy ever sees it; more importantly at this project's scale, it replaces
    # slow per-float JSON parsing with numpy's binary loader entirely.
    embeddings = np.load(embeddings_path, mmap_mode="r") if embeddings_path.exists() else None
    if embeddings is not None and len(embeddings) != len(rows):
        raise RuntimeError(
            f"Artifact embeddings.npy has {len(embeddings)} row(s) but nodes.json has {len(rows)}; artifact is corrupt."
        )
    return [
        IndexedNode(
            node_id=row["node_id"],
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text_for_embedding=row["text_for_embedding"],
            text_for_generation=row["text_for_generation"],
            # A view into the mmap'd array, not materialized into a Python
            # list of boxed floats — keeps large corpora out of per-node RAM.
            embedding=embeddings[index] if embeddings is not None else None,
            metadata=dict(row.get("metadata", {})),
        )
        for index, row in enumerate(rows)
    ]


def load_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path) / "index_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Artifact is missing required manifest: {manifest_path}")
    manifest = dict(read_json(manifest_path))
    version = str(manifest.get("artifact_version", ""))
    if version in {"2", "3"}:
        raise RuntimeError(f"Artifact v{version} không được hỗ trợ trong v0.2+; hãy chạy ingest lại.")
    if version != ARTIFACT_VERSION:
        raise RuntimeError(
            f"Unsupported artifact version {version or '<missing>'}; expected {ARTIFACT_VERSION}. Re-run ingest."
        )
    return manifest  # type: ignore[return-value]


def load_artifact(
    path: str | Path, expected_pipeline_id: str | None = None
) -> tuple[ArtifactManifest, list[IndexedNode]]:
    """Load and fully validate an artifact before any provider call is made."""
    manifest = load_manifest(path)
    nodes_path = Path(path) / "nodes.json"
    if not nodes_path.exists():
        raise RuntimeError(f"Artifact is missing required nodes file: {nodes_path}")
    nodes = load_nodes(path)
    validate_manifest(manifest, nodes, expected_pipeline_id=expected_pipeline_id)
    _validate_artifact_files(Path(path), manifest)
    _validate_source_fingerprint(manifest)
    return manifest, nodes


def validate_manifest(
    manifest: ArtifactManifest,
    nodes: list[IndexedNode],
    expected_pipeline_id: str | None = None,
) -> None:
    required = {"pipeline", "embedding", "store", "corpus", "runtime"}
    missing = sorted(required - set(manifest))
    if missing:
        raise RuntimeError(f"Artifact v5 manifest is missing fields: {', '.join(missing)}")

    pipeline_id = str(manifest["pipeline"].get("id", ""))
    if not pipeline_id:
        raise RuntimeError("Artifact v5 manifest is missing pipeline.id")
    if expected_pipeline_id is not None and pipeline_id != expected_pipeline_id:
        raise RuntimeError(
            f"Artifact belongs to pipeline '{pipeline_id}', not requested pipeline '{expected_pipeline_id}'."
        )
    config = manifest["pipeline"].get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Artifact v5 manifest is missing pipeline.config")
    expected_config_fingerprint = canonical_fingerprint({"id": pipeline_id, "config": config})
    if manifest["pipeline"].get("config_fingerprint") != expected_config_fingerprint:
        raise RuntimeError("Artifact pipeline config fingerprint is missing or invalid")

    expected_count = int(manifest["corpus"].get("node_count", -1))
    if expected_count != len(nodes):
        raise RuntimeError(f"Artifact node count mismatch: manifest={expected_count}, nodes.json={len(nodes)}")

    model = manifest["embedding"].get("model")
    expected_dimension = manifest["embedding"].get("dimension")
    embeddings = [node.embedding for node in nodes if node.embedding is not None]
    if embeddings and model is None:
        raise RuntimeError("Artifact contains embeddings but embedding.model is missing")
    if embeddings and expected_dimension is None:
        raise RuntimeError("Artifact contains embeddings but embedding.dimension is missing")
    if model is not None and len(embeddings) != len(nodes):
        raise RuntimeError("Artifact embedding metadata declares a model but one or more nodes have no embedding")
    if not embeddings and (model is not None or expected_dimension is not None):
        raise RuntimeError("Artifact embedding metadata is present but nodes contain no embeddings")
    if expected_dimension is not None:
        invalid = [len(vector) for vector in embeddings if len(vector) != int(expected_dimension)]
        if invalid:
            raise RuntimeError(
                f"Artifact embedding dimension mismatch: expected {expected_dimension}, observed {sorted(set(invalid))}"
            )

    backend = manifest["store"].get("backend")
    if backend not in {"json_memory", "faiss_local"}:
        raise RuntimeError(f"Artifact uses unsupported vector store backend: {backend}")

    corpus_fingerprint = manifest["corpus"].get("fingerprint")
    if not isinstance(corpus_fingerprint, str) or not corpus_fingerprint.startswith("sha256:"):
        raise RuntimeError("Artifact corpus fingerprint is missing or invalid")
    if not manifest["runtime"].get("package_version"):
        raise RuntimeError("Artifact runtime.package_version is missing")


def _record_provenance(manifest: ArtifactManifest, target: Path) -> None:
    """Record source/dependency fingerprints and every persisted artifact file."""
    pipeline_id = manifest["pipeline"]["id"]
    dependency_versions: dict[str, str | None] = {}
    for package in ("litellm", "numpy", "faiss-cpu", "sentence-transformers", "torch"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    manifest["runtime"]["ingest_fingerprint"] = _ingest_fingerprint(pipeline_id)
    manifest["runtime"]["runtime_fingerprint"] = _runtime_fingerprint(pipeline_id)
    manifest["runtime"]["dependency_versions"] = dependency_versions
    manifest.setdefault("extra", {})["artifact_files"] = _artifact_file_hashes(target)


# Every module under one of these ``ragbench/<dir>`` prefixes is grouped by
# which artifact-affecting stage it can change. ``core`` and ``providers`` are
# genuinely shared (schema, config, the LLM/embedding client) so they gate
# both. ``cli``, ``benchmarks``, and ``datasets`` never run inside a
# technique's ingest() or query() and are deliberately left out of both — a
# CLI help-text edit or a new dataset adapter must not invalidate every
# existing artifact.
_INGEST_ONLY_DIRS = ("processing",)
_RUNTIME_ONLY_DIRS = ("inference",)
_SHARED_DIRS = ("core", "providers")


def _technique_module(pipeline_id: str) -> Any | None:
    try:
        return __import__(f"ragbench.techniques.{pipeline_id}.pipeline", fromlist=["*"])
    except ModuleNotFoundError:
        # Low-level artifact callers and external techniques may not live in
        # the bundled ``techniques`` package. Their empty source inventory is
        # still fingerprinted deterministically rather than preventing save.
        return None


def _technique_files(pipeline_id: str) -> list[Path]:
    module = _technique_module(pipeline_id)
    if module is None:
        return []
    pipeline_path = Path(inspect.getfile(module))
    metadata_path = pipeline_path.with_name("technique.yaml")
    return [path for path in (pipeline_path, metadata_path) if path.exists()]


def _shared_files(*, stage: str) -> list[Path]:
    """``stage`` is ``"ingest"`` or ``"runtime"``; picks the dirs relevant to it."""
    package_root = Path(__file__).resolve().parents[1]
    dirs = _SHARED_DIRS + (_INGEST_ONLY_DIRS if stage == "ingest" else _RUNTIME_ONLY_DIRS)
    files: list[Path] = []
    for name in dirs:
        files.extend(package_root.joinpath(name).rglob("*.py"))
    return [path for path in files if "__pycache__" not in path.parts]


def _fingerprint_files(pipeline_id: str, *, stage: str) -> list[Path]:
    return sorted({*_technique_files(pipeline_id), *_shared_files(stage=stage)})


def _hash_files(files: list[Path]) -> str:
    project_root = Path(__file__).resolve().parents[2]
    return canonical_fingerprint(
        {
            path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
    )


def _ingest_fingerprint(pipeline_id: str) -> str:
    return _hash_files(_fingerprint_files(pipeline_id, stage="ingest"))


def _runtime_fingerprint(pipeline_id: str) -> str:
    return _hash_files(_fingerprint_files(pipeline_id, stage="runtime"))


def _validate_source_fingerprint(manifest: ArtifactManifest) -> None:
    """Gate ``load_artifact()`` on ``ingest_fingerprint`` only.

    A mismatched ``runtime_fingerprint`` (retriever/reranker/generator/verifier
    changed) does not block loading — the corpus and embeddings on disk are
    still valid — but is surfaced via ``runtime_fingerprint_stale()`` so
    callers that care about reproducibility (the eval runner, benchmark
    reports) can record it instead of silently ignoring it.
    """
    expected = manifest["runtime"].get("ingest_fingerprint")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise RuntimeError("Artifact runtime.ingest_fingerprint is missing or invalid")
    current = _ingest_fingerprint(manifest["pipeline"]["id"])
    if expected != current:
        raise RuntimeError(
            "Artifact ingest fingerprint (parser/cleaner/chunker/enricher/embedder/vector "
            "store schema) does not match the current implementation. Re-run ingest."
        )


def runtime_fingerprint_stale(manifest: ArtifactManifest) -> bool:
    """True if retriever/reranker/generator/verifier code changed since this artifact was built.

    Non-fatal by design (see ``_validate_source_fingerprint``) — the artifact
    can still be loaded and queried; this is a reproducibility signal, not a
    correctness gate.
    """
    recorded = manifest["runtime"].get("runtime_fingerprint")
    if not isinstance(recorded, str):
        return True
    return recorded != _runtime_fingerprint(manifest["pipeline"]["id"])


def _artifact_file_hashes(target: Path) -> dict[str, JSONValue]:
    return {
        path.relative_to(target).as_posix(): f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "index_manifest.json"
    }


def _validate_artifact_files(target: Path, manifest: ArtifactManifest) -> None:
    recorded = manifest.get("extra", {}).get("artifact_files")
    if not isinstance(recorded, dict):
        raise RuntimeError("Artifact v5 manifest is missing extra.artifact_files")
    observed = _artifact_file_hashes(target)
    if recorded != observed:
        raise RuntimeError("Artifact file inventory or checksum does not match manifest")


def load_vector_store(path: str | Path, nodes: list[IndexedNode]):
    manifest = load_manifest(path)
    store_backend = manifest["store"].get("backend")
    if not store_backend:
        return None
    store = create_vector_store({"type": store_backend})
    if store is None:
        return None
    store.load(path, nodes)
    return store


def inspect_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    manifest, nodes = load_artifact(artifact_path)
    return {
        "path": str(artifact_path),
        "manifest": manifest,
        "node_count": len(nodes),
        "has_embeddings": any(node.embedding is not None for node in nodes),
        "files": sorted(item.name for item in artifact_path.iterdir() if item.is_file()),
    }
