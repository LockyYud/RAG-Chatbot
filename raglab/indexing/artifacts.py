from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

from raglab.core.io import read_json, write_json
from raglab.core.measure import canonical_fingerprint
from raglab.core.schema import ArtifactManifest, IndexedNode, JSONValue
from raglab.indexing.vector_stores import create_vector_store

ARTIFACT_VERSION = "3"


def save_nodes(
    path: str | Path,
    nodes: list[IndexedNode],
    manifest: ArtifactManifest,
    store_spec: dict[str, Any] | str | None = None,
) -> None:
    validate_manifest(manifest, nodes)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "nodes.json", [node.to_dict() for node in nodes])
    if store_spec is not None:
        store = create_vector_store(store_spec)
        if store is not None:
            store.build(nodes)
            store.save(target)
    _record_provenance(manifest, target)
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


def load_manifest(path: str | Path) -> ArtifactManifest:
    manifest_path = Path(path) / "index_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Artifact is missing required manifest: {manifest_path}")
    manifest = dict(read_json(manifest_path))
    version = str(manifest.get("artifact_version", ""))
    if version == "2":
        raise RuntimeError("Artifact v2 không được hỗ trợ trong v0.2; hãy chạy ingest lại.")
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
        raise RuntimeError(f"Artifact v3 manifest is missing fields: {', '.join(missing)}")

    pipeline_id = str(manifest["pipeline"].get("id", ""))
    if not pipeline_id:
        raise RuntimeError("Artifact v3 manifest is missing pipeline.id")
    if expected_pipeline_id is not None and pipeline_id != expected_pipeline_id:
        raise RuntimeError(
            f"Artifact belongs to pipeline '{pipeline_id}', not requested pipeline '{expected_pipeline_id}'."
        )
    config = manifest["pipeline"].get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Artifact v3 manifest is missing pipeline.config")
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
    if not manifest["runtime"].get("raglab_version"):
        raise RuntimeError("Artifact runtime.raglab_version is missing")


def _record_provenance(manifest: ArtifactManifest, target: Path) -> None:
    """Record source/dependency fingerprints and every persisted artifact file."""
    pipeline_id = manifest["pipeline"]["id"]
    source_fingerprint = _source_fingerprint(pipeline_id)
    dependency_versions: dict[str, str | None] = {}
    for package in ("litellm", "numpy", "faiss-cpu", "sentence-transformers", "torch"):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = None
    manifest["runtime"]["source_fingerprint"] = source_fingerprint
    manifest["runtime"]["dependency_versions"] = dependency_versions
    manifest.setdefault("extra", {})["artifact_files"] = _artifact_file_hashes(target)


def _technique_source_files(pipeline_id: str) -> list[Path]:
    try:
        module = __import__(f"techniques.{pipeline_id}.pipeline", fromlist=["*"])
    except ModuleNotFoundError:
        # Low-level artifact callers and external techniques may not live in
        # the bundled ``techniques`` package. Their empty source inventory is
        # still fingerprinted deterministically rather than preventing save.
        return []
    pipeline_path = Path(inspect.getfile(module))
    metadata_path = pipeline_path.with_name("technique.yaml")
    technique_files = [path for path in (pipeline_path, metadata_path) if path.exists()]
    # A technique's behavior also depends on shared engine code. Include every
    # shipped raglab module so artifacts cannot silently execute a changed
    # retriever, chunker, generator, or provider wrapper.
    package_root = Path(__file__).resolve().parents[1]
    shared_files = sorted(path for path in package_root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted({*technique_files, *shared_files})


def _source_fingerprint(pipeline_id: str) -> str:
    files = _technique_source_files(pipeline_id)
    project_root = Path(__file__).resolve().parents[2]
    return canonical_fingerprint(
        {
            path.relative_to(project_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        }
    )


def _validate_source_fingerprint(manifest: ArtifactManifest) -> None:
    expected = manifest["runtime"].get("source_fingerprint")
    if not isinstance(expected, str) or not expected.startswith("sha256:"):
        raise RuntimeError("Artifact runtime.source_fingerprint is missing or invalid")
    current = _source_fingerprint(manifest["pipeline"]["id"])
    if expected != current:
        raise RuntimeError("Artifact source fingerprint does not match the current implementation. Re-run ingest.")


def _artifact_file_hashes(target: Path) -> dict[str, JSONValue]:
    return {
        path.relative_to(target).as_posix(): f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "index_manifest.json"
    }


def _validate_artifact_files(target: Path, manifest: ArtifactManifest) -> None:
    recorded = manifest.get("extra", {}).get("artifact_files")
    if not isinstance(recorded, dict):
        raise RuntimeError("Artifact v3 manifest is missing extra.artifact_files")
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
