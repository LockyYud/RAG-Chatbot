"""Contract every technique pipeline must satisfy.

A technique lives in ``techniques/<id>/pipeline.py`` and exposes a class
inheriting from :class:`BasePipeline`. The CLI, the benchmark runner, and the
evaluation runner all interact with techniques only through this interface, so
that any paper — no matter how exotic its internals — can be ingested,
queried, evaluated and benchmarked uniformly.

The two abstract methods are intentionally minimal:

* :meth:`ingest` reads documents, runs the technique's processing/indexing
  flow, persists everything needed for retrieval, and returns a manifest dict.
* :meth:`query` loads the persisted artifact, runs the technique's inference
  flow, and returns a :class:`RAGAnswer` whose ``metadata`` field follows the
  schema produced by :func:`raglab.core.measure.build_query_metadata`.

Everything else (chunking, embedding, generation, custom retrievers …) lives
inside the technique's own ``pipeline.py``.  There is no plugin registry, no
typed config dataclass, no YAML overlay — reading the file top-to-bottom tells
you exactly what the paper does.
"""

from __future__ import annotations

import importlib
import inspect
import json
from abc import ABC, abstractmethod
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from raglab.core.schema import JSONValue
from raglab.core.spec import TechniqueSpec, technique_spec

if TYPE_CHECKING:
    from raglab.core.schema import ArtifactManifest, RAGAnswer


class BasePipeline(ABC):
    """Interface contract for a self-contained RAG technique."""

    #: Unique identifier matching the directory name under ``techniques/``.
    id: ClassVar[str]

    #: Human-readable display name (paper title or short label).
    name: ClassVar[str] = ""

    #: Paper/research implementation fidelity recorded in artifact metadata.
    implementation_level: ClassVar[str] = "unspecified"

    #: Constructor fields that may be changed without rebuilding the artifact.
    query_override_fields: ClassVar[frozenset[str]] = frozenset()

    @abstractmethod
    def ingest(self, input_path: str, output_path: str) -> ArtifactManifest:
        """Run end-to-end ingestion and persist a queryable artifact.

        Returns the manifest dict that was written to ``output_path``. Use
        :func:`raglab.core.measure.build_ingest_manifest` to build it so
        every technique produces a comparable schema.
        """

    @abstractmethod
    def query(self, artifact_path: str, question: str, mode: str = "full_rag") -> RAGAnswer:
        """Run end-to-end inference against a persisted artifact.

        ``mode`` is either ``"full_rag"`` (retrieve + generate + verify) or
        ``"retrieval_only"`` (skip generation; useful for retrieval-only
        evaluation).  Returns a :class:`RAGAnswer` whose ``metadata`` field
        was populated via :func:`raglab.core.measure.build_query_metadata`.
        """

    # ─── Optional helpers ──────────────────────────────────────────────────

    @property
    def technique_dir(self) -> Path:
        """Directory containing this pipeline's ``pipeline.py`` file."""
        if hasattr(self, "_technique_dir"):
            return self._technique_dir
        try:
            return Path(inspect.getfile(type(self))).parent
        except TypeError:
            raise RuntimeError(
                f"Cannot determine technique_dir for {type(self).__name__}. "
                "Use load_pipeline_class() to instantiate it."
            ) from None

    def load_metadata(self) -> dict[str, Any]:
        """Read the paper metadata sitting next to ``pipeline.py``."""
        yaml_path = self.technique_dir / "technique.yaml"
        if yaml_path.exists():
            from raglab.core.config import load_config

            return load_config(yaml_path)
        return {"id": self.id}

    def resolved_config(self) -> dict[str, JSONValue]:
        """Return every resolved constructor parameter as JSON-safe data.

        Pipeline constructors deliberately assign each parameter to an instance
        attribute with the same name.  Enforcing that convention here prevents
        artifact configuration from silently drifting when a new parameter is
        added to a technique.
        """
        config: dict[str, JSONValue] = {}
        signature = inspect.signature(type(self).__init__)
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            if not hasattr(self, name):
                raise RuntimeError(
                    f"{type(self).__name__} must assign constructor parameter {name!r} "
                    "to an attribute so it can be persisted in artifact v3"
                )
            value = getattr(self, name)
            try:
                json.dumps(value, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Pipeline parameter {name!r} is not JSON serializable") from exc
            config[name] = value
        return config

    def load_artifact(self, artifact_path: str):
        """Load an artifact and reject configuration drift before querying."""
        from raglab.indexing.artifacts import load_artifact

        manifest, nodes = load_artifact(artifact_path, expected_pipeline_id=self.id)
        persisted = manifest["pipeline"]["config"]
        current = self.resolved_config()
        locked_fields = set(persisted) - set(self.query_override_fields)
        mismatches = [name for name in sorted(locked_fields) if name not in current or current[name] != persisted[name]]
        if mismatches:
            details = ", ".join(
                f"{name}={current.get(name)!r} (artifact={persisted.get(name)!r})" for name in mismatches
            )
            raise RuntimeError(f"Pipeline configuration does not match artifact v3: {details}")
        return manifest, nodes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r})"


# ─── Loader ──────────────────────────────────────────────────────────────────


def load_pipeline_class(technique_id: str) -> type[BasePipeline] | None:
    """Import a bundled ``techniques.<id>.pipeline`` module safely."""
    if technique_id not in _technique_ids():
        return None
    module = importlib.import_module(f"techniques.{technique_id}.pipeline")

    for _name, obj in vars(module).items():
        if (
            isinstance(obj, type)
            and issubclass(obj, BasePipeline)
            and obj is not BasePipeline
            and not getattr(obj, "__abstractmethods__", None)
        ):
            return obj

    return None


def load_pipeline(
    technique_id: str,
    params: dict[str, Any] | None = None,
    **legacy_kwargs: Any,
) -> BasePipeline | None:
    """Instantiate the technique's pipeline class with optional kwargs.

    ``kwargs`` are forwarded to the pipeline class constructor.  This is how
    CLI ``--param key=value`` flags propagate to a technique without the CLI
    having to know its parameter schema.
    """
    cls = load_pipeline_class(technique_id)
    if cls is None:
        return None
    kwargs = {**(params or {}), **legacy_kwargs}
    return cls(**kwargs)


def load_pipeline_for_artifact(
    technique_id: str,
    artifact_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> BasePipeline:
    """Instantiate a pipeline from persisted config plus safe query overrides."""
    from raglab.indexing.artifacts import load_manifest

    cls = load_pipeline_class(technique_id)
    if cls is None:
        raise KeyError(f"Unknown technique '{technique_id}'")
    manifest = load_manifest(artifact_path)
    artifact_pipeline_id = manifest["pipeline"].get("id")
    if artifact_pipeline_id != technique_id:
        raise RuntimeError(
            f"Artifact belongs to pipeline '{artifact_pipeline_id}', not requested pipeline '{technique_id}'."
        )
    requested = overrides or {}
    invalid = sorted(set(requested) - set(cls.query_override_fields))
    if invalid:
        raise ValueError("Query overrides affect the persisted pipeline or are unsupported: " + ", ".join(invalid))
    config = dict(manifest["pipeline"].get("config", {}))
    config.update(requested)
    return cls(**config)


def list_pipelines(root: str | Path | None = None) -> list[dict[str, Any]]:
    """Return paper metadata for every technique under *root*.

    A technique is recognised by the presence of ``pipeline.py``.  Metadata
    comes from ``technique.yaml`` (paper title, authors, tags, ...) — the
    code itself is *not* imported here, keeping the listing fast and side
    effect–free.
    """
    del root
    items: list[dict[str, Any]] = []
    package_root = resources.files("techniques")
    for technique_id in _technique_ids():
        technique_dir = package_root.joinpath(technique_id)
        yaml_resource = technique_dir.joinpath("technique.yaml")
        metadata = (
            json.loads(yaml_resource.read_text(encoding="utf-8")) if yaml_resource.is_file() else {"id": technique_id}
        )
        metadata.setdefault("id", technique_id)
        technique_spec(metadata)
        metadata["_package"] = f"techniques.{technique_id}"
        items.append(metadata)

    return items


def get_pipeline_metadata(technique_id: str, root: str | Path | None = None) -> dict[str, Any]:
    """Return metadata for one technique or raise ``KeyError``."""
    for item in list_pipelines(root):
        if item.get("id") == technique_id:
            return item
    known = ", ".join(item.get("id", "?") for item in list_pipelines(root))
    raise KeyError(f"Unknown technique '{technique_id}'. Known: {known}")


def get_pipeline_spec(technique_id: str, root: str | Path | None = None) -> TechniqueSpec:
    """Return the validated runner-facing contract for one bundled technique."""
    return technique_spec(get_pipeline_metadata(technique_id, root))


def _technique_ids() -> list[str]:
    root = resources.files("techniques")
    return sorted(
        item.name
        for item in root.iterdir()
        if item.is_dir() and not item.name.startswith("_") and item.joinpath("pipeline.py").is_file()
    )
