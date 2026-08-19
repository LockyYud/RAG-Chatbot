from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

import numpy as np

BlockType = Literal["title", "heading", "paragraph", "table", "figure", "list", "footnote"]
QueryMode = Literal["full_rag", "retrieval_only"]
VerificationStatus = Literal["run", "skipped"]
JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


class PipelineArtifactSpec(TypedDict):
    id: str
    implementation_level: str
    config: dict[str, JSONValue]
    config_fingerprint: str


class EmbeddingArtifactSpec(TypedDict):
    model: str | None
    dimension: int | None
    type: str


class StoreArtifactSpec(TypedDict, total=False):
    backend: str | None
    # Present only when at least one node carries an embedding (artifact v5+).
    # The embeddings themselves live in this sibling .npy file, not in
    # nodes.json — embeddings_path is relative to the artifact directory.
    embeddings_path: str | None
    embeddings_dtype: str | None


class CorpusArtifactSpec(TypedDict):
    fingerprint: str
    documents: list[str]
    document_count: int
    block_count: int
    chunk_count: int
    node_count: int


class RuntimeArtifactSpec(TypedDict):
    created_at: str
    package_version: str
    input_path: str
    # Split so a generator/reranker/retriever-only change doesn't force a full
    # re-ingest: only ingest_fingerprint gates ``load_artifact()``.
    ingest_fingerprint: str | None
    runtime_fingerprint: str | None
    dependency_versions: dict[str, str | None]


class ArtifactManifest(TypedDict, total=False):
    artifact_version: str
    pipeline: PipelineArtifactSpec
    embedding: EmbeddingArtifactSpec
    store: StoreArtifactSpec
    corpus: CorpusArtifactSpec
    runtime: RuntimeArtifactSpec
    extra: dict[str, JSONValue]


@dataclass(slots=True)
class Document:
    doc_id: str
    source_path: str | None = None
    title: str | None = None
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentBlock:
    block_id: str
    doc_id: str
    type: BlockType
    text: str
    page: int | None = None
    bbox: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    parent_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IndexedNode:
    node_id: str
    chunk_id: str
    doc_id: str
    text_for_embedding: str
    text_for_generation: str
    # A plain list right after an embedder call; a memory-mapped row view once
    # loaded from a saved artifact (``load_nodes`` no longer materializes each
    # node's vector into boxed Python floats — see ``dense_cosine``, which is
    # written to accept either without forcing a copy).
    embedding: list[float] | np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if isinstance(self.embedding, np.ndarray):
            payload["embedding"] = self.embedding.tolist()
        return payload


@dataclass(slots=True)
class RetrievalResult:
    node_id: str
    chunk_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    """A predicted citation as a structured evidence pointer, not a bare doc_id string.

    ``start_char``/``end_char`` are optional (byte-offset-free character
    offsets into the *chunk's* text, i.e. the same string as the matching
    ``RetrievalResult.text``) — populated only by a generator that can
    actually locate the supporting span (see
    ``inference.generators.extractive.CitationExtractiveGenerator``); every
    other generator leaves them ``None`` rather than fabricate a span. This
    is deliberately incremental: document-identity matching (what the old
    ``list[str]`` supported) still works via ``doc_id``/``chunk_id``, and
    claim-to-span entailment can be added later without another schema
    migration.
    """

    citation_id: str
    doc_id: str
    chunk_id: str
    start_char: int | None = None
    end_char: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BuiltContext:
    text: str
    results: list[RetrievalResult]
    citation_map: dict[str, RetrievalResult]
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation_map"] = {key: value.to_dict() for key, value in self.citation_map.items()}
        return data


@dataclass(slots=True)
class RAGAnswer:
    query: str
    answer: str
    contexts: list[RetrievalResult]
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationReport:
    grounded: bool
    citation_coverage: float
    evidence_count: int
    status: VerificationStatus = "run"
    unsupported_citations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalItem:
    question_id: str
    question: str
    ground_truth_answer: str | None = None
    expected_doc_ids: list[str] = field(default_factory=list)
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalItem:
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            ground_truth_answer=data.get("ground_truth_answer"),
            expected_doc_ids=list(data.get("expected_doc_ids", [])),
            expected_chunk_ids=list(data.get("expected_chunk_ids", [])),
            expected_citations=list(data.get("expected_citations", [])),
            metadata=dict(data.get("metadata", {})),
        )
