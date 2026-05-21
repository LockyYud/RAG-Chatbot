from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BlockType = Literal["title", "heading", "paragraph", "table", "figure", "list", "footnote"]


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
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationReport:
    grounded: bool
    citation_coverage: float
    evidence_count: int
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
    def from_dict(cls, data: dict[str, Any]) -> "EvalItem":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            ground_truth_answer=data.get("ground_truth_answer"),
            expected_doc_ids=list(data.get("expected_doc_ids", [])),
            expected_chunk_ids=list(data.get("expected_chunk_ids", [])),
            expected_citations=list(data.get("expected_citations", [])),
            metadata=dict(data.get("metadata", {})),
        )
