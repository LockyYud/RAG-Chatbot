"""Task-specific evaluation contracts used by the shared runner."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raglab.core.schema import EvalItem

PROFILES = frozenset({"retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"})


def resolve_profile(requested: str, mode: str) -> str:
    if requested == "auto":
        return "retrieval" if mode == "retrieval_only" else "single_hop_rag"
    if requested not in PROFILES:
        raise ValueError(f"Unknown evaluation profile {requested!r}. Choose one of: {', '.join(sorted(PROFILES))}")
    if mode == "retrieval_only" and requested != "retrieval":
        raise ValueError("retrieval_only mode requires the retrieval evaluation profile")
    return requested


def validate_profile(profile: str, items: list[EvalItem]) -> None:
    if profile == "retrieval" and not any(item.expected_doc_ids or item.expected_chunk_ids for item in items):
        raise ValueError("retrieval profile requires expected_doc_ids or expected_chunk_ids")
    if profile == "citation_rag" and not any(item.expected_citations for item in items):
        raise ValueError("citation_rag profile requires expected_citations")
    if profile == "multi_hop_rag" and not any(
        len(item.expected_doc_ids) + len(item.expected_chunk_ids) >= 2 for item in items
    ):
        raise ValueError("multi_hop_rag profile requires at least one item with two or more expected evidence IDs")
