"""Task-specific evaluation contracts used by the shared runner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragbench.core.schema import EvalItem

PROFILES = frozenset({"retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"})


def resolve_profile(requested: str, mode: str) -> str:
    if requested == "auto":
        return "retrieval" if mode == "retrieval_only" else "single_hop_rag"
    if requested not in PROFILES:
        raise ValueError(f"Unknown evaluation profile {requested!r}. Choose one of: {', '.join(sorted(PROFILES))}")
    if mode == "retrieval_only" and requested != "retrieval":
        raise ValueError("retrieval_only mode requires the retrieval evaluation profile")
    return requested


def _evidence_count(item: EvalItem) -> int:
    return len(item.expected_doc_ids) + len(item.expected_chunk_ids)


def _is_answerable(item: EvalItem) -> bool:
    return item.metadata.get("is_answerable") is not False


def _require_coverage(covered: int, total: int, minimum_ratio: float | None, profile: str, what: str) -> None:
    """Without ``minimum_ratio`` (unconfigured coverage — smoke/exploratory use),
    only "at least one qualifying item" is required, same as before this
    function existed. A configured ratio (e.g. 0.95 from a claim-eligible
    suite's ``coverage`` block) enforces the real thing a per-slice research
    claim needs: not "one query happened to have a qrel", but "the overwhelming
    majority of the retrieval slice does."
    """
    if minimum_ratio is None:
        if covered < 1:
            raise ValueError(f"{profile} profile requires at least one item with {what}")
        return
    ratio = covered / total if total else 0.0
    if ratio < minimum_ratio:
        raise ValueError(
            f"{profile} profile requires >= {minimum_ratio:.0%} of items to have {what}; "
            f"got {covered}/{total} ({ratio:.1%})"
        )


def _require_count(covered: int, minimum: int, label: str, what: str) -> None:
    if covered < minimum:
        raise ValueError(f"{label} requires >= {minimum} {what}; got {covered}")


def validate_profile(profile: str, items: list[EvalItem], *, coverage: dict[str, Any] | None = None) -> None:
    """Validate a profile's minimum data requirements.

    Without ``coverage`` this checks the same thing it always has: "at least
    one qualifying item" — good enough for a smoke test, too weak for a
    claim-eligible research benchmark (a suite could satisfy the old check
    with 1 qrel out of 500 queries). Pass ``coverage`` — a claim-eligible
    suite's own ``coverage`` block (see ``suites.load_suite``) — to enforce
    real per-slice thresholds instead:

        min_retrieval_coverage: 0.95      # ratio of items with a qrel
        min_citation_coverage: 1.0        # ratio of *answerable* items with an expected citation
        min_multi_hop_questions: 20       # count, not ratio
        min_unanswerable_questions: 10    # count
        min_per_question_type: {factual: 20, citation_sensitive: 20}
    """
    coverage = coverage or {}
    total = len(items)
    if profile == "retrieval":
        covered = sum(1 for item in items if item.expected_doc_ids or item.expected_chunk_ids)
        _require_coverage(
            covered, total, coverage.get("min_retrieval_coverage"), "retrieval", "a qrel (expected_doc_ids/chunk_ids)"
        )
    if profile == "citation_rag":
        # Unanswerable items legitimately carry no citation — the coverage
        # denominator is the answerable slice, not every item in the dataset.
        answerable_items = [item for item in items if _is_answerable(item)]
        covered = sum(1 for item in answerable_items if item.expected_citations)
        _require_coverage(
            covered,
            len(answerable_items),
            coverage.get("min_citation_coverage"),
            "citation_rag",
            "an expected citation",
        )
    if profile == "multi_hop_rag":
        covered = sum(1 for item in items if _evidence_count(item) >= 2)
        _require_count(
            covered,
            int(coverage.get("min_multi_hop_questions", 1)),
            "multi_hop_rag profile",
            "item(s) with two or more expected evidence IDs",
        )
    min_unanswerable = coverage.get("min_unanswerable_questions")
    if min_unanswerable is not None:
        covered = sum(1 for item in items if not _is_answerable(item))
        _require_count(
            covered,
            int(min_unanswerable),
            "suite.coverage.min_unanswerable_questions",
            "unanswerable question(s) (metadata.is_answerable=false)",
        )
    for question_type, minimum in (coverage.get("min_per_question_type") or {}).items():
        covered = sum(1 for item in items if item.metadata.get("question_type") == question_type)
        _require_count(
            covered,
            int(minimum),
            f"suite.coverage.min_per_question_type[{question_type!r}]",
            f"question(s) with question_type={question_type!r}",
        )
