from __future__ import annotations

import pytest

from ragbench.core.schema import EvalItem
from ragbench.evaluation.profiles import validate_profile


def _item(question_id: str, **kwargs) -> EvalItem:
    return EvalItem(question_id=question_id, question="q", **kwargs)


def test_retrieval_profile_without_coverage_config_only_needs_one_qrel() -> None:
    """Backward-compatible default — unconfigured coverage behaves exactly
    like before this feature existed."""
    items = [_item("q1", expected_doc_ids=["d1"]), _item("q2"), _item("q3")]
    validate_profile("retrieval", items)  # does not raise


def test_retrieval_profile_rejects_low_coverage_ratio_when_configured() -> None:
    """Regression: a suite with 1 qrel out of 100 queries used to pass this
    check trivially. A claim-eligible suite declaring min_retrieval_coverage
    must actually enforce it."""
    items = [_item("q1", expected_doc_ids=["d1"])] + [_item(f"q{i}") for i in range(2, 21)]  # 1/20 = 5%
    with pytest.raises(ValueError, match="retrieval profile requires >= 95%"):
        validate_profile("retrieval", items, coverage={"min_retrieval_coverage": 0.95})


def test_retrieval_profile_accepts_sufficient_coverage_ratio() -> None:
    items = [_item(f"q{i}", expected_doc_ids=["d1"]) for i in range(19)] + [_item("q20")]  # 19/20 = 95%
    validate_profile("retrieval", items, coverage={"min_retrieval_coverage": 0.95})


def test_citation_rag_coverage_denominator_excludes_unanswerable_items() -> None:
    """An unanswerable item legitimately has no citation — it must not count
    against the citation-coverage ratio."""
    items = [
        _item("q1", expected_citations=["d1:c1"]),
        _item("q2", expected_citations=["d2:c1"]),
        _item("q3", metadata={"is_answerable": False}),  # no citation, but excluded from the denominator
    ]
    validate_profile("citation_rag", items, coverage={"min_citation_coverage": 1.0})


def test_citation_rag_coverage_rejects_missing_citation_on_an_answerable_item() -> None:
    items = [
        _item("q1", expected_citations=["d1:c1"]),
        _item("q2"),  # answerable (default) but missing an expected citation
    ]
    with pytest.raises(ValueError, match="citation_rag profile requires >= 100%"):
        validate_profile("citation_rag", items, coverage={"min_citation_coverage": 1.0})


def test_multi_hop_profile_enforces_a_minimum_question_count_not_just_one() -> None:
    items = [_item("q1", expected_doc_ids=["d1", "d2"])]  # only 1 multi-hop question
    with pytest.raises(ValueError, match="multi_hop_rag profile requires >= 5"):
        validate_profile("multi_hop_rag", items, coverage={"min_multi_hop_questions": 5})


def test_min_unanswerable_questions_enforced_when_configured() -> None:
    items = [_item("q1", expected_doc_ids=["d1"]), _item("q2", expected_doc_ids=["d2"])]  # both answerable by default
    with pytest.raises(ValueError, match="min_unanswerable_questions"):
        validate_profile("retrieval", items, coverage={"min_unanswerable_questions": 1})


def test_min_per_question_type_enforced_when_configured() -> None:
    items = [
        _item("q1", expected_doc_ids=["d1"], metadata={"question_type": "factual"}),
        _item("q2", expected_doc_ids=["d2"], metadata={"question_type": "citation_sensitive"}),
    ]
    with pytest.raises(ValueError, match=r"min_per_question_type\['factual'\]"):
        validate_profile("retrieval", items, coverage={"min_per_question_type": {"factual": 2}})
