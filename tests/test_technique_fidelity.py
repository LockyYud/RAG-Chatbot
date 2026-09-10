from __future__ import annotations

from typing import Any

import pytest

from ragbench.core.base import list_pipelines
from ragbench.core.spec import technique_spec


def _metadata(**implementation_overrides: Any) -> dict[str, Any]:
    return {
        "id": "t1",
        "implementation": {"level": "baseline", "status": "runnable", **implementation_overrides},
        "capabilities": {"evaluation_profiles": ["retrieval"]},
    }


def test_baseline_level_does_not_require_fidelity_documentation() -> None:
    technique_spec(_metadata(level="baseline"))  # does not raise


def test_paper_inspired_level_requires_reproduced_or_omitted() -> None:
    """Regression: a technique claiming anything beyond "baseline" (i.e. some
    relationship to a paper) used to be able to declare that with zero
    documentation of what was actually reproduced vs. skipped — exactly the
    silent fidelity gap the paper-fidelity contract exists to close."""
    with pytest.raises(ValueError, match="must document at least one of"):
        technique_spec(_metadata(level="paper_inspired"))


def test_paper_inspired_level_is_satisfied_by_reproduced_alone() -> None:
    technique_spec(_metadata(level="paper_inspired", reproduced=["the core retrieval idea"]))


def test_paper_inspired_level_is_satisfied_by_omitted_alone() -> None:
    technique_spec(_metadata(level="paper_inspired", omitted=["training the retriever"]))


def test_fidelity_fields_must_be_string_lists() -> None:
    with pytest.raises(ValueError, match="implementation.reproduced must be a string list"):
        technique_spec(_metadata(level="paper_inspired", reproduced="not a list"))
    with pytest.raises(ValueError, match="implementation.omitted must be a string list"):
        technique_spec(_metadata(level="paper_inspired", omitted=[123]))


def test_every_bundled_technique_satisfies_the_fidelity_contract() -> None:
    """Conformance test: every technique shipped in this repo — not just a
    synthetic example — must pass technique_spec()'s fidelity check. This is
    exercised implicitly by list_pipelines() (which calls technique_spec()
    per technique), asserted here explicitly so a future regression fails
    with a clear message instead of an unrelated-looking test breaking."""
    for item in list_pipelines():
        spec = technique_spec(item)
        if spec.implementation_level != "baseline":
            implementation = item.get("implementation", {})
            assert implementation.get("reproduced") or implementation.get("omitted"), (
                f"{spec.id}: level={spec.implementation_level!r} but documents neither "
                "reproduced nor omitted"
            )
