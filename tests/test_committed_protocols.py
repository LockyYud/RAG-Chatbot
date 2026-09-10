"""Every benchmark protocol file committed to this repo must actually load.

Regression: ``load_suite()`` gained a decision-rule contract for
claim-eligible suites (``primary_metric`` + ``minimum_effect``, or
``pareto_improvement``) without the one committed claim-eligible suite being
updated to satisfy it. The suite was invalid against its own validator and
nothing caught it, because no test — and no CI step — ever loaded a file from
``ragbench/evaluation/protocol/``. CI validated a dataset fixture, ran a
benchmark smoke on inline paths, and built the wheel; the committed research
protocol was never exercised at all.

These tests close that gap generically: tightening a validator now breaks the
suite of tests rather than only breaking the next person who runs a real
benchmark.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragbench.benchmarks.suites import load_suite
from ragbench.core.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "ragbench" / "evaluation" / "protocol"

# Files in protocol/ that are deliberately *not* runnable benchmark suites.
# ``standard.yaml`` is a metrics reference document (``required_dataset_fields``,
# ``metrics``, ``minimum_recommended_questions``) with an entirely different
# schema — it is not, and should not be made, loadable by ``load_suite()``.
#
# The allowlist is explicit on purpose: selecting suites by shape (e.g. "has a
# tier key") would let a real suite that *forgot* ``tier`` be silently skipped,
# which is the same class of failure this module exists to prevent.
NON_SUITE_DOCUMENTS = {"standard.yaml"}


def _protocol_files() -> list[Path]:
    return sorted(PROTOCOL_DIR.glob("*.yaml"))


def _suite_files() -> list[Path]:
    return [path for path in _protocol_files() if path.name not in NON_SUITE_DOCUMENTS]


def test_protocol_directory_is_not_empty() -> None:
    """Guards the rest of this module: a parametrization over an empty glob
    passes vacuously, so a moved/renamed protocol directory would silently
    turn every check below into a no-op."""
    assert _protocol_files(), f"no protocol files found under {PROTOCOL_DIR}"


@pytest.mark.parametrize("suite_path", _suite_files(), ids=lambda path: path.name)
def test_committed_suite_loads(suite_path: Path) -> None:
    suite = load_suite(suite_path)
    assert suite["id"]
    assert suite["tier"] in {"smoke_only", "exploratory", "claim_eligible"}


@pytest.mark.parametrize("suite_path", _suite_files(), ids=lambda path: path.name)
def test_committed_suite_dataset_paths_exist(suite_path: Path) -> None:
    """A suite that loads but points at a moved dataset is still broken — it
    just fails later, during a real benchmark run, instead of at load time."""
    dataset = load_suite(suite_path)["dataset"]
    for key in ("docs", "qa"):
        target = REPO_ROOT / str(dataset[key])
        assert target.exists(), f"{suite_path.name}: dataset.{key} does not exist: {target}"


@pytest.mark.parametrize("document_name", sorted(NON_SUITE_DOCUMENTS))
def test_allowlisted_non_suite_document_still_exists_and_is_not_a_suite(document_name: str) -> None:
    """Keeps the allowlist honest in both directions.

    A stale entry (file renamed or deleted) would silently excuse nothing,
    and a document that later grows a ``tier`` key is a suite that is being
    skipped rather than validated — both are caught here instead of quietly
    shrinking this module's coverage.
    """
    path = PROTOCOL_DIR / document_name
    assert path.exists(), f"allowlisted non-suite document no longer exists: {path}"
    payload = load_config(path)
    assert "tier" not in payload, (
        f"{document_name} declares a tier, so it is a benchmark suite — "
        "remove it from NON_SUITE_DOCUMENTS so load_suite() validates it."
    )
