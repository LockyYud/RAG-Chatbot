"""``protocol_split`` is declared by whoever runs the protocol, never derived.

The invariant these tests protect: a dataset's upstream split name and its
held-out status in this protocol are independent facts, and neither may be
inferred from the other. Upstream "validation" is a dev split under another
name; upstream "test" does not mean *this* project held it out. So adapters —
which describe upstream data — must never emit ``protocol_split``, and it must
stay absent unless a human asked for it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ragbench.datasets.adapters import registry
from ragbench.datasets.schema import (
    DocumentRecord,
    PreparedDataset,
    QrelRecord,
    QueryRecord,
    sample_processed_dataset,
    validate_protocol_split,
    write_prepared_dataset,
)

COMMITTED_DATASET = Path(__file__).resolve().parents[1] / "datasets" / "processed" / "vi_wiki_retrieval"


def _fake_dataset(**metadata: Any) -> PreparedDataset:
    return PreparedDataset(
        dataset_id="fake",
        documents=[
            DocumentRecord(doc_id="d1", title="t", text="alpha beta"),
            DocumentRecord(doc_id="d2", title="t", text="gamma delta"),
        ],
        queries=[QueryRecord(query_id="q1", question="alpha?"), QueryRecord(query_id="q2", question="gamma?")],
        qrels=[QrelRecord(query_id="q1", doc_id="d1"), QrelRecord(query_id="q2", doc_id="d2")],
        metadata={"split": "validation", "adapter": "fake", **metadata},
    )


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Register an offline adapter so preparation can be tested without HF."""
    calls: list[dict[str, Any]] = []

    def adapter(*, split: str | None = None, limit: int | None = None, seed: int = 42) -> PreparedDataset:
        calls.append({"split": split, "limit": limit, "seed": seed})
        return _fake_dataset()

    monkeypatch.setitem(registry.DATASET_ADAPTERS, "fake", adapter)
    return calls


def _manifest_metadata(output_dir: Path) -> dict[str, Any]:
    return dict(json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["metadata"])


def test_protocol_split_is_absent_unless_explicitly_requested(tmp_path: Path, fake_adapter) -> None:
    """An absent label means nobody asserted anything — it must not be
    defaulted to either value, since "dev" would block legitimate claims and
    "test" would manufacture a held-out guarantee nobody gave."""
    registry.prepare_dataset("fake", str(tmp_path / "out"))

    assert "protocol_split" not in _manifest_metadata(tmp_path / "out")


def test_protocol_split_is_recorded_when_requested(tmp_path: Path, fake_adapter) -> None:
    registry.prepare_dataset("fake", str(tmp_path / "out"), protocol_split="dev")

    assert _manifest_metadata(tmp_path / "out")["protocol_split"] == "dev"


def test_upstream_split_is_untouched_by_the_protocol_label(tmp_path: Path, fake_adapter) -> None:
    """The two fields must coexist without either overwriting the other."""
    registry.prepare_dataset("fake", str(tmp_path / "out"), split="validation", protocol_split="test")

    metadata = _manifest_metadata(tmp_path / "out")
    assert metadata["split"] == "validation"  # upstream provenance, unchanged
    assert metadata["protocol_split"] == "test"  # this protocol's own declaration
    assert fake_adapter[0]["split"] == "validation"  # only the upstream split reaches the adapter


def test_adapters_never_receive_the_protocol_label(tmp_path: Path, fake_adapter) -> None:
    """Regression guard on the layering: ``protocol_split`` is a decision made
    by whoever runs the protocol, so it is stamped above the adapter. An
    adapter cannot even see it, let alone act on it."""
    registry.prepare_dataset("fake", str(tmp_path / "out"), protocol_split="dev")

    assert "protocol_split" not in fake_adapter[0]
    assert "protocol_split" not in _fake_dataset().metadata


@pytest.fixture
def emitting_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A (buggy) adapter that asserts held-out status on its own."""

    def adapter(*, split: str | None = None, limit: int | None = None, seed: int = 42) -> PreparedDataset:
        return _fake_dataset(protocol_split="test")

    monkeypatch.setitem(registry.DATASET_ADAPTERS, "emitting", adapter)


def test_an_adapter_that_emits_the_protocol_label_is_rejected(tmp_path: Path, emitting_adapter) -> None:
    """Regression: the layering was only a convention. Nothing stopped an
    adapter from returning metadata.protocol_split, and prepare_dataset()
    passed it straight through whenever the caller did not supply the flag —
    so a new adapter shipping "test" would silently manufacture a held-out
    guarantee that nothing upstream can give."""
    with pytest.raises(ValueError, match="must not emit metadata.protocol_split"):
        registry.prepare_dataset("emitting", str(tmp_path / "out"))

    assert not (tmp_path / "out").exists(), "nothing may be written when the layering is violated"


def test_an_emitted_protocol_label_is_rejected_rather_than_silently_overwritten(
    tmp_path: Path, emitting_adapter
) -> None:
    """Even with an explicit flag, overwriting would hide the adapter bug — and
    the two values disagreeing ("test" from the adapter, "dev" from the
    operator) is precisely the case worth surfacing loudly."""
    with pytest.raises(ValueError, match="must not emit metadata.protocol_split"):
        registry.prepare_dataset("emitting", str(tmp_path / "out"), protocol_split="dev")


def test_no_bundled_adapter_mentions_the_protocol_label() -> None:
    """Static counterpart to the runtime guard above: catches a new adapter at
    test time without needing network access to actually run it."""
    adapters_dir = Path(registry.__file__).parent
    offenders = [
        path.name
        for path in sorted(adapters_dir.glob("*.py"))
        if path.name not in {"registry.py", "__init__.py"} and "protocol_split" in path.read_text(encoding="utf-8")
    ]

    assert not offenders, f"adapters must not reference protocol_split: {offenders}"


def test_invalid_protocol_split_is_rejected_before_any_work_happens(tmp_path: Path, fake_adapter) -> None:
    with pytest.raises(ValueError, match="protocol_split must be one of"):
        registry.prepare_dataset("fake", str(tmp_path / "out"), protocol_split="validation")

    assert not fake_adapter, "the adapter must not run when the label is invalid"
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", ["dev", "test", None])
def test_validate_protocol_split_accepts_the_declared_vocabulary(value: str | None) -> None:
    assert validate_protocol_split(value) == value


def test_a_sample_inherits_its_source_protocol_split(tmp_path: Path) -> None:
    """A sample of a tuning set is still a tuning set. Losing the label would
    let someone later mark the sample "test" without seeing that its parent had
    been tuned against."""
    source = tmp_path / "source"
    write_prepared_dataset(_fake_dataset(protocol_split="dev"), source)

    sample_processed_dataset(source, tmp_path / "sample", limit=1)

    assert _manifest_metadata(tmp_path / "sample")["protocol_split"] == "dev"


def test_a_sample_of_an_unlabelled_dataset_stays_unlabelled(tmp_path: Path) -> None:
    source = tmp_path / "source"
    write_prepared_dataset(_fake_dataset(), source)

    sample_processed_dataset(source, tmp_path / "sample", limit=1)

    assert "protocol_split" not in _manifest_metadata(tmp_path / "sample")


def test_committed_research_dataset_is_labelled_dev() -> None:
    """vi_wiki_retrieval is upstream VieQuAD *validation* and is the only
    labelled data in the repo, so it is the tuning set for the calibration
    pilot. Pinned here so the label cannot silently disappear or drift to
    "test" — a held-out claim set has to be a separate snapshot that was never
    used to choose config."""
    metadata = _manifest_metadata(COMMITTED_DATASET)

    assert metadata["protocol_split"] == "dev"
    assert metadata["split"] == "validation", "upstream provenance must stay recorded separately"


def test_labelling_the_committed_dataset_did_not_change_its_fingerprint() -> None:
    """The dataset fingerprint covers documents, queries, and qrels only, so
    relabelling costs nothing — no re-freeze of any suite that locks it."""
    from ragbench.datasets.schema import validate_processed_dataset

    manifest = json.loads((COMMITTED_DATASET / "manifest.json").read_text(encoding="utf-8"))

    assert validate_processed_dataset(COMMITTED_DATASET)["fingerprint"] == manifest["fingerprint"]
