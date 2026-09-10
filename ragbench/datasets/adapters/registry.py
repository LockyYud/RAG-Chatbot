from __future__ import annotations

from collections.abc import Callable

from ragbench.datasets.schema import PreparedDataset, validate_protocol_split, write_prepared_dataset

DatasetAdapter = Callable[..., PreparedDataset]


def _load_adapters() -> dict[str, DatasetAdapter]:
    from ragbench.datasets.adapters.uit_viquad import prepare_uit_viquad
    from ragbench.datasets.adapters.viequad_retrieval import prepare_viequad_retrieval
    from ragbench.datasets.adapters.vietnamese_legal_documents import prepare_vietnamese_legal_documents
    from ragbench.datasets.adapters.vietnamese_legal_qa_rag import prepare_vietnamese_legal_qa_rag
    from ragbench.datasets.adapters.vimqa import prepare_vimqa
    from ragbench.datasets.adapters.vnfinsqa import prepare_vnfinsqa

    return {
        "viequad_retrieval": prepare_viequad_retrieval,
        "uit_viquad": prepare_uit_viquad,
        "vietnamese_legal_documents": prepare_vietnamese_legal_documents,
        "vietnamese_legal_qa_rag": prepare_vietnamese_legal_qa_rag,
        "vimqa": prepare_vimqa,
        "vnfinsqa": prepare_vnfinsqa,
    }


DATASET_ADAPTERS = _load_adapters()


def prepare_dataset(
    dataset_name: str,
    output_dir: str,
    split: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    overwrite: bool = False,
    protocol_split: str | None = None,
) -> dict:
    """Prepare a fixed research dataset snapshot.

    ``split`` selects the *upstream* split to pull from and is passed to the
    adapter. ``protocol_split`` is a different kind of statement — whether this
    snapshot is a tuning set or a held-out claim set in this protocol — so it is
    stamped here rather than by the adapter: it is a decision made by whoever
    runs the protocol, not a property of the upstream dataset. Adapters
    therefore never emit it, and it stays absent unless explicitly requested.
    """
    if dataset_name not in DATASET_ADAPTERS:
        names = ", ".join(sorted(DATASET_ADAPTERS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {names}")
    validate_protocol_split(protocol_split)
    dataset = DATASET_ADAPTERS[dataset_name](split=split, limit=limit, seed=seed)
    if "protocol_split" in dataset.metadata:
        # Enforced, not merely conventional: an adapter that emits this is
        # asserting held-out status as a property of the upstream dataset,
        # which nothing upstream can know. Rejected even when the caller also
        # passed the flag — silently overwriting would hide the bug, and an
        # adapter shipping "test" that nobody notices is exactly how a tuning
        # set becomes a claim set.
        raise ValueError(
            f"Dataset adapter '{dataset_name}' must not emit metadata.protocol_split "
            f"(got {dataset.metadata['protocol_split']!r}). Held-out status is declared by whoever "
            "runs the protocol, via prepare_dataset(protocol_split=...) / --protocol-split. Adapters "
            "record upstream provenance only — use metadata.split for the upstream split name."
        )
    if protocol_split is not None:
        dataset.metadata["protocol_split"] = protocol_split
    return write_prepared_dataset(dataset, output_dir, overwrite=overwrite)
