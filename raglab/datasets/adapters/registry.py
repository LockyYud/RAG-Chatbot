from __future__ import annotations

from collections.abc import Callable

from raglab.datasets.schema import PreparedDataset, write_prepared_dataset

DatasetAdapter = Callable[..., PreparedDataset]


def _load_adapters() -> dict[str, DatasetAdapter]:
    from raglab.datasets.adapters.uit_viquad import prepare_uit_viquad
    from raglab.datasets.adapters.viequad_retrieval import prepare_viequad_retrieval
    from raglab.datasets.adapters.vietnamese_legal_documents import prepare_vietnamese_legal_documents
    from raglab.datasets.adapters.vietnamese_legal_qa_rag import prepare_vietnamese_legal_qa_rag
    from raglab.datasets.adapters.vimqa import prepare_vimqa
    from raglab.datasets.adapters.vnfinsqa import prepare_vnfinsqa

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
) -> dict:
    if dataset_name not in DATASET_ADAPTERS:
        names = ", ".join(sorted(DATASET_ADAPTERS))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {names}")
    dataset = DATASET_ADAPTERS[dataset_name](split=split, limit=limit, seed=seed)
    return write_prepared_dataset(dataset, output_dir, overwrite=overwrite)
