from __future__ import annotations

import importlib
import random
from collections.abc import Iterable, Sequence
from typing import Any


def require_datasets() -> Any:
    try:
        module = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Dataset adapters require the optional research dependencies. "
            "Run with `uv run --extra research ...`, install with `uv sync --extra research`, "
            "or use `pip install -e .[research]`."
        ) from exc
    if not callable(getattr(module, "load_dataset", None)):
        raise RuntimeError(
            "Dataset adapters require Hugging Face `datasets`, but it is not installed; "
            "the local `datasets/` fixture directory was imported instead. "
            "Run with `uv run --extra research ...`, install with `uv sync --extra research`, "
            "or use `pip install -e .[research]`."
        )
    return module


def load_hf_dataset(repo_id: str, *args: Any, **kwargs: Any) -> Any:
    return require_datasets().load_dataset(repo_id, *args, **kwargs)


def rows_from_split(dataset: Any, split: str | None = None) -> list[dict[str, Any]]:
    if isinstance(dataset, dict):
        selected_split = split or _first_split(dataset)
        return [dict(row) for row in dataset[selected_split]]
    return [dict(row) for row in dataset]


def limit_rows(rows: Sequence[dict[str, Any]], limit: int | None, seed: int = 42) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows]
    if limit is None or limit <= 0 or len(selected) <= limit:
        return selected
    random.Random(seed).shuffle(selected)
    return selected[:limit]


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, list):
                parts.append(" ".join(str(part) for part in item))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def answer_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, list):
            return "; ".join(str(item) for item in text if item)
        if text is not None:
            return str(text)
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item)
    return str(value)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def first_present(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _first_split(dataset: dict[str, Any]) -> str:
    for candidate in ("validation", "test", "train", "default"):
        if candidate in dataset:
            return candidate
    return next(iter(dataset))
