from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def iter_input_files(input_path: str | Path) -> list[Path]:
    """Return the sorted list of text/markdown files under *input_path*.

    Accepts either a single file or a directory.  Used by every pipeline's
    ``ingest()`` to walk the input documents.
    """
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Input document path does not exist: {source}")
    if source.is_file():
        return [source]
    allowed = {".txt", ".md", ".markdown"}
    files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in allowed)
    if not files:
        raise ValueError(f"No text or Markdown documents found under input path: {source}")
    return files
