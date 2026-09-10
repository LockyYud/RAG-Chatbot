"""Minimal YAML/JSON loader used to read ``technique.yaml`` paper metadata.

The repo used to load full pipeline configs from YAML; that responsibility has
moved into each technique's ``pipeline.py``. The only YAML left is paper
metadata (title, authors, tags) — small, flat, and parsed by the same
function as before. ``${VAR:-default}`` substitution is still supported in
case anyone embeds env-driven values in metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ragbench.providers.env import load_dotenv


class ConfigError(ValueError):
    """Raised when a YAML/JSON metadata file cannot be parsed."""


def load_config(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        return _resolve_env(json.loads(text))
    except json.JSONDecodeError:
        pass
    if config_path.suffix.lower() == ".json":
        return _resolve_env(json.loads(text))
    try:
        import yaml

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return _resolve_env(loaded)
    except ImportError:
        pass
    return _resolve_env(_parse_minimal_yaml(text))


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    # Collected up front (not consumed line-by-line) so an empty-value key
    # can look ahead at its first child to decide whether it opens a mapping
    # or a list — deciding eagerly (always assuming list) is what used to
    # make ``key:\n  nested_key: value`` raise instead of parsing.
    entries = [
        (len(raw_line) - len(raw_line.lstrip(" ")), raw_line.strip())
        for raw_line in text.splitlines()
        if raw_line.strip() and not raw_line.lstrip().startswith("#")
    ]
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for index, (indent, line) in enumerate(entries):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError(f"List item without list parent: {line}")
            item_text = line[2:].strip()
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                item: dict[str, Any] = {}
                parent.append(item)
                stack.append((indent, item))
                _assign_or_open(item, key.strip(), value.strip(), entries, index, indent, stack)
            else:
                parent.append(_parse_scalar(item_text))
            continue

        if ":" not in line:
            raise ConfigError(f"Unsupported config line: {line}")

        key, value = line.split(":", 1)
        if not isinstance(parent, dict):
            raise ConfigError(f"Mapping item inside list without object: {line}")
        _assign_or_open(parent, key.strip(), value.strip(), entries, index, indent, stack)

    return root


def _assign_or_open(
    parent: dict[str, Any],
    key: str,
    value: str,
    entries: list[tuple[int, str]],
    index: int,
    indent: int,
    stack: list[tuple[int, dict[str, Any] | list[Any]]],
) -> None:
    """Assign a scalar, or — for an empty value — open the mapping/list its
    first deeper-indented child implies (empty ``{}`` if it has no child at
    all, since there is nothing left to look ahead at by then)."""
    if value:
        parent[key] = _parse_scalar(value)
        return
    next_entry = entries[index + 1] if index + 1 < len(entries) else None
    if next_entry is None or next_entry[0] <= indent:
        parent[key] = {}
        return
    next_container: dict[str, Any] | list[Any] = [] if next_entry[1].startswith("- ") else {}
    parent[key] = next_container
    stack.append((indent, next_container))


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_resolve_env(child) for child in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name_default = value[2:-1]
        if ":-" in name_default:
            name, default = name_default.split(":-", 1)
            return os.getenv(name, default)
        return os.getenv(name_default, value)
    return value
