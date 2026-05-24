from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from raglab.providers.env import load_dotenv


class ConfigError(ValueError):
    pass


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


def validate_config(config: dict[str, Any], *, for_ingest: bool = False, for_query: bool = False) -> None:
    if not isinstance(config.get("processing", {}), dict):
        raise ConfigError("processing must be a mapping")
    if for_ingest:
        _validate_stage(config, "processing.parser")
        _validate_stage(config, "processing.chunker")
        chunker = get_stage(config, "processing.chunker", {})
        params = dict(chunker.get("params", {})) if isinstance(chunker, dict) else {}
        for key in ("chunk_size", "overlap", "child_size", "child_overlap"):
            if key in params and int(params[key]) < 0:
                raise ConfigError(f"processing.chunker.params.{key} must be >= 0")
    if for_query:
        _validate_stage(config, "inference.retriever")
        _validate_stage(config, "inference.context_builder")
        _validate_stage(config, "inference.generator")
        retriever = get_stage(config, "inference.retriever", {})
        params = dict(retriever.get("params", {})) if isinstance(retriever, dict) else {}
        if "top_k" in params and int(params["top_k"]) <= 0:
            raise ConfigError("inference.retriever.params.top_k must be > 0")
    embedding = get_stage(config, "indexing.embedding")
    if isinstance(embedding, dict) and embedding.get("type") == "openai":
        model = str(dict(embedding.get("params", {})).get("model", "")).strip()
        if not model:
            raise ConfigError("indexing.embedding.params.model is required for openai embeddings")
    store = get_stage(config, "indexing.store")
    if isinstance(store, dict) and store.get("type") not in {None, "json_memory", "faiss_local"}:
        raise ConfigError("indexing.store.type must be json_memory or faiss_local")


def _validate_stage(config: dict[str, Any], path: str) -> None:
    stage = get_stage(config, path)
    if stage is None:
        raise ConfigError(f"Missing required config stage: {path}")
    if isinstance(stage, dict) and not stage.get("type"):
        raise ConfigError(f"{path}.type is required")


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
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError(f"List item without list parent: {raw_line}")
            item_text = line[2:].strip()
            if ":" in item_text:
                key, value = item_text.split(":", 1)
                item: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_text))
            continue

        if ":" not in line:
            raise ConfigError(f"Unsupported config line: {raw_line}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ConfigError(f"Mapping item inside list without object: {raw_line}")

        if value:
            parent[key] = _parse_scalar(value)
            continue

        next_container: dict[str, Any] | list[Any]
        next_container = []
        parent[key] = next_container
        stack.append((indent, next_container))

    _fix_empty_lists(root)
    return root


def _fix_empty_lists(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if isinstance(child, list) and not child:
                value[key] = {}
            else:
                _fix_empty_lists(child)
    elif isinstance(value, list):
        for child in value:
            _fix_empty_lists(child)


def get_stage(config: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


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
        return os.getenv(name, value)
    return value
