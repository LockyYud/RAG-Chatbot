from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from raglab.core.config import load_config


def list_techniques(root: str | Path = "techniques") -> list[dict[str, Any]]:
    root_path = Path(root)
    techniques: list[dict[str, Any]] = []
    for path in sorted(root_path.glob("*/technique.yaml")):
        if path.parent.name.startswith("_"):
            continue
        metadata = load_config(path)
        metadata["_path"] = str(path.parent)
        metadata["_config_path"] = str(path.parent / metadata.get("config", "config.yaml"))
        techniques.append(metadata)
    return techniques


def get_technique(technique_id: str, root: str | Path = "techniques") -> dict[str, Any]:
    for technique in list_techniques(root):
        if technique.get("id") == technique_id or Path(technique["_path"]).name == technique_id:
            return technique
    known = ", ".join(item["id"] for item in list_techniques(root))
    raise KeyError(f"Unknown technique '{technique_id}'. Known: {known}")


def register_custom_for_config(config_path: str | Path) -> None:
    technique_dir = Path(config_path).resolve().parent
    metadata_path = technique_dir / "technique.yaml"
    if not metadata_path.exists():
        return
    metadata = load_config(metadata_path)
    custom_register = metadata.get("custom_register")
    if not custom_register:
        return
    register_path = technique_dir / custom_register
    if not register_path.exists():
        return

    custom_path = str(register_path.parent)
    if custom_path not in sys.path:
        sys.path.insert(0, custom_path)

    spec = importlib.util.spec_from_file_location(f"raglab_technique_{metadata['id']}_register", register_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load custom register file: {register_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "register"):
        raise AttributeError(f"{register_path} must define register(registry)")

    from raglab.core import registry

    module.register(registry)
