"""Machine-readable contract for a bundled research technique.

The implementation remains code-first in ``pipeline.py``.  This small spec
only exposes the boundaries the runner must understand: supported evaluation
profiles, external requirements, and whether a technique writes state beyond
the canonical nodes artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EVALUATION_PROFILES = frozenset({"retrieval", "single_hop_rag", "multi_hop_rag", "citation_rag"})
IMPLEMENTATION_LEVELS = frozenset(
    {"baseline", "faithful_reproduction", "paper_inspired", "production_pattern", "concept_only"}
)


@dataclass(frozen=True, slots=True)
class TechniqueSpec:
    id: str
    implementation_level: str
    evaluation_profiles: frozenset[str]
    requirements: frozenset[str]
    custom_artifacts: bool


def technique_spec(metadata: dict[str, Any]) -> TechniqueSpec:
    """Validate and normalize the portable portion of ``technique.yaml``."""
    technique_id = metadata.get("id")
    if not isinstance(technique_id, str) or not technique_id:
        raise ValueError("technique metadata requires a non-empty id")
    implementation = metadata.get("implementation", {})
    level = implementation.get("level") if isinstance(implementation, dict) else None
    if level not in IMPLEMENTATION_LEVELS:
        raise ValueError(f"{technique_id}: invalid implementation.level {level!r}")
    capabilities = metadata.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise ValueError(f"{technique_id}: capabilities must be an object")
    profiles = capabilities.get("evaluation_profiles", ["retrieval", "single_hop_rag"])
    if not isinstance(profiles, list) or not profiles or not all(isinstance(item, str) for item in profiles):
        raise ValueError(f"{technique_id}: capabilities.evaluation_profiles must be a non-empty string list")
    invalid_profiles = sorted(set(profiles) - EVALUATION_PROFILES)
    if invalid_profiles:
        raise ValueError(f"{technique_id}: unsupported evaluation profiles: {', '.join(invalid_profiles)}")
    requirements = metadata.get("requires", [])
    if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
        raise ValueError(f"{technique_id}: requires must be a string list")
    custom_artifacts = capabilities.get("custom_artifacts", False)
    if not isinstance(custom_artifacts, bool):
        raise ValueError(f"{technique_id}: capabilities.custom_artifacts must be boolean")
    return TechniqueSpec(
        id=technique_id,
        implementation_level=level,
        evaluation_profiles=frozenset(profiles),
        requirements=frozenset(requirements),
        custom_artifacts=custom_artifacts,
    )
