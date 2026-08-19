from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from ragbench.core.base import load_pipeline


def test_cli_artifacts_inspect(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    completed = subprocess.run(
        [sys.executable, "-m", "ragbench.cli.main", "artifacts", "inspect", "--artifact", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["node_count"] > 0
    assert payload["manifest"]["artifact_version"] == "5"
