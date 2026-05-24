from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from raglab.core.pipeline import ingest


def test_cli_artifacts_inspect(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    ingest("techniques/parent_child/config.yaml", "datasets/sample/docs", str(artifact))
    completed = subprocess.run(
        [sys.executable, "-m", "raglab.cli.main", "artifacts", "inspect", "--artifact", str(artifact)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["node_count"] > 0
    assert payload["manifest"]["artifact_version"] == "2"
