"""Fail the build if the wheel ships anything beyond the `ragbench` namespace.

`python -m build` reuses `build/lib` as an incremental cache — a wheel built
without cleaning that directory first can silently ship whatever packages
were there from a previous build (e.g. the pre-rename `raglab`/`evaluation`/
`techniques` top-level packages), even though `pyproject.toml`'s
`packages.find` config is correct and the source tree is clean. This check
catches that class of contamination directly from the built artifact, not
from the source tree (which would never have shown the problem).

Usage: run after `python -m build --wheel` (see Makefile's `build` target,
which cleans `build/`/`dist/` first, and .github/workflows/ci.yml, which
always builds from a fresh checkout).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ALLOWED_TOP_LEVEL_PREFIXES = ("ragbench", "rag_pipeline_lab-")


def main() -> int:
    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        print("No wheel found in dist/ — run `python -m build --wheel` first.", file=sys.stderr)
        return 1
    wheel_path = wheels[-1]

    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
        top_level = sorted({name.split("/", 1)[0] for name in names})
        unexpected = [entry for entry in top_level if not entry.startswith(ALLOWED_TOP_LEVEL_PREFIXES)]
        if unexpected:
            print(
                f"{wheel_path} ships unexpected top-level entries: {unexpected} "
                f"(only {ALLOWED_TOP_LEVEL_PREFIXES} are expected — did you build without "
                "cleaning build/ first?)",
                file=sys.stderr,
            )
            return 1

        entry_points_name = next((name for name in names if name.endswith("entry_points.txt")), None)
        if entry_points_name is None:
            print(f"{wheel_path} has no entry_points.txt — expected the `ragbench` console script.", file=sys.stderr)
            return 1
        entry_points = archive.read(entry_points_name).decode("utf-8")
        if "ragbench = ragbench.cli.main:main" not in entry_points:
            print(
                f"{wheel_path}'s entry_points.txt is missing the ragbench console script:\n{entry_points}",
                file=sys.stderr,
            )
            return 1
        if "raglab" in entry_points:
            print(
                f"{wheel_path}'s entry_points.txt still references the old raglab command:\n{entry_points}",
                file=sys.stderr,
            )
            return 1

    print(f"{wheel_path}: top-level contents OK ({top_level}), console script OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
