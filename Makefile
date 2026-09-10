.PHONY: test lint typecheck bench-sample build wheel-smoke ci

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q --cov=ragbench --cov-fail-under=60

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy ragbench benchmarks scripts tests

bench-sample:
	$(PYTHON) benchmarks/run_all.py --techniques parent_child --docs datasets/sample/docs --qa datasets/sample/qa.jsonl --output /tmp/ragbench-bench-v02 --mode full_rag

build:
	# build/ is an incremental cache setuptools reuses across builds — a
	# leftover build/lib from before a package rename (or any other stale
	# state) can silently ship extra top-level packages in the wheel that
	# the current source tree and pyproject.toml would never produce on
	# their own. Cleaned every time so "the wheel matches the source tree"
	# is actually true, not just usually true.
	rm -rf build dist
	$(PYTHON) -m build --wheel --no-isolation
	$(PYTHON) scripts/check_wheel_contents.py

wheel-smoke: build
	@repo_dir="$$(pwd)"; smoke_dir="$$(mktemp -d)"; \
	$(PYTHON) -m venv "$$smoke_dir/venv"; \
	"$$smoke_dir/venv/bin/pip" install dist/*.whl; \
	cd /tmp; \
	"$$smoke_dir/venv/bin/ragbench" techniques list; \
	"$$smoke_dir/venv/bin/ragbench" ingest --technique parent_child --input "$$repo_dir/datasets/sample/docs" --output "$$smoke_dir/artifact"; \
	"$$smoke_dir/venv/bin/ragbench" query --technique parent_child --artifact "$$smoke_dir/artifact" --query "Điều kiện xét tuyển là gì?" --mode retrieval_only

ci: lint typecheck test bench-sample wheel-smoke
