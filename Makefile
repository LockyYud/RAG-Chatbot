.PHONY: test lint typecheck bench-sample build wheel-smoke ci

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q --cov=raglab --cov=evaluation --cov-fail-under=60

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy raglab evaluation benchmarks tests techniques

bench-sample:
	$(PYTHON) benchmarks/run_all.py --techniques parent_child --docs datasets/sample/docs --qa datasets/sample/qa.jsonl --output /tmp/raglab-bench-v02 --mode full_rag

build:
	$(PYTHON) -m build --wheel --no-isolation

wheel-smoke: build
	@repo_dir="$$(pwd)"; smoke_dir="$$(mktemp -d)"; \
	$(PYTHON) -m venv "$$smoke_dir/venv"; \
	"$$smoke_dir/venv/bin/pip" install --no-deps dist/*.whl; \
	cd /tmp; \
	"$$smoke_dir/venv/bin/raglab" techniques list; \
	"$$smoke_dir/venv/bin/raglab" ingest --technique parent_child --input "$$repo_dir/datasets/sample/docs" --output "$$smoke_dir/artifact"; \
	"$$smoke_dir/venv/bin/raglab" query --technique parent_child --artifact "$$smoke_dir/artifact" --query "Điều kiện xét tuyển là gì?" --mode retrieval_only

ci: lint typecheck test bench-sample wheel-smoke
