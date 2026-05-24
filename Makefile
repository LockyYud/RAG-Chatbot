.PHONY: test lint typecheck bench-sample ci

PYTHON ?= python

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy raglab evaluation benchmarks tests

bench-sample:
	$(PYTHON) benchmarks/run_all.py --techniques naive_rag parent_child --docs datasets/sample/docs --qa datasets/sample/qa.jsonl --output benchmarks/results/sample --mode full_rag

ci: lint typecheck test
