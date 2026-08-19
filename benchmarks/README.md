# Benchmarks

Benchmark outputs live in `benchmarks/results/`.

Most of the directory is ignored by git because results depend on local datasets, model choices, API providers, and
pricing configuration. The tiny `sample_research` run is committed as a reproducible smoke report.

Run a single technique:

```bash
python -m ragbench.cli.main ingest \
  --config techniques/naive_rag/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/naive_rag

python -m ragbench.cli.main eval \
  --config techniques/naive_rag/config.yaml \
  --artifact artifacts/naive_rag \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/naive_rag_eval.json
```

Run multiple techniques:

```bash
python benchmarks/run_all.py \
  --techniques naive_rag parent_child \
  --docs datasets/sample/docs \
  --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/sample \
  --mode full_rag
```

Outputs:

- `<technique>_eval.json`: per-technique report with metrics, predictions, failures, cost summary, and run metadata.
- `summary.json`: machine-readable aggregate.
- `summary.csv`: spreadsheet-friendly aggregate.
- `summary.md`: README-friendly benchmark table.
