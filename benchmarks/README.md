# Benchmarks

Benchmark outputs live in `benchmarks/results/`.

The directory is ignored by git because results depend on local datasets, model choices, API providers, and pricing configuration.

Run a single technique:

```bash
python -m raglab.cli.main ingest \
  --config techniques/naive_rag/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/naive_rag

python -m raglab.cli.main eval \
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
  --output benchmarks/results/sample
```
