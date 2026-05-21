# Naive RAG Baseline

## Source

This is an engineering baseline, not a paper reproduction.

## Core Idea

Use the simplest end-to-end RAG path to establish a reference point for every dataset:

```text
documents -> fixed chunking -> local dense term-vector retrieval -> citation context -> extractive answer
```

The local dense retriever is only a zero-dependency smoke-test baseline. Use OpenAI-compatible embedding pipelines for real retrieval experiments.

## RAG Stage

- chunking
- retrieval
- context_construction
- generation

## Suitable Data

Use this baseline for:

- sanity checking new datasets
- validating pipeline wiring
- debugging citation/evaluation format

## Weak Cases

Do not use this baseline for empirical claims about dense retrieval or generation quality.

## How To Run

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
