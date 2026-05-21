# RAG Pipeline Lab

`rag-pipeline-lab` is a runnable MVP of a full-stack RAG strategy lab. It is designed to configure, run, and benchmark end-to-end RAG pipelines from raw documents to grounded answers, instead of demonstrating isolated RAG techniques one by one.

The project has two execution paths:

- **Local baseline path**: zero-dependency, useful for inspecting architecture, testing pipeline orchestration, and validating reports.
- **Real RAG path**: OpenAI-compatible embeddings and chat generation, useful for actual strategy experiments.

The local baseline is not a meaningful empirical benchmark for dense retrieval or answer quality. Use `configs/pipelines/openai_rag.yaml` when you want real embedding and generation behavior.

## What It Benchmarks

The lab compares complete pipeline strategies across:

- document parsing and cleaning
- fixed, recursive, heading-aware, and parent-child chunking
- section-title enrichment
- BM25, term-vector dense baseline, OpenAI embedding retrieval, and hybrid retrieval
- lexical-overlap reranking
- citation-aware context construction
- extractive fallback generation and OpenAI-compatible chat generation with required citations
- citation coverage verification
- Recall@k, MRR, hit rate, citation accuracy, latency, context tokens, and local cost estimates

## Pipeline Shape

```text
raw documents
  -> parser
  -> cleaners
  -> chunker
  -> enricher
  -> indexed nodes / artifacts
  -> retriever
  -> reranker
  -> context builder
  -> generator
  -> verifier
  -> metrics / reports
```

## Included Strategies

| Pipeline | Chunking | Retrieval | Reranking | Generation | Verification |
| --- | --- | --- | --- | --- | --- |
| `naive_dense` | fixed-size | dense term-vector baseline | none | citation extractive | citation coverage |
| `heading_hybrid` | heading-aware | hybrid dense + BM25 | lexical overlap | citation extractive | citation coverage |
| `parent_child_bm25` | parent-child | BM25 | lexical overlap | citation extractive | citation coverage |
| `openai_rag` | heading-aware | OpenAI embeddings + BM25 hybrid | lexical overlap | OpenAI-compatible chat | citation coverage |

## Sample Benchmark

Current sample reports use the included Vietnamese documents and QA dataset.

| Pipeline | Recall@5 | MRR | Citation Accuracy | Avg Latency ms | Avg Context Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `naive_dense` | 1.000 | 1.000 | 0.333 | 0.550 | 354.0 |
| `heading_hybrid` | 1.000 | 1.000 | 1.000 | 0.738 | 286.3 |
| `parent_child_bm25` | 1.000 | 1.000 | 1.000 | 0.618 | 286.3 |

The dataset is intentionally small, so these numbers should be read as a smoke benchmark and a comparison fixture, not as a general RAG leaderboard.

## Real OpenAI-Compatible Path

Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

Then build real embedding artifacts and query with a real model:

```bash
python -m raglab.cli.main ingest \
  --config configs/pipelines/openai_rag.yaml \
  --input datasets/sample_docs \
  --output artifacts/openai_rag

python -m raglab.cli.main query \
  --config configs/pipelines/openai_rag.yaml \
  --artifact artifacts/openai_rag \
  --query "Điều kiện xét tuyển ngành trí tuệ nhân tạo là gì?"
```

For meaningful strategy comparison, replace `datasets/benchmark_qa/qa.jsonl` with at least 50-100 labeled QA pairs. The included `datasets/sample_qa/qa.jsonl` has only 3 rows and is strictly for smoke tests.

## Quick Start

```bash
python -m raglab.cli.main ingest \
  --config configs/pipelines/heading_hybrid.yaml \
  --input datasets/sample_docs \
  --output artifacts/heading_hybrid

python -m raglab.cli.main query \
  --config configs/pipelines/heading_hybrid.yaml \
  --artifact artifacts/heading_hybrid \
  --query "Điều kiện xét tuyển ngành trí tuệ nhân tạo là gì?"

python -m raglab.cli.main eval \
  --config configs/pipelines/heading_hybrid.yaml \
  --artifact artifacts/heading_hybrid \
  --dataset datasets/sample_qa/qa.jsonl \
  --output reports/heading_hybrid_eval.json
```

Compare multiple strategies:

```bash
python -m raglab.cli.main compare \
  --runs \
    configs/pipelines/naive_dense.yaml=artifacts/naive_dense \
    configs/pipelines/heading_hybrid.yaml=artifacts/heading_hybrid \
    configs/pipelines/parent_child_bm25.yaml=artifacts/parent_child_bm25 \
  --dataset datasets/sample_qa/qa.jsonl \
  --output reports/comparison.json
```

## Development

Install test dependencies when needed:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

The config files are JSON-compatible YAML files. This keeps the project dependency-free while preserving the familiar pipeline config layout.

## Scope

This repo is a portfolio-ready MVP, not a production RAG stack. The local reports are architecture smoke tests. Empirical claims should use OpenAI-compatible embeddings/chat plus a larger labeled dataset. Planned extensions include vector stores, LLM reranking, PDF layout parsing, semantic chunking, HyDE or multi-query retrieval, and LLM-as-judge verification.
