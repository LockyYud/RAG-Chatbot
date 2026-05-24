# RAG Pipeline Lab

`rag-pipeline-lab` is a scalable, evaluation-first RAG research base for comparing retrieval, chunking,
generation, verification, and evaluation strategies under one reproducible protocol.

The repo is designed as a framework, not a notebook collection. A technique owns its metadata, config, optional custom
code, and benchmark report. The engine keeps stable interfaces for processing, indexing, retrieval, context building,
generation, verification, and evaluation so new research ideas can be added without rewriting the whole pipeline.

## What It Is

- A config-driven RAG pipeline engine with stable component registries.
- A benchmark harness that writes per-run reports plus JSON, CSV, and Markdown summaries.
- A local smoke path that runs without API keys.
- An OpenAI-compatible research path for embeddings, generation, synthetic QA, and LLM-as-judge metrics.
- An artifact format with manifest versioning, config hashes, embedding model metadata, and vector store metadata.

## Architecture

```text
raw docs
  -> parser / cleaner / chunker / enricher
  -> embedder
  -> vector store artifact
  -> retriever / reranker
  -> context builder
  -> generator
  -> verifier
  -> evaluator / judge / benchmark report
```

Main package layout:

```text
raglab/
  core/        # schema, interfaces, registry, config, pipeline orchestration
  processing/  # parsers, cleaners, chunkers, enrichers
  indexing/    # embeddings, retrievers, artifacts, vector stores
  inference/   # context builders, generators, rerankers, verifiers
  datasets/    # synthetic QA generation
  cli/         # developer-facing CLI

evaluation/    # retrieval metrics, judge metrics, eval runner
benchmarks/    # reproducible multi-technique benchmark runner
techniques/    # baseline and paper-inspired technique configs
```

## Quickstart

Local path, no API key:

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

Run a reproducible local benchmark:

```bash
python -m raglab.cli.main bench \
  --techniques naive_rag parent_child \
  --docs datasets/sample/docs \
  --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/sample \
  --mode full_rag
```

Inspect an artifact:

```bash
python -m raglab.cli.main artifacts inspect --artifact artifacts/naive_rag
```

## OpenAI-Compatible Research Path

Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

Generate synthetic QA:

```bash
python -m raglab.cli.main dataset generate \
  --docs datasets/sample/docs \
  --output datasets/generated/sample_qa.jsonl \
  --limit 50 \
  --model "$OPENAI_CHAT_MODEL"
```

Run full RAG evaluation with LLM-as-judge metrics:

```bash
python -m raglab.cli.main eval \
  --config techniques/rag_sequence_2020/config.yaml \
  --artifact artifacts/rag_sequence_2020 \
  --dataset datasets/generated/sample_qa.jsonl \
  --output benchmarks/results/rag_sequence_2020_eval.json \
  --mode full_rag \
  --judge \
  --judge-model "$OPENAI_JUDGE_MODEL"
```

## Benchmark Results

Committed smoke benchmark on the tiny sample dataset:

| Technique | Mode | Recall@5 | MRR | Citation Accuracy | Latency Avg | Cost Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `naive_rag` | `full_rag` | 1.0 | 1.0 | 0.333333 | 0.469 ms | 0.0 |
| `parent_child` | `full_rag` | 1.0 | 1.0 | 1.0 | 0.466 ms | 0.0 |

The committed sample report is in `benchmarks/results/sample_research/`. It is intentionally small and exists to prove
the benchmark contract. Use at least 50-100 QA pairs before making empirical claims.

## Technique Catalog

| Technique | Type | Base | Requires |
| --- | --- | --- | --- |
| `naive_rag` | local baseline | none | none |
| `parent_child` | production retrieval pattern | `naive_rag` | none |
| `rag_sequence_2020` | paper-inspired RAG-Sequence | `naive_rag` | OpenAI-compatible embeddings/chat |
| `hyde_2022` | paper-inspired HyDE | `rag_sequence_2020` | OpenAI-compatible embeddings/chat |
| `rag_fusion_2024` | paper-inspired RAG-Fusion | `rag_sequence_2020` | OpenAI-compatible embeddings/chat |
| `self_rag_2023` | concept-only critique verifier | `parent_child` | OpenAI-compatible chat |

List and inspect techniques:

```bash
python -m raglab.cli.main techniques list
python -m raglab.cli.main techniques show rag_sequence_2020
```

## Adding A Technique

1. Copy `techniques/_template/`.
2. Fill `technique.yaml` with metadata, implementation level, requirements, and weak cases.
3. Compose the pipeline in `config.yaml`.
4. Add custom code only when existing registry components are not enough.
5. Register custom code through `custom/register.py`.
6. Run the same benchmark dataset as the baselines.

See `docs/adding_techniques.md` for the full extension contract.

Config supports these scalable interfaces:

```json
{
  "indexing": {
    "embedding": {"type": "openai", "params": {"model": "${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}"}},
    "store": {"type": "json_memory"}
  },
  "evaluation": {
    "mode": "full_rag",
    "judge": {"type": "openai", "params": {"model": "${OPENAI_JUDGE_MODEL:-gpt-4.1-mini}"}}
  }
}
```

Vector store backends:

- `json_memory`: zero-dependency artifact backend for smoke tests and small experiments.
- `faiss_local`: optional local vector backend for larger research runs. Install with `pip install ".[vector]"`.

## Development

Install development dependencies:

```bash
pip install -e ".[dev,research]"
```

Quality gates:

```bash
make lint
make typecheck
make test
make bench-sample
make ci
```

CI runs Ruff, mypy, and pytest on Python 3.11.

Evaluation details are documented in `docs/evaluation_protocol.md`.

## Roadmap

- Add larger public benchmark datasets with 50-100+ labeled QA pairs.
- Add FAISS-backed committed example config.
- Add regression comparison across benchmark summaries.
- Add richer failure analysis for multi-hop and unanswerable questions.
- Add more retrieval and verification techniques on top of the stable base.
