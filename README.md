# RAG Pipeline Lab

`rag-pipeline-lab` is a scalable, evaluation-first RAG research base for comparing retrieval, chunking,
generation, verification, and evaluation strategies under one reproducible protocol.

The repo is designed as a framework, not a notebook collection. A technique owns its metadata, config, optional custom
code, and benchmark report. The engine keeps stable interfaces for processing, indexing, retrieval, context building,
generation, verification, and evaluation so new research ideas can be added without rewriting the whole pipeline.

## What It Is

- A self-contained, code-first RAG pipeline engine with stable component contracts.
- A benchmark harness that writes per-run reports plus JSON, CSV, and Markdown summaries.
- A local smoke path that runs without API keys.
- An OpenAI-compatible research path for embeddings, generation, synthetic QA, and LLM-as-judge metrics.
- Artifact v5 with canonical config/corpus fingerprints, locked embedding metadata, embeddings stored in a
  binary `.npy` file (not inline JSON), and vector-store validation — FAISS by default above a configurable
  node-count threshold, numpy-vectorized exact search below it.

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
ragbench/
  core/        # schema, interfaces, config, the BasePipeline contract (base.py), measure helpers
  processing/  # parsers, cleaners, chunkers, enrichers
  indexing/    # embeddings, retrievers, artifacts, vector stores
  inference/   # context builders, generators, rerankers, verifiers, controllers
  datasets/    # synthetic QA plus research dataset adapters
  cli/         # developer-facing CLI

evaluation/    # retrieval metrics, judge metrics, eval runner
benchmarks/    # reproducible multi-technique benchmark runner
techniques/    # one self-contained pipeline.py + technique.yaml per technique
```

Each technique is a self-contained `techniques/<id>/pipeline.py` exposing a
`BasePipeline` subclass (no plugin registry, no YAML config overlay) — reading the
file top-to-bottom tells you exactly what the paper does. The CLI loads it by id.

## Quickstart

Offline smoke path, no API key:

```bash
python -m ragbench.cli.main ingest \
  --technique parent_child \
  --input datasets/sample/docs \
  --output artifacts/parent_child

python -m ragbench.cli.main eval \
  --technique parent_child \
  --artifact artifacts/parent_child \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/parent_child_eval.json
```

Run a local smoke benchmark (this is a CLI regression check, not evidence of
technique quality):

```bash
python -m ragbench.cli.main bench \
  --techniques parent_child \
  --docs datasets/sample/docs \
  --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/sample_research \
  --mode full_rag
```

For a claim-eligible suite, use a frozen machine-readable contract. It locks
the dataset, mode, cutoff, and required baselines, then writes an explicit
eligibility verdict instead of treating a smoke run as research evidence:

```bash
python -m ragbench.cli.main bench \
  --suite evaluation/protocol/vi_retrieval_core.yaml \
  --techniques parent_child naive_rag bm25_hybrid_rerank \
  --output benchmarks/results/vi_retrieval_core
```

For production latency measurements, declare warm-up and repetition policy in
the report. Add `--judge --judge-model …` only for a full-RAG golden suite;
judge cost is reported separately from technique cost.

Inspect an artifact:

```bash
python -m ragbench.cli.main artifacts inspect --artifact artifacts/parent_child
```

## OpenAI-Compatible Research Path

Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4.1-mini
OPENAI_JUDGE_MODEL=gpt-4.1-mini
```

Pass model choices into a technique explicitly with `--param`, for example
`--param embedding_model='"ollama/nomic-embed-text"' --param generator_model='"ollama/llama3"'`.

Generate synthetic QA:

```bash
python -m ragbench.cli.main dataset generate \
  --docs datasets/sample/docs \
  --output datasets/generated/sample_qa.jsonl \
  --limit 50 \
  --model "$CHAT_MODEL"
```

Run full RAG evaluation with LLM-as-judge metrics:

```bash
python -m ragbench.cli.main eval \
  --technique rag_sequence_2020 \
  --artifact artifacts/rag_sequence_2020 \
  --dataset datasets/generated/sample_qa.jsonl \
  --output benchmarks/results/rag_sequence_2020_eval.json \
  --mode full_rag \
  --judge \
  --judge-model "$OPENAI_JUDGE_MODEL"
```

## Vietnamese Research Datasets

The fixed Vietnamese datasets are for **evaluation and research experiments only**. They do not change the normal user
document flow. User-facing RAG still works by ingesting raw user documents through `ragbench ingest` and querying the saved
artifact through `ragbench query`.

Research dataset flow:

```text
public benchmark dataset
  -> ragbench dataset prepare
  -> processed evaluation fixture
  -> ragbench ingest processed docs
  -> ragbench eval processed qa/qrels
```

List supported adapters:

```bash
python -m ragbench.cli.main dataset list
```

Current adapters:

| Adapter | Dataset | Purpose |
| --- | --- | --- |
| `viequad_retrieval` | `mteb/VieQuADRetrieval` | Vietnamese retrieval benchmark |
| `uit_viquad` | `taidng/UIT-ViQuAD2.0` | Vietnamese QA and abstention evaluation |
| `vietnamese_legal_documents` | `YuITC/Vietnamese-Legal-Documents` | Vietnamese legal retrieval/RAG |
| `vietnamese_legal_qa_rag` | `NamSyntax/Vietnamese-Legal-QA-RAG` | Small legal full-RAG evaluation |
| `vimqa` | VIMQA-style multi-hop QA | Multi-hop Vietnamese reasoning, subject to upstream access |
| `vnfinsqa` | `duykhangh/VNFinsQA` | Vietnamese finance QA |

Prepare a processed evaluation fixture:

```bash
uv run --extra research python -m ragbench.cli.main dataset prepare viequad_retrieval \
  --output datasets/processed/vi_wiki_retrieval \
  --limit 200
```

Validate the processed fixture:

```bash
python -m ragbench.cli.main dataset validate datasets/processed/vi_wiki_retrieval
```

The prepared directory contains both canonical evaluation files and compatibility exports for the current pipeline:

```text
datasets/processed/vi_wiki_retrieval/
  documents.jsonl
  queries.jsonl
  qrels.jsonl
  docs/
  qa.jsonl
  dataset_card.md
  manifest.json
```

Run ingest/eval on a prepared dataset:

```bash
uv run python -m ragbench.cli.main ingest \
  --technique parent_child \
  --input datasets/processed/vi_wiki_retrieval/docs \
  --output artifacts/vi_wiki_retrieval/parent_child

uv run python -m ragbench.cli.main eval \
  --technique parent_child \
  --artifact artifacts/vi_wiki_retrieval/parent_child \
  --dataset datasets/processed/vi_wiki_retrieval \
  --output benchmarks/results/vi_wiki_retrieval/parent_child_eval.json \
  --mode retrieval_only
```

Create a smaller local sample from a prepared fixture:

```bash
python -m ragbench.cli.main dataset sample datasets/processed/vi_wiki_retrieval \
  --output datasets/processed/vi_wiki_retrieval_sample_50 \
  --limit 50
```

The `prepare` command above enables research dependencies for that run. To install
them in the project environment before downloading Hugging Face datasets:

```bash
uv sync --extra research
```

Some upstream datasets may require network access, Hugging Face authentication, or explicit acceptance of dataset terms.
The repo should not commit full raw external datasets; commit adapters, small permitted samples, configs, and benchmark
reports instead.

## Benchmark Results

Committed smoke benchmark on the tiny sample dataset:

| Technique | Mode | Recall@5 | MRR | Citation F1 | Latency Avg | Cost Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `parent_child` | `full_rag` | 1.0 | 1.0 | 1.0 | machine-dependent | 0.0 |

The committed sample report is in `benchmarks/results/sample_research/`. It is intentionally small and exists to prove
the benchmark contract. Use at least 50-100 QA pairs before making empirical claims.

## Technique Catalog

Read top-to-bottom as a progression: local baselines, the LLM-tier paper baselines,
then the modern retrieval stack (hybrid → contextual indexing → agentic control).

| Technique | Type | Base | Requires |
| --- | --- | --- | --- |
| `naive_rag` | local baseline | none | none |
| `parent_child` | production retrieval pattern | `naive_rag` | none |
| `rag_sequence_2020` | paper-inspired RAG-Sequence | `naive_rag` | OpenAI-compatible embeddings/chat |
| `hyde_2022` | paper-inspired HyDE | `rag_sequence_2020` | OpenAI-compatible embeddings/chat |
| `rag_fusion_2024` | paper-inspired RAG-Fusion | `rag_sequence_2020` | OpenAI-compatible embeddings/chat |
| `self_rag_2023` | concept-only critique verifier | `parent_child` | OpenAI-compatible chat |
| `bm25_hybrid_rerank` | production hybrid baseline (BM25 + dense, RRF, cross-encoder) | `rag_sequence_2020` | OpenAI-compatible embeddings/chat; optional `[rerank]` extra |
| `contextual_retrieval_2024` | paper-inspired Contextual Retrieval (Anthropic 2024) | `bm25_hybrid_rerank` | OpenAI-compatible embeddings/chat; optional `[rerank]` extra |
| `agentic_rag_arag` | paper-inspired Agentic RAG (A-RAG, 2026) | `bm25_hybrid_rerank` | OpenAI-compatible embeddings/chat; optional `[rerank]` extra |

List and inspect techniques:

```bash
python -m ragbench.cli.main techniques list
python -m ragbench.cli.main techniques show rag_sequence_2020
```

## Adding A Technique

1. Copy `techniques/_template/` to `techniques/<your_id>/`.
2. Fill `technique.yaml` with metadata: implementation level, requirements, best/weak cases.
3. Implement the `BasePipeline` subclass in `pipeline.py` — wire up the engine
   components you need (chunkers, embedders, retrievers, rerankers, controllers,
   generators, verifiers) and finish `query()` with `build_query_metadata(...)` so
   benchmarks compare fairly.
4. Keep paper-specific code inline in `pipeline.py`; only split into a `custom/`
   folder when the file would grow past ~250 lines.
5. Run the same benchmark dataset as the baselines.

The technique is then discovered automatically — no registry to edit:

```bash
python -m ragbench.cli.main techniques list
python -m ragbench.cli.main ingest --technique <your_id> --input docs/ --output artifacts/<your_id>
```

Override any `pipeline.py` constructor parameter from the CLI with `--param` (JSON-parsed, repeatable):

```bash
python -m ragbench.cli.main ingest --technique <your_id> --input docs/ --output art/ \
  --param chunk_size=300 --param top_k=10
```

See `docs/adding_techniques.md` for the full extension contract.

For the end-to-end paper-to-benchmark workflow, evaluation profiles, artifact
provenance, and multi-trial experiments, see [the research workflow contract](docs/research_contract.md).

Vector store backends:

- `json_memory`: zero-dependency artifact backend for smoke tests and small experiments.
- `faiss_local`: optional local vector backend for larger research runs. Install with `pip install ".[vector]"`.

Retrieval, reranking and orchestration components (reusable across techniques):

- `RRFHybridRetriever` (`ragbench/indexing/retrievers.py`): fuses `BM25Retriever` and
  `DenseRetriever` with Reciprocal Rank Fusion (rank-based, no score-scale tuning).
  `reciprocal_rank_fusion()` is exposed as a pure helper.
- `CrossEncoderReranker` (`ragbench/inference/rerankers/cross_encoder.py`): precision
  reranking via a `sentence-transformers` cross-encoder (`reranker_backend="local"`, default) or a hosted
  rerank API via litellm (`reranker_backend="api"`, e.g. `cohere/rerank-english-v3.0`). Evaluation and
  benchmark runs are strict; interactive query fallback requires `--allow-fallback` and is recorded in metadata.
- `ContextualEnricher` (`ragbench/processing/enrichers/contextual.py`): prepends an
  LLM-generated, document-aware context to each chunk's index text (Anthropic's
  Contextual Embeddings + Contextual BM25 in one step).
- `AgenticRetrievalController` (`ragbench/inference/controllers/agentic.py`): a
  training-free, multi-step retrieval loop with an injectable policy, `max_steps`
  guards, and a structured decision trace — the shared engine for agentic techniques.

Install the optional cross-encoder model: `pip install ".[rerank]"`.

## Development

Install development dependencies:

```bash
pip install -e ".[dev,research]"
```

Add `rerank` for the cross-encoder reranker (otherwise it falls back to lexical overlap):

```bash
pip install -e ".[dev,research,rerank]"
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

Recently landed:

- Strong hybrid baseline: `bm25_hybrid_rerank` (BM25 + dense, RRF fusion, cross-encoder rerank).
- Modern indexing: `contextual_retrieval_2024` (LLM-situated chunks).
- 2026 paradigm: `agentic_rag_arag` (training-free agentic retrieval loop) + Wave 6 in the research roadmap.

Next:

- Add larger public benchmark datasets with 50-100+ labeled QA pairs.
- Add FAISS-backed committed example config.
- Add regression comparison across benchmark summaries.
- Add richer failure analysis for multi-hop and unanswerable questions (use `metadata.agent.trace`).
- Adaptive router (Wave 6): decide per-question between a single hybrid pass and the agentic loop.

The full paper roadmap lives in [`docs/research_roadmap.md`](docs/research_roadmap.md);
per-technique study notes are in [`docs/blogs/`](docs/blogs/README.md).
