# Datasets

This repo is bring-your-own-data by default.

The fixed Vietnamese datasets are **research/evaluation fixtures only**. They are not assumptions in the user document
ingest/query path.

## Sample Dataset

The committed sample lives under:

```text
datasets/sample/
  docs/
  qa.jsonl
```

It is intentionally tiny and exists only for smoke tests.

## User Data

Place private or larger datasets under:

```text
datasets/user_data/
```

This directory is ignored by git except for `.gitkeep`.

## Generated Data

Synthetic QA can be generated with an OpenAI-compatible chat model:

```bash
python -m raglab.cli.main dataset generate \
  --docs datasets/sample/docs \
  --output datasets/generated/sample_qa.jsonl \
  --limit 50
```

Generated rows follow the same JSONL schema and include metadata fields for `question_type`, `difficulty`,
`source_chunk_id`, and `generated`.

## QA Format

Evaluation datasets use JSONL:

```json
{"question_id":"q001","question":"...","ground_truth_answer":"...","expected_doc_ids":["doc_id"],"expected_chunk_ids":[],"expected_citations":["doc_id"],"metadata":{"type":"FACTUAL","is_answerable":true}}
```

For meaningful technique comparison, prepare at least 50-100 labeled QA pairs.

## Fixed Vietnamese Evaluation Datasets

Dataset adapters convert public benchmark datasets into a canonical processed format:

```text
datasets/processed/<dataset_id>/
  documents.jsonl
  queries.jsonl
  qrels.jsonl
  docs/
  qa.jsonl
  dataset_card.md
  manifest.json
```

Available adapters:

```bash
raglab dataset list
```

Prepare a benchmark fixture:

```bash
uv run --extra research python -m raglab.cli.main dataset prepare viequad_retrieval \
  --output datasets/processed/vi_wiki_retrieval \
  --limit 200
```

Preparation refuses to write into a non-empty directory. Pass `--overwrite` for a validated atomic replacement. Dataset
sampling accepts `--seed`; the seed and canonical dataset fingerprint are persisted in `manifest.json`.

Validate it:

```bash
raglab dataset validate datasets/processed/vi_wiki_retrieval
```

Then run the existing RAG path against the exported docs/eval files:

```bash
uv run python -m raglab.cli.main ingest \
  --technique parent_child \
  --input datasets/processed/vi_wiki_retrieval/docs \
  --output artifacts/vi_wiki_retrieval/parent_child

uv run python -m raglab.cli.main eval \
  --technique parent_child \
  --artifact artifacts/vi_wiki_retrieval/parent_child \
  --dataset datasets/processed/vi_wiki_retrieval \
  --output benchmarks/results/vi_wiki_retrieval/parent_child_eval.json \
  --mode retrieval_only
```

Current adapters:

- `viequad_retrieval`: `mteb/VieQuADRetrieval`.
- `uit_viquad`: `taidng/UIT-ViQuAD2.0`.
- `vietnamese_legal_documents`: `YuITC/Vietnamese-Legal-Documents`.
- `vietnamese_legal_qa_rag`: `NamSyntax/Vietnamese-Legal-QA-RAG`.
- `vimqa`: VIMQA-style multi-hop QA, conditional on upstream access.
- `vnfinsqa`: `duykhangh/VNFinsQA`.
# Research datasets

For reproducible retrieval benchmarks, a sampled query set must retain the
full upstream corpus. See [`docs/benchmark_protocol_v1.md`](../docs/benchmark_protocol_v1.md).

`golden/TEMPLATE.jsonl` is the schema for human-reviewed end-to-end RAG
evaluation. Validate a set with `raglab dataset validate-golden <path>`.
