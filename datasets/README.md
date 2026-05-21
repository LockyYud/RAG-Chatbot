# Datasets

This repo is bring-your-own-data by default.

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

## QA Format

Evaluation datasets use JSONL:

```json
{"question_id":"q001","question":"...","ground_truth_answer":"...","expected_doc_ids":["doc_id"],"expected_chunk_ids":[],"expected_citations":["doc_id:section"],"metadata":{"type":"FACTUAL"}}
```

For meaningful technique comparison, prepare at least 50-100 labeled QA pairs.
