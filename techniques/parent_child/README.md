# Parent-Child Retrieval

## Source

This is a production RAG pattern rather than a single paper reproduction.

## Core Idea

Index small child chunks for precise retrieval, but use larger parent context for answer generation. This can improve citation and answer completeness on structured documents.

## RAG Stage

- chunking
- indexing
- retrieval
- context_construction

## What This Repo Implements

The current implementation uses `parent_child` chunking with BM25 retrieval and citation-aware context construction.

## Suitable Data

Good for:

- structured policy documents
- legal-like documents
- manuals with section hierarchy
- cases where child chunks retrieve well but need parent context to answer

## Weak Cases

Can add unnecessary context for very short FAQ-like documents. It may also hurt latency and context cost if parent sections are large.

## How To Run

```bash
python -m raglab.cli.main ingest \
  --input datasets/sample/docs \
  --output artifacts/parent_child

python -m raglab.cli.main eval \
  --artifact artifacts/parent_child \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/parent_child_eval.json
```
