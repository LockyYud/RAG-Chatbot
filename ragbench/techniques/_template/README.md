# Technique Name

## Source

- Paper:
- Authors:
- Year:
- URL:

## Core Idea

Explain the research idea in practical RAG terms.

## RAG Stage

List the stages this technique changes:

- data_processing
- chunking
- enrichment
- indexing
- query_transformation
- retrieval
- reranking
- context_construction
- generation
- verification

## What This Repo Implements

State whether this is a faithful reproduction or a paper-inspired adaptation.

## What This Repo Does Not Reproduce

List any original training, datasets, model architecture, or evaluation protocol not reproduced.

## Suitable Data

Describe document and query types where this technique is expected to help.

## Weak Cases

Describe failure modes and cases where a cheaper baseline is likely enough.

## How To Run

```bash
python -m ragbench.cli.main ingest \
  --input datasets/sample/docs \
  --output artifacts/<technique_id>

python -m ragbench.cli.main eval \
  --artifact artifacts/<technique_id> \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/<technique_id>_eval.json
```
