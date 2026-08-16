# RAG-Fusion

## Source

- Paper: [RAG-Fusion: a New Take on Retrieval-Augmented Generation](https://arxiv.org/abs/2402.03367)
- Author: Zackary Rackauckas
- Year: 2024

## Core Idea

RAG-Fusion expands the original query into multiple related queries, retrieves documents for each query, and fuses the ranked lists with Reciprocal Rank Fusion (RRF). The goal is to improve recall and context diversity compared with a single retrieval pass.

## RAG Stage

- query_transformation
- retrieval
- reranking

## What This Repo Implements

This is a paper-inspired RAG-Fusion retriever:

```text
query
  -> OpenAI-compatible chat model generates query variants
  -> each query variant retrieves dense candidates
  -> candidates are fused with RRF
  -> top fused candidates feed normal context construction and generation
```

## Suitable Data

RAG-Fusion is worth testing when:

- questions can be phrased in many ways
- document wording differs from user wording
- recall is more important than minimal latency
- the dataset contains heterogeneous documents

## Weak Cases

RAG-Fusion can hurt when:

- exact keyword lookup is already strong
- the generated queries drift from the user intent
- latency/cost constraints are strict
- the corpus is tiny and single-query retrieval already saturates recall

## How To Run

```bash
python -m raglab.cli.main ingest \
  --input datasets/sample/docs \
  --output artifacts/rag_fusion_2024

python -m raglab.cli.main eval \
  --artifact artifacts/rag_fusion_2024 \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/rag_fusion_2024_eval.json
```
