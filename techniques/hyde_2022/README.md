# HyDE: Hypothetical Document Embeddings

## Source

- Paper: [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496)
- Authors: Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- Year: 2022

## Core Idea

HyDE generates multiple hypothetical documents from the query, embeds those generated documents, averages their embeddings, and retrieves real corpus documents near the averaged vector. The generated documents may contain false details, but the dense embedding bottleneck plus averaging is intended to reduce hallucination noise while preserving relevance patterns.

## RAG Stage

- query_transformation
- retrieval

## What This Repo Implements

This is a paper-inspired HyDE retriever:

```text
query
  -> OpenAI-compatible chat model samples N hypothetical documents
  -> OpenAI-compatible embedding model embeds each hypothetical document
  -> average hypothetical document embeddings
  -> cosine search over pre-embedded indexed nodes using the averaged vector
  -> normal RAG context construction and generation
```

The original paper used an instruction-following generator and Contriever-style unsupervised dense encoder. This repo uses OpenAI-compatible chat and embedding APIs for practical experimentation.

Default config uses `samples=5` and `temperature=0.7` to preserve the N-sample averaging mechanism. This is still paper-inspired rather than faithful because the model stack differs from the original paper.

## Suitable Data

HyDE is worth testing when:

- user queries are vague or underspecified
- documents are prose-heavy
- semantic similarity matters more than exact lexical overlap
- there are no relevance labels for retriever fine-tuning

## Weak Cases

HyDE can hurt when:

- exact keyword matching is required
- the generated hypothetical document introduces misleading terms
- latency/cost constraints are strict
- answers depend on tables, numbers, or exact clauses

## How To Run

```bash
python -m raglab.cli.main ingest \
  --config techniques/hyde_2022/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/hyde_2022

python -m raglab.cli.main eval \
  --config techniques/hyde_2022/config.yaml \
  --artifact artifacts/hyde_2022 \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/hyde_2022_eval.json
```
