# Self-RAG-Inspired Critique Verifier

## Source

- Paper: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- Authors: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- Year: 2023

## Core Idea

Self-RAG trains a language model to retrieve, generate, and critique its own output through reflection signals. The paper introduces reflection-style decisions such as retrieval need, passage relevance, support, and utility.

## RAG Stage

- generation
- verification

## What This Repo Implements

This is not a faithful Self-RAG reproduction. This technique is marked as `concept_only` in `technique.yaml`.

The repo implements only a Self-RAG-inspired verifier:

```text
answer + retrieved context
  -> OpenAI-compatible critique model
  -> JSON groundedness/citation judgment
  -> VerificationReport used by the pipeline metadata
```

The current verifier is a practical post-generation critique step. It is closest to the paper's support-checking idea, but it does not reproduce the core training or decoding mechanism.

It does not implement:

- adaptive retrieval decisions
- passage relevance scoring
- utility scoring
- reflection-token training
- segment-level beam search
- fine-tuned Self-RAG model weights

## Suitable Data

Use this technique when:

- answer factuality matters
- citations must be checked against retrieved context
- you want an additional judge before trusting outputs
- latency/cost overhead is acceptable

## Weak Cases

This verifier can fail when:

- the critique model is weak or overly lenient
- evidence requires exact table/line reasoning
- context is too long and contains conflicting snippets
- you need deterministic, auditable human-grade verification

## How To Run

```bash
python -m raglab.cli.main ingest \
  --config techniques/self_rag_2023/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/self_rag_2023

python -m raglab.cli.main eval \
  --config techniques/self_rag_2023/config.yaml \
  --artifact artifacts/self_rag_2023 \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/self_rag_2023_eval.json
```
