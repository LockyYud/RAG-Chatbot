# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

## Source

- arXiv: [2005.11401](https://arxiv.org/abs/2005.11401)
- Paper title: *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Authors: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rocktaschel, Sebastian Riedel, Douwe Kiela
- Year: 2020

## Core Idea

The paper combines a parametric seq2seq generator with a non-parametric retrieval memory. Instead of relying only on model parameters, the system retrieves relevant passages and conditions generation on those passages.

The paper describes two main variants:

- **RAG-Sequence**: retrieve passages once for the input and generate the whole output conditioned on those retrieved passages.
- **RAG-Token**: allow different retrieved passages to support different generated tokens.

## RAG Stage

This paper primarily affects:

- indexing
- dense retrieval
- context construction
- generation

It is not mainly a chunking, reranking, or verification paper.

## What This Repo Implements

This repo starts with a **paper-inspired RAG-Sequence pipeline**:

```text
documents
  -> recursive passage chunking
  -> optional section-title enrichment
  -> OpenAI-compatible dense embeddings
  -> top-k dense retrieval
  -> citation-aware context
  -> OpenAI-compatible chat generation
  -> citation coverage verification
```

Configure constructor parameters with repeatable `--param key=value` flags; the pipeline's resolved configuration is persisted in each artifact.

This implementation is designed for practical document QA over a user-provided corpus.

## What This Repo Does Not Reproduce

This is not a faithful reproduction of the original paper.

It does not reproduce:

- DPR training
- BART fine-tuning
- end-to-end marginalization over retrieved documents
- RAG-Token decoding
- original Wikipedia preprocessing
- original open-domain QA benchmark setup

The current implementation is an engineering adaptation of the RAG-Sequence idea for modern OpenAI-compatible embedding and chat APIs.

## Suitable Data

This strategy is a good first baseline for:

- factual QA
- policy documents
- documentation pages
- manuals and handbooks
- documents where answers appear in one or a few nearby passages
- corpora where semantic retrieval is more important than exact keyword matching

## Weak Cases

This strategy can struggle with:

- table-heavy documents
- multi-hop questions requiring evidence from many sections
- questions requiring numerical aggregation
- very long legal clauses where section boundaries matter more than semantic similarity
- ambiguous queries that need query rewriting or clarification
- datasets where citations must point to exact page/line spans

For those cases, compare against parent-child chunking, hybrid retrieval, reranking, query rewriting, or graph/hierarchical methods.

## Evaluation Plan

Compare this pipeline against:

- `naive_dense`
- `heading_hybrid`
- `parent_child_bm25`

Recommended metrics:

- Recall@k
- MRR
- citation accuracy
- answer correctness
- faithfulness
- latency
- token usage
- estimated cost

Use at least 50-100 labeled QA pairs before making any empirical claim.

## Production Notes

RAG-Sequence-style retrieval is simple and strong as a default baseline. It is often the first production candidate because it is easy to reason about and has predictable latency.

Use it when:

- documents are mostly prose
- retrieval evidence is localized
- cost and implementation complexity matter

Do not stop at this baseline when:

- citation precision is poor
- relevant context is split across sections
- query wording differs strongly from document wording
- answer quality depends on structured tables or calculations
