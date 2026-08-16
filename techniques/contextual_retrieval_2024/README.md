# Contextual Retrieval (Anthropic, 2024)

## Source
- Post: *Introducing Contextual Retrieval*, Anthropic, 2024 — <https://www.anthropic.com/news/contextual-retrieval>
- Background blog: [`docs/blogs/contextual_retrieval_2024.md`](../../docs/blogs/contextual_retrieval_2024.md)

## Core Idea
Chunking strips the context a chunk needs to be retrievable ("revenue grew 3%
that quarter" — *which* company, *which* quarter?). Before indexing, an LLM
writes a 1–2 sentence snippet that situates each chunk inside its full document,
and that snippet is prepended to the chunk's indexing text.

## Stage Changed
Enrichment / indexing only. Because both the dense embedder and BM25 read
`IndexedNode.text_for_embedding`, prepending there produces both of Anthropic's
variants at once — **Contextual Embeddings** and **Contextual BM25**. Retrieval,
reranking, generation and verification are unchanged from `bm25_hybrid_rerank`.

## What This Repo Implements
- `ContextualEnricher` (`raglab/processing/enrichers/contextual.py`): one LLM call
  per chunk, `(document, chunk) -> situating context`, prepended to the index text.
  `text_for_generation` keeps the original chunk so answers never quote the
  synthetic context. The context function is injectable for offline tests.
- A pipeline that reuses `RRFHybridRetriever` + `CrossEncoderReranker` so a
  head-to-head benchmark isolates exactly what contextualization buys.

## What This Repo Does Not Reproduce
- No prompt-caching of the document across chunks (Anthropic's cost trick) — each
  chunk sends a (truncated) copy of its document.
- No automated context-quality evaluation; quality depends on `context_model`.

## Expected Strengths
- Large recall gains on long, context-dependent corpora — Anthropic reports
  ~49% fewer top-20 retrieval failures (≈67% with reranking).
- Pure indexing-stage change: composes with any downstream retriever/reranker.

## Expected Failure Modes
- Ingest cost/latency: one LLM call per chunk.
- Marginal on already self-contained chunks (FAQ entries, short standalone docs).
- Context quality is bounded by `context_model`; a weak model can add noise.

## Config
| Param | Default | Role |
| --- | --- | --- |
| `context_model` | `gpt-4.1-mini` | LLM that writes each chunk's situating context |
| `max_doc_tokens` | `4000` | Truncate document sent to the context LLM (cost guard) |
| `rrf_k` / `candidate_k` / `rerank_top_k` | `60` / `30` / `6` | Inherited hybrid + rerank knobs |
| `embedding_model` | `text-embedding-3-small` | dense embedding model |

## Benchmark Results
Compare directly against the hybrid baseline (same retriever/reranker):

```bash
raglab bench --techniques bm25_hybrid_rerank contextual_retrieval_2024 \
  --docs datasets/sample/docs --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/contextual --mode retrieval_only
```

`manifest.contextualized_nodes` records how many chunks got a context prefix.

## Implementation Notes
- Requires embeddings + a chat model. A failed per-chunk context call degrades to
  the plain chunk rather than aborting ingest.
- Per-node `metadata['contextual_prefix']` stores the generated context for audit.
