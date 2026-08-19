# BM25 + Dense Hybrid with RRF and Cross-Encoder Reranking

## Source
- Pattern: production hybrid retrieval (not a single paper)
- Origins:
  - BM25 — Robertson & Zaragoza, 2009, *The Probabilistic Relevance Framework: BM25 and Beyond*
  - DPR — Karpukhin et al., 2020, *Dense Passage Retrieval* — <https://arxiv.org/abs/2004.04906>
  - RRF — Cormack et al., 2009, *Reciprocal rank fusion outperforms condorcet and individual rank learning methods*
  - Cross-encoder reranking — Nogueira & Cho, 2019, *Passage Re-ranking with BERT* — <https://arxiv.org/abs/1901.04085>
- Background blog: [`docs/blogs/bm25_hybrid_rerank.md`](../../docs/blogs/bm25_hybrid_rerank.md)

## Core Idea
No single retriever wins on every corpus. BM25 reliably catches exact lexical
matches (codes, identifiers, rare entities); dense retrieval catches paraphrase
and semantics. Fuse both ranked lists with Reciprocal Rank Fusion — which uses
*rank*, not raw score, so the BM25/cosine scale mismatch never matters and there
is no per-corpus `alpha` to tune — then rerank the fused candidate pool with a
cross-encoder for precision.

## Stage Changed
Retrieval (BM25 + dense), score fusion (RRF), and reranking (cross-encoder).
Chunking, context construction, generation and verification are the same as the
LLM-tier baseline.

## What This Repo Implements
- `RRFHybridRetriever` (`ragbench/indexing/retrievers.py`): runs `BM25Retriever`
  and `DenseRetriever`, fuses their lists via `reciprocal_rank_fusion(k=60)`.
- `CrossEncoderReranker` (`ragbench/inference/rerankers/cross_encoder.py`): scores
  `(query, passage)` pairs jointly with a `sentence-transformers` CrossEncoder.
- End-to-end pipeline wiring hybrid retrieval → rerank → citation context →
  chat generation → citation-coverage verification.

## What This Repo Does Not Reproduce
- No trained DPR encoders — dense retrieval uses pretrained embeddings.
- No learned fusion weights — RRF is unsupervised by design.
- No GPU-served reranker; the default CrossEncoder is a small CPU MS MARCO model,
  and falls back to lexical-overlap reranking when the `rerank` extra is absent.

## Expected Strengths
- Strong, fair baseline: harder to beat than any single-retriever setup.
- Robust across mixed query distributions (keyword-exact and paraphrased).
- RRF needs no score normalization or per-corpus tuning.

## Expected Failure Modes
- Higher latency: two retrievers plus a reranker per query.
- On tiny homogeneous corpora, the dense half can add noise over plain BM25.
- Reranker can only reorder what fusion surfaced — a small `candidate_k` caps
  achievable recall (set it generously).

## Config
Tunable constructor params (override with `--param key=value`):

| Param | Default | Role |
| --- | --- | --- |
| `rrf_k` | `60.0` | RRF smoothing constant |
| `candidate_k` | `30` | Pool size pulled from each retriever before fusion/rerank |
| `rerank_top_k` | `6` | Final results kept after reranking |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | sentence-transformers CrossEncoder id |
| `embedding_model` | `text-embedding-3-small` | dense embedding model |
| `generator_model` | `gpt-4.1-mini` | answer generator (full_rag only) |

Install the real cross-encoder: `pip install ".[rerank]"`. Without it the
reranker degrades gracefully to lexical overlap.

## Benchmark Results
Run against the same dataset as the other techniques:

```bash
ragbench bench --techniques naive_rag parent_child bm25_hybrid_rerank \
  --docs datasets/sample/docs --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/sample --mode retrieval_only
```

## Implementation Notes
- Requires embeddings saved at ingest (same as `naive_rag` / `rag_sequence_2020`).
- `reciprocal_rank_fusion` is a pure helper — unit-tested without any API call.
- Per-result `metadata['reranker']` records whether the real cross-encoder or
  the lexical fallback ran.
