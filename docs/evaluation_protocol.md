# Evaluation Protocol

`rag-pipeline-lab` supports two evaluation modes:

- `retrieval_only`: runs retrieval, reranking, and context construction only. Use this for cheap deterministic recall,
  hit rate, and MRR checks; it does not generate citations.
- `full_rag`: runs retrieval through generation and verification. Use this for end-to-end answer quality.

It also records an explicit evaluation profile: `retrieval`, `single_hop_rag`, `multi_hop_rag`, or
`citation_rag`. The profile is checked against the technique's declared capabilities and stored in run metadata;
use `multi_hop_rag` when the benchmark requires all of several evidence items, and `citation_rag` when citation
labels are available.

## Dataset Schema

Each JSONL row must contain:

```json
{
  "question_id": "q001",
  "question": "...",
  "ground_truth_answer": "...",
  "expected_doc_ids": ["doc_id"],
  "expected_chunk_ids": ["chunk_id"],
  "expected_citations": ["doc_id"],
  "metadata": {"question_type": "factual"}
}
```

`expected_chunk_ids` gives the strictest retrieval signal. If it is empty, metrics fall back to `expected_doc_ids`.

## Metrics

- `recall_at_k`: fraction of expected chunks/docs retrieved in top-k.
- `hit_rate`: whether at least one expected chunk/doc appears in top-k.
- `mrr`: reciprocal rank of the first expected chunk/doc.
- `citation_precision`, `citation_recall`, `citation_f1` (`full_rag` only): document-level citation provenance.
- `abstention_accuracy`: explicit answer/refusal correctness for answerable and unanswerable questions.
- Retrieval metrics exclude unanswerable or unlabeled questions and report their evaluated denominator.
- `latency_ms_avg`: average query latency.
- `context_tokens_avg`: average retrieved context size.
- `estimated_cost_avg`: average estimated model/API cost from configured pricing.

Optional LLM-as-judge metrics are added when `--judge` is enabled:

- `answer_correctness`
- `faithfulness`
- `citation_support`
- `abstention_correctness`

Judge output is parsed as JSON with a defensive fallback. Malformed judge responses become zero-score notes instead of
crashing the whole benchmark.

Evaluation is strict for pipeline and provider failures: the command exits without writing a partial report. This avoids
mixing a failed component run with valid benchmark metrics.

## Report Shape

Evaluation reports include:

- `run_metadata`
- `metrics`
- `cost_summary`
- `failures`
- `predictions`

Benchmark runs additionally write `summary.json`, `summary.csv`, and `summary.md`.

## Cost accounting

`cost_summary.pipeline_cost` and `cost_summary.judge_cost` each break down into `embedding_cost_total` and
`chat_cost_total` — a technique that is expensive because of retrieval-time LLM calls (HyDE, RAG-Fusion) is
distinguishable from one expensive because of generation, and judge spend never gets folded into technique cost.

A technique using an API-backed cross-encoder (`reranker_backend="api"` — see below) additionally reports
`rerank_calls`/`rerank_cost`, priced with a flat `LLM_RERANK_COST_PER_CALL` (most hosted rerank APIs bill per
search unit, not per token, so this isn't a per-1k-token rate like the others). A run with zero rerank calls
is unaffected — `rerank_priced` defaults to true when `rerank_calls == 0`, so `backend="local"` runs never see
their `cost_status` change.

Each prediction's `cost_estimate.status` (and `evaluation_cost_estimate.status` for the judge) is `"estimated"` only
when *every* call type that run actually used — embedding, chat, or both — had its pricing env vars **configured**.
"Configured" means the variable is explicitly set, including `0` for a free/local model — it is distinct from the
variable being missing or empty, which always means `"unknown"` regardless of what the resulting dollar amount
happens to be. A negative rate is rejected outright as a configuration error. One priced call type does not mask
an unpriced one: a run that calls both embeddings and chat models is `"unknown"` unless `LLM_EMBEDDING_INPUT_COST_PER_1K`
is set **and** both `LLM_CHAT_INPUT_COST_PER_1K` and `LLM_CHAT_OUTPUT_COST_PER_1K` are set.

`--max-estimated-cost-usd` (on `ragbench eval` and `ragbench bench`) aborts a run once its estimated pipeline+judge cost
exceeds the given USD amount, checked after each query. Completed predictions up to that point stay in the
per-query checkpoint, so the run can be resumed (`--resume` on `bench`) instead of losing the spend that already
happened. The guard only ever compares a running total that is entirely `"estimated"` so far — the moment any
query's cost is `"unknown"`, the guard stops evaluating the cap for the rest of that run, no matter how large the
partial (necessarily incomplete) total looks.

## Embedding cache

Every embedding call (ingest-time chunk embedding *and* query-time question embedding — `DenseRetriever` embeds
the question fresh on every `query()` call) is looked up first in a persistent cache keyed by
`(model, normalized_text)`, backed by sqlite at `.raglab_cache/embeddings.sqlite`
(`RAGLAB_EMBEDDING_CACHE_DIR` to relocate it). A cache hit never calls the provider and contributes **zero**
cost/tokens to the usage ledger. Disable with `--no-embedding-cache` (`ingest`/`eval`/`bench`) or
`RAGLAB_EMBEDDING_CACHE=0`.

Ingest-time hit/miss counts are recorded in the artifact manifest at `extra.embedding_usage` (same shape as
`provider_usage` — `embedding_cache_hits`, `embedding_cache_misses`, `embedding_calls`, `embedding_cost`, etc.) —
this is also the first place ingest cost is tracked at all; previously only query-time cost was.

The cache file runs in WAL mode with a 30s busy timeout: under `--concurrency`, every worker thread opens its own
connection to the same sqlite file, and without this a concurrent writer would occasionally hit
`sqlite3.OperationalError: database is locked` instead of just waiting its turn.

## FAISS backend selection and claim-eligibility

Dense techniques pick their vector store backend by corpus size: `json_memory` (numpy-vectorized exact cosine
search) below `RAGLAB_FAISS_NODE_THRESHOLD` (default 2000 nodes) or when `faiss` isn't installed, `faiss_local`
(exact search via `IndexFlatIP`, just vectorized in C++) above it. Both backends produce **identical rankings** —
faiss is not approximate here — so this choice only affects speed, never quality.

For a `claim_eligible` suite, silently substituting `json_memory` because `faiss` happens to be missing on this
machine is a reproducibility problem, not a quality one — a formal benchmark claim must not depend on what's
installed where it happened to run. `ragbench bench --preflight` fails fast if a `claim_eligible` suite is
requested without `faiss` installed (`pip install '.[vector]'`), and `claim_eligibility()` authoritatively
rejects any completed run whose actual (post-ingest) node count crossed the threshold but whose manifest records
a backend other than `faiss_local` — catching the case where the environment silently never engaged faiss.
`smoke_only`/`exploratory` tiers keep the plain silent fallback.

## Cross-encoder reranker backend

`bm25_hybrid_rerank`, `contextual_retrieval_2024`, and `agentic_rag_arag` all rerank their candidate pool with
`CrossEncoderReranker`, chosen with `reranker_backend`:

- `"local"` (default): a `sentence-transformers` model (`reranker_model`, e.g.
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) loaded once in `load()`. Needs the `[rerank]` extra installed; no
  network call at query time, no per-call cost.
- `"api"`: a hosted rerank endpoint called once per query via litellm (`reranker_model` becomes e.g.
  `cohere/rerank-english-v3.0` or `jina_ai/jina-reranker-v2-base-multilingual`). No local model/GPU needed;
  costs a network round trip and (optionally) money per query — see `LLM_RERANK_COST_PER_CALL` above.

Both backends degrade to `LexicalOverlapReranker` when unavailable/failing **and** `allow_fallback=True` is set
(same as before this option existed); with the default `allow_fallback=False`, a missing local model or a
failed API call raises instead of silently substituting a weaker reranker — the same strict-by-default
philosophy as the FAISS/embedding-cache backend choices above. `ragbench doctor` checks readiness for whichever
backend is configured: `sentence-transformers` importability for `"local"`, the provider API key
(`check_provider_ready(reranker_model)`) for `"api"`.

## Concurrent quality pass and sequential latency pass

`--concurrency N` (`ragbench eval`/`ragbench bench`, default `1` = fully sequential, unchanged behavior) splits a run
into two passes once `N > 1`:

- **Latency pass**: the first `--latency-sample-size` (default 5) not-yet-resumed queries run one at a time, with
  no concurrency — real predictions (never re-run), and the only honest source of per-request `latency_ms_p50/p95`
  since nothing is contending for resources yet.
- **Quality pass**: every remaining query runs concurrently across `N` worker threads. Ranking/correctness doesn't
  depend on wall-clock timing, so concurrent execution is safe there; only *individual* latency becomes
  untrustworthy under contention, which is exactly why it isn't measured here.

The report gains a `performance` section (only present when `concurrency > 1` — at `concurrency=1` there is one
pass and `metrics.latency_ms_p50/p95` already covers it):

```json
"performance": {
  "latency_pass": {"mode": "sequential", "sampled_queries": 5, "latency_ms_p50": 340.0, "latency_ms_p95": 480.0},
  "quality_pass": {"mode": "concurrent", "workers": 8, "queries": 45, "elapsed_s": 12.3, "throughput_qps": 3.66}
}
```

Concurrency assumes a technique's `pipeline.query()` does not mutate `self` state beyond what's local to that
call — already an implicit requirement given `--latency-repetitions` already calls `query()` repeatedly and
expects independent, reproducible results per call; concurrency additionally requires no data races between
*simultaneous* calls. Checkpoint writes, cost/retry accounting, and the budget guard are all lock-protected and
concurrency-invariant (same final totals regardless of how many workers ran); predictions are always reassembled
into dataset order before metrics are computed, regardless of completion order.

This was audited across every bundled technique: `HyDERetriever` and `RAGFusionRetriever` used to write a
per-call result onto their own `self.last_metadata` and have the pipeline read it back afterward — a real race,
since a second concurrent query could overwrite it before the first query's read. Fixed (`retrieve()` now returns
`(results, runtime_metadata)` instead); regression-tested in `tests/test_concurrent_technique_metadata.py` by
deterministically forcing one query's entire retrieval to complete inside another query's window between
"retrieval finished" and "runtime metadata read back," rather than relying on incidental thread-scheduling luck.
Every other bundled retriever/reranker/verifier either holds no per-call mutable state or builds it fresh inside
`query()` — see `docs/adding_techniques.md` for the rule new techniques must follow.

`metrics.latency_ms_avg/p50/p95` (and every `metrics_by_cutoff` bucket) are substituted with the latency-pass
sample whenever `concurrency > 1` — the full-dataset aggregate would otherwise silently mix in the quality
pass's contended timings, and `claim_eligibility()` reads exactly this field. The true per-query latency
(including the contended ones) is still visible in `query_metrics`/`query_metrics_by_cutoff` for anyone who wants
to inspect it; only the headline aggregate is replaced.

**Bounded concurrency, not front-loaded**: the quality pass keeps at most `concurrency` futures in flight at
once, submitting a replacement only as each one completes and only while the budget guard hasn't tripped.
Submitting the entire remaining dataset to the executor up front would let the guard trip only after however many
queries happened to complete before the cumulative cost crossed the cap — bounded by cost, not by `concurrency`.
With the sliding window, once the guard trips no new work is ever dispatched, so at most the (at most
`concurrency`) queries already in flight at that moment get to finish.

**Protocol identity**: `concurrency` and `latency_sample_size` are part of the checkpoint header (a resume under a
different value starts a fresh checkpoint rather than mixing predictions whose latency was recorded under a
different meaning of "trustworthy"), the run metadata's `latency_protocol`, and `--resume`'s report-matching key.
A `claim_eligible` suite must declare both explicitly (mirroring `warmup_queries`) so every technique in a
comparison runs under the identical protocol — `latency_sample_size` must be at least 1 whenever
`concurrency > 1`, or there would be no trustworthy latency source left at all.
