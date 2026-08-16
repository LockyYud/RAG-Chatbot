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

Each prediction's `cost_estimate.status` (and `evaluation_cost_estimate.status` for the judge) is `"estimated"` only
when *every* call type that run actually used — embedding, chat, or both — had its pricing env vars **configured**.
"Configured" means the variable is explicitly set, including `0` for a free/local model — it is distinct from the
variable being missing or empty, which always means `"unknown"` regardless of what the resulting dollar amount
happens to be. A negative rate is rejected outright as a configuration error. One priced call type does not mask
an unpriced one: a run that calls both embeddings and chat models is `"unknown"` unless `LLM_EMBEDDING_INPUT_COST_PER_1K`
is set **and** both `LLM_CHAT_INPUT_COST_PER_1K` and `LLM_CHAT_OUTPUT_COST_PER_1K` are set.

`--max-estimated-cost-usd` (on `raglab eval` and `raglab bench`) aborts a run once its estimated pipeline+judge cost
exceeds the given USD amount, checked after each query. Completed predictions up to that point stay in the
per-query checkpoint, so the run can be resumed (`--resume` on `bench`) instead of losing the spend that already
happened. The guard only ever compares a running total that is entirely `"estimated"` so far — the moment any
query's cost is `"unknown"`, the guard stops evaluating the cap for the rest of that run, no matter how large the
partial (necessarily incomplete) total looks.
