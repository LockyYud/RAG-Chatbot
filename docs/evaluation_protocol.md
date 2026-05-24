# Evaluation Protocol

`rag-pipeline-lab` supports two evaluation modes:

- `retrieval_only`: runs retrieval, reranking, and context construction only. Use this for cheap deterministic recall,
  MRR, and citation-context checks.
- `full_rag`: runs retrieval through generation and verification. Use this for end-to-end answer quality.

## Dataset Schema

Each JSONL row must contain:

```json
{
  "question_id": "q001",
  "question": "...",
  "ground_truth_answer": "...",
  "expected_doc_ids": ["doc_id"],
  "expected_chunk_ids": ["chunk_id"],
  "expected_citations": ["doc_id:section"],
  "metadata": {"question_type": "factual"}
}
```

`expected_chunk_ids` gives the strictest retrieval signal. If it is empty, metrics fall back to `expected_doc_ids`.

## Metrics

- `recall_at_k`: fraction of expected chunks/docs retrieved in top-k.
- `hit_rate`: whether at least one expected chunk/doc appears in top-k.
- `mrr`: reciprocal rank of the first expected chunk/doc.
- `citation_accuracy`: whether predicted citations overlap expected citations or expected docs.
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

## Report Shape

Evaluation reports include:

- `run_metadata`
- `metrics`
- `cost_summary`
- `failures`
- `predictions`

Benchmark runs additionally write `summary.json`, `summary.csv`, and `summary.md`.
