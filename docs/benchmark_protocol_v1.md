# Benchmark Protocol v1

This document defines the minimum evidence required before a technique is
described as an improvement in `rag-pipeline-lab`.

## Suites

| Suite | Purpose | Corpus policy | Primary metrics |
| --- | --- | --- | --- |
| `smoke` | CLI and artifact regression | tiny fixture | pass/fail only |
| `vi_retrieval_core` | deterministic retriever comparison | full upstream corpus | nDCG@10, Recall@5/10/20, MRR@10 |
| `vi_retrieval_llm` | query-transform methods | same frozen corpus/query subset | retrieval metrics plus cost and p95 latency |
| `vi_rag_gold` | end-to-end answer quality | domain corpus plus human labels | claim correctness, completeness, grounding, abstention |

The smoke fixture must never be used as empirical evidence. It exists solely
to prove that the benchmark contract runs.

Suite files under `evaluation/protocol/` are executable contracts. A
`claim_eligible` run must include its required baselines and uses the locked
dataset, mode, and cutoff. The runner records `claim_eligibility`; it is false
for a dirty worktree, fallback component, non-full corpus policy, missing
baseline, failed run, or insufficient query count.

Each frozen suite must declare the processed dataset fingerprint, a
`reference_baseline`, `cutoffs`, and `bootstrap_samples`. Reports persist the
suite fingerprint. They also expose separate `quality_claim_eligible`,
`cost_claim_eligible`, and `production_claim_eligible` verdicts: the latter
two additionally require known pricing plus index and latency instrumentation.
`protocol_eligible` means the run obeyed the frozen contract. It does **not**
mean any candidate won. `improvement_supported` is only true when at least one
declared primary metric has enough paired observations and a CI95 entirely in
the favourable direction against `reference_baseline`.

## Retrieval protocol

- When sampling queries, preserve the complete source corpus. Never reduce it
  to documents with positive qrels: that removes hard negatives.
- Freeze corpus, query IDs, seed, chunker, embedding model, cutoff, and
  resource budget before comparing systems.
- Report query-level results, macro averages by dataset/question type, and
  paired bootstrap 95% confidence intervals for candidate-minus-baseline.
- A quality claim requires a confidence interval that excludes zero and must
  disclose latency, cost, index size, and index time trade-offs.
- For production latency, record an explicit warm-up count and repeated
  measurements per query. The median is the reported query latency; keep
  evaluator and repeated-measurement spend separate from one-request pipeline
  cost.

## End-to-end protocol

Create a reviewed golden set from `datasets/golden/TEMPLATE.jsonl`. Each
answerable item needs atomic required claims and source evidence spans. Include
single-hop, multi-hop, unanswerable, numeric/table, vocabulary-mismatch, and
distractor-heavy questions. Two reviewers should independently review a sample
before relying on an LLM judge.

Document citation identity metrics (`citation_document_*`) only establish that
a cited document matches a gold document. They are not entailment metrics.
Claim-to-span support must be assessed separately by a calibrated judge or
human review.

## Judge protocol

- Save model, temperature, prompt fingerprint, and status per query.
- Treat `parse_failure` and `provider_failure` as evaluator failures, not a
  quality score of zero; report `judge_failure_rate` separately.
- Calibrate an automated judge against a human-reviewed sample before using it
  for ranking. Blind the technique identity and randomize candidate ordering
  for pairwise review.
