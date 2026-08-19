# Changelog

## 0.3.0

A correctness- and reproducibility-focused release. No new RAG techniques —
this closes gaps in the benchmark/evaluation infrastructure itself so
existing and future experiment results can be trusted. Several changes are
breaking; see **Migration notes** below.

### Breaking changes

- **Package renamed `raglab` → `ragbench`.** The top-level `evaluation` and
  `techniques` packages moved under it too (`ragbench.evaluation`,
  `ragbench.techniques`). The CLI command is now `ragbench` (was `raglab`).
  `RAGLAB_*` environment variables and the `.raglab_cache/` directory name
  are unchanged for backward compatibility.
- **Artifact format bumped to v5.** `runtime.raglab_version` is now
  `runtime.package_version`; `runtime.source_fingerprint` is split into
  `runtime.ingest_fingerprint` (gates loading — re-run ingest on mismatch)
  and `runtime.runtime_fingerprint` (informational only — a
  retriever/reranker/generator/verifier change no longer forces a
  re-ingest). Existing v4 artifacts must be re-ingested.
- **`RAGAnswer.citations` is now `list[Citation]`, not `list[str]`.**
  `Citation` carries `citation_id`, `doc_id`, `chunk_id`, and optional
  `start_char`/`end_char`. Code reading `prediction.citations` as bare
  doc-id strings needs `citation.doc_id` instead.
- **`evaluation.judge.LLMJudge` now makes two LLM calls per query, not one**
  (a `CorrectnessJudge` call and a separate `FaithfulnessJudge` call — see
  below). Judge cost roughly doubles; `JudgeResult` gained
  `correctness_status`/`faithfulness_status` alongside the existing blanket
  `status`.
- **`suites.load_suite()` requires `primary_metric` + `minimum_effect`** for
  `claim_eligible`-tier suites, unless `pareto_improvement: true` is set
  instead. Existing suites with only `primary_metrics` (plural) must add
  one of these.
- **`_cost_summary`'s budget-relevant total is now `total_spend`**, not
  `total_estimated_cost` (which keeps its old "technique cost only"
  meaning as a back-compat alias). `run_eval(..., max_estimated_cost_usd=)`
  now enforces `total_spend` (technique + measurement + warmup + judge).

### Fixed

- Cost budget guard undercounted real spend whenever `warmup_queries > 0`
  or `latency_repetitions > 1` — warm-up and repeated-measurement calls
  were captured and then discarded instead of counting toward the cap.
- Benchmark latency sampling could measure a different set of questions
  before vs. after a `--resume`, silently changing what a run's headline
  `latency_ms_p95` was based on. The sampled question set is now frozen
  once (in the checkpoint header) and reused verbatim on resume.
- `TextParser`/synthetic dataset chunking used a file's bare stem as
  `doc_id`, so `legal/report.md` and `finance/report.md` under the same
  ingest root collided into one `doc_id`, silently merging their blocks.
  `doc_id` is now the path relative to the ingest root (unchanged for a
  flat, single-directory corpus).
- The LLM judge classified valid-but-schema-incomplete JSON as a
  successful judgment, defaulting missing fields to `0.0` while still
  reporting `status="ok"` — a provider drifting off-schema would silently
  drag every mean judge metric toward 0. Now classified as
  `schema_failure` and excluded from aggregates, same as a parse failure.
- `_improvement_supported()` treated *any one* of several `primary_metrics`
  clearing its confidence interval as sufficient — a candidate could raise
  one metric while regressing others and still be reported
  `improvement_supported: true`. Now requires a single named
  `primary_metric` to clear a `minimum_effect` margin, with every metric
  in `non_inferiority` held to its own regression bound (or, in
  `pareto_improvement` mode, requires every primary metric to not regress
  with at least one improving).
- A global `random.seed()` call in the benchmark runner made LLM
  retry-jitter deterministic per trial seed while never actually reaching
  the LLM provider — trial-seed reruns looked seed-controlled but weren't.
  Removed; `providers.llm_client.generation_seed()` now requests the seed
  from the provider directly (best-effort — see its docstring on why this
  is a replicate control, not a determinism guarantee) and the provider's
  `system_fingerprint` (when available) is recorded so a changed serving
  snapshot across "identical" seeded calls is detectable.
- `load_nodes()` materialized every embedding from the mmap'd `.npy` file
  into a Python list of boxed floats on load, defeating the point of
  memory-mapping for large corpora. Embeddings now stay as array views;
  `dense_cosine()` was made numpy-native so it accepts either.
- `BM25Retriever.retrieve()` rebuilt a `Counter` of term frequencies for
  every document on every query. Precomputed once in `__init__` instead.
- A benchmark comparison could pass its per-technique node-count/backend
  check while the baseline and candidate still ran on *different* vector
  store backends (one under the FAISS threshold, one over) — now flagged
  as a `production_reasons` claim-eligibility issue.

### Added

- `evaluation.profiles.validate_profile()` accepts a suite `coverage` block
  (`min_retrieval_coverage`, `min_citation_coverage`,
  `min_multi_hop_questions`, `min_unanswerable_questions`,
  `min_per_question_type`) enforcing real per-slice thresholds instead of
  "at least one qualifying item."
- `suites.claim_eligibility()` rejects a dataset whose manifest declares
  `metadata.split: "dev"` for claim-eligible runs, and surfaces
  `dataset_split`, `tuned_on_dataset`, `config_frozen_at` in its verdict —
  an unmarked dataset (every dataset in this repo today) is unaffected.
- Evaluation reports now include `run_metadata.environment` (OS, CPU,
  logical cores, RAM, GPU, torch device, numpy BLAS backend, OMP/MKL
  thread counts, chat/embed model) for cross-machine benchmark
  comparability.
- Every bundled technique's `technique.yaml` now documents a paper-fidelity
  contract (`implementation.reproduced` / `.omitted` / `.deviations`),
  enforced at load time for any `implementation.level` beyond `"baseline"`.
- `ragbench.core.io.relative_doc_id()`, `ragbench.indexing.artifacts.
  runtime_fingerprint_stale()`, `providers.llm_client.generation_seed()`.

### Migration notes

- Re-run `ingest` for every existing artifact (v4 → v5).
- If you read judge output programmatically: `answer_correctness` /
  `abstention_correctness` come from the correctness call;
  `faithfulness` / `citation_support` come from the faithfulness call. A
  failure in either sets `status` to that failure, even if the other
  succeeded — check `correctness_status` / `faithfulness_status`
  individually if you need finer granularity.
- If you read `prediction["citations"]` from a saved report: entries are
  now objects (`{citation_id, doc_id, chunk_id, start_char, end_char}`),
  not bare strings. `doc_id` is the direct replacement for the old string
  value.
- Claim-eligible suites need `primary_metric` + `minimum_effect` added (or
  `pareto_improvement: true`).
- `import raglab...` / `from evaluation...` / `from techniques...` become
  `import ragbench...` / `from ragbench.evaluation...` /
  `from ragbench.techniques...`. The `raglab` console command becomes
  `ragbench`.
