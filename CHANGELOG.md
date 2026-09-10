# Changelog

## Unreleased

Follow-up to the 0.3.0 review: a suite regression introduced by 0.3.0's own
validator, plus the two claim-validity gaps that review surfaced.

### Fixed

- **Synthetic benchmarks scored every technique at exactly 0, silently.**
  `dataset generate` wrote its own chunk ids (`doc:synthetic:0001`) into
  `expected_chunk_ids`, while every pipeline emits `doc:c1:<sha1>`
  (`processing.chunkers.common.chunk_id`). `evaluate_prediction_rows()` prefers
  chunk labels whenever they are present, so the two id families never
  intersected and recall/nDCG/MRR/MAP were 0 for every technique — while
  `retrieval_evaluated` stayed `True`, so a report of all-zero scores looked
  like a real result. Labels are now document-level, which is the only
  identifier that survives across techniques that chunk differently; the source
  chunk is kept in metadata as provenance. `metrics.retrieval_label_level`
  reports which identifier the qrels use, so the failure is visible rather than
  silent, and `dataset validate-synthetic` rejects a generated set that carries
  chunk labels at all.
- **The default judge model was also the default question generator.** Both
  defaulted to `gpt-4.1-mini`, so one model wrote the questions, wrote the
  ground truth, and then graded answers against them. `eval`, `bench` and
  `experiment` now refuse to judge a generated dataset with a model that
  generated it, unless `--allow-same-model` is passed — which records the
  circularity and downgrades the trust verdict rather than hiding it.


- **`vi_retrieval_core` could not be loaded at all.** 0.3.0 tightened
  `load_suite()` to require a decision rule (`primary_metric` +
  `minimum_effect`, or `pareto_improvement`) for claim-eligible suites but did
  not update the one committed claim-eligible suite, so `load_suite()` raised
  on it. Nothing caught this because no test and no CI step ever loaded a file
  from `ragbench/evaluation/protocol/`. `tests/test_committed_protocols.py`
  now loads every committed protocol file (with an explicit allowlist for
  `standard.yaml`, which is a metrics reference document rather than a suite).
- **A failure in one LLM judge no longer discards the other's score.** The
  judge makes two independent provider calls, but aggregation gated all four
  score fields on the blanket `judge_status`, which is "not ok" when *either*
  half failed — so a malformed faithfulness response threw away a valid
  `answer_correctness`. Each metric is now averaged over the queries whose own
  sub-judge succeeded, and `correctness_judge_failure_rate` /
  `faithfulness_judge_failure_rate` are reported alongside the blanket rate.

### Added

- **Synthetic benchmark generation v2** (`docs/synthetic_benchmark_v2.md`).
  `dataset generate` now runs a seven-stage pipeline instead of one prompt:
  chunk with the pipeline's own chunker, require a verbatim `answer_span`
  (a free, no-API guard against a weak generator inventing answers), verify
  answerability with a *different* model, measure lexical overlap and rewrite
  questions that copy the passage, pool BM25 (and optionally dense) candidates
  so relevance labels cover every passage that answers the question rather than
  only the one that seeded it, deduplicate, then write `qa.jsonl` beside a
  `manifest.json` recording the corpus fingerprint, every model role, and the
  reject rate of each stage. Generated sets are always `protocol_split: "dev"`;
  there is deliberately no way to label one `"test"`.
- **`dataset audit`** measures how far a generated benchmark can be trusted:
  Kendall tau-b (tie-corrected, verified against scipy) between the technique
  ranking on the synthetic set and on a human-labelled golden set, with a
  bootstrap interval resampled independently on each side, plus a
  human-reviewed label-precision pass (`--emit-label-sample` writes the review
  file). The audit is recorded on the dataset it audited, so later evals find
  it without being told where it is.
- **A `trust` block on every report built from a generated dataset**, carrying
  the verdict (`synthetic_untrusted` / `synthetic_trusted_for_ranking`), the
  reasons, and the generation statistics. No verdict ever licenses an absolute
  number: the best available outcome is trust for *ranking* techniques. An
  un-audited dataset is `synthetic_untrusted`, and so is one whose judge shares
  a model with its generator.
- **`dataset validate-synthetic`** validates a generated set's labels, split
  label and manifest.
- **Model roles are now separable** via `RAGBENCH_GENERATOR_MODEL`,
  `RAGBENCH_VERIFIER_MODEL` and `RAGBENCH_JUDGE_MODEL`.

### Breaking changes

- **`dataset generate` writes a directory, not a file.** `--output` is now a
  directory receiving `qa.jsonl` and `manifest.json`, and `--limit` (a cap on
  questions) is replaced by `--max-chunks` (a cap on input), which is the
  control that actually bounds cost. Generation also requires the generator and
  verifier to be different models unless `--allow-same-model` is passed.
- **Held-out status must be declared explicitly.** `metadata.split` now means
  only the upstream split name and no longer gates anything; the new
  `metadata.protocol_split` (`"dev"` or `"test"`) carries this protocol's own
  dev/test role. Claim-eligible runs require `protocol_split: "test"`.
  Previously only the literal string `"dev"` was rejected, so the committed
  dataset's upstream `split: "validation"` — a dev split under another name —
  passed as claim-eligible, and an unlabelled dataset did too. An absent label
  is now treated as unsupported rather than as "probably fine".
- **`coverage` is now required on claim-eligible suites, and must constrain
  the profile being run.** Without it, `validate_profile()` falls back to "at
  least one qualifying item", so a suite could be claim-eligible while a
  single query in the whole dataset carried a qrel. Requiring merely "a
  mapping" was not enough — `coverage: {}` declares nothing and left the same
  fallback in place — so each profile must declare the thresholds for the
  slices it publishes: `min_retrieval_coverage` for every profile,
  `min_unanswerable_questions` additionally for the RAG profiles,
  `min_multi_hop_questions` for `multi_hop_rag`, and `min_citation_coverage`
  for `citation_rag`. Declaring `0` is a valid explicit opt-out. `suite.profile`
  is now also validated against `suite.mode` at load time rather than at run time.
- **A declared `min_retrieval_coverage` is enforced for every profile**, not
  just `retrieval` — otherwise the requirement above would be decorative for
  RAG suites. The *undeclared* "at least one qrel" fallback stays scoped to
  the `retrieval` profile, so a RAG dataset with no qrels (a legitimate
  answer-quality-only evaluation) is still accepted.
- `vi_retrieval_core` now ships at `tier: exploratory`. Its decision rule has
  not been calibrated against a pilot run and no dataset in the repo is
  labelled `protocol_split: "test"`; see the promotion checklist in
  `docs/benchmark_protocol_v1.md`.

### Added

- `ragbench dataset prepare --protocol-split {dev,test}` declares a snapshot's
  role in the protocol. It is stamped above the adapter layer and defaults to
  absent: adapters describe upstream data and must never assert held-out
  status, since that is a decision made by whoever runs the protocol. That
  layering is enforced rather than conventional — `prepare_dataset()` rejects
  an adapter that returns `metadata.protocol_split` at all, including when the
  caller also passed the flag, because silently overwriting it would hide an
  adapter shipping `"test"` and turn a tuning set into a claim set unnoticed.
  A sample taken from a labelled dataset inherits the label — a sample of a
  tuning set is still a tuning set — and the dataset card now shows both the
  upstream split (provenance) and the protocol split (with what it means for
  claims).
- `datasets/processed/vi_wiki_retrieval` is labelled `protocol_split: "dev"`.
  Git history shows no evidence it has been tuned against, but the upcoming
  calibration pilot is expected to inform config choices, so it is the tuning
  set; a held-out claim set must be a separate snapshot. Labelling did not
  change the dataset fingerprint, so `vi_retrieval_core` still matches.

### Migration notes

- To make a suite claim-eligible again: add a `coverage` block, label the
  dataset manifest `"protocol_split": "test"` (only if config/prompts were
  never tuned against it), and follow the promotion checklist in
  `docs/benchmark_protocol_v1.md`. Adding `protocol_split` does **not** change
  the dataset fingerprint — it covers documents, queries, and qrels only.
- Eval reports gain `dataset_protocol_split`; `dataset_split` keeps its key
  but now means the upstream split name.

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
