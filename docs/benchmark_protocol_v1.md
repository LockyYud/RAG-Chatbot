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

Suite files under `ragbench/evaluation/protocol/` are executable contracts. A
`claim_eligible` run must include its required baselines and uses the locked
dataset, mode, and cutoff. The runner records `claim_eligibility`; it is false
for a dirty worktree, fallback component, non-full corpus policy, missing
baseline, failed run, or insufficient query count.

Each frozen suite must declare the processed dataset fingerprint, a
`reference_baseline`, `cutoffs`, `bootstrap_samples`, a `coverage` block, and a
decision rule. Reports persist the suite fingerprint. They also expose separate
`quality_claim_eligible`, `cost_claim_eligible`, and
`production_claim_eligible` verdicts: the latter two additionally require known
pricing plus index and latency instrumentation. `protocol_eligible` means the
run obeyed the frozen contract. It does **not** mean any candidate won.

`improvement_supported` applies whichever decision rule the suite declared:

- **Default** — the single named `primary_metric` shows a paired CI95
  improvement of at least `minimum_effect`, *and* no metric listed in
  `non_inferiority` regressed past its allowed margin. A CI that merely clears
  zero is not enough, and a candidate cannot buy a win on one metric by
  quietly cratering the others.
- **`pareto_improvement: true`** — every metric in `primary_metrics` holds or
  improves, and at least one improvement is real (CI95 clears zero). Use this
  only when there genuinely is no single decision metric worth elevating.

`vi_retrieval_core` currently ships at `tier: exploratory`: its decision rule
has not been calibrated against a pilot run yet, and the repo has no dataset
labelled `protocol_split: "test"`. Both are prerequisites, not paperwork — see
the promotion checklist below.

`vi_wiki_retrieval` is labelled `protocol_split: "dev"`. Git history shows no
sign it has been tuned against — no retrieval default changed after it landed,
no committed benchmark results exist — but the calibration pilot is deliberately
going to read confidence intervals and may well lead to config changes, which is
tuning. Keeping it as the tuning set preserves the option of a genuine held-out
test set later; labelling it `"test"` would spend the only labelled data in the
repo on a claim set that the next phase would immediately contaminate.

## Promoting a suite to `claim_eligible`

A `claim_eligible` suite must declare a decision rule (`primary_metric` plus
`minimum_effect`, or `pareto_improvement: true`), a `coverage` block, and it
must run against a dataset labelled `metadata.protocol_split: "test"`. Those
values decide what counts as evidence, so choosing them to make the validator
pass — rather than from the data — quietly defeats the whole tier.

That creates a genuine ordering problem: a defensible `minimum_effect` needs a
sense of the real effect sizes and their variance, which needs a run, which the
validator blocks. Resolve it by running the pilot at a lower tier, where none
of these fields are required:

1. Run the suite at `tier: exploratory`. Nothing about it is claimable, which
   is exactly the point — this pass exists to measure, not to conclude.
2. Read the paired CI95 widths from the report. `minimum_effect` should be an
   effect that matters for the application and is larger than the noise floor
   the pilot revealed; record the reasoning in the suite's `description`.
3. Derive the `coverage` numbers from the dataset's actual composition
   (`ragbench dataset validate` reports the slice counts), not from whatever
   the pilot run happened to clear. Each profile must declare the thresholds
   that constrain the slices it publishes; an empty or off-profile block is
   rejected, because it would leave the slice on the weak "at least one
   qualifying item" fallback:

   | Evaluation profile | Required `coverage` keys |
   | --- | --- |
   | `retrieval` | `min_retrieval_coverage` |
   | `single_hop_rag` | `min_retrieval_coverage`, `min_unanswerable_questions` |
   | `multi_hop_rag` | + `min_multi_hop_questions` |
   | `citation_rag` | + `min_citation_coverage` |

   `min_retrieval_coverage` is required for every profile because every report
   publishes recall/nDCG/MRR, and `min_unanswerable_questions` for the RAG
   profiles because abstention metrics are vacuous on a dataset with no
   unanswerable items. Declaring `0` is a valid answer — "this suite
   deliberately does not measure that slice" — but it has to be stated rather
   than left to a default.
4. Set `non_inferiority` margins for the metrics that must not silently
   regress while the primary metric improves.
5. Label the dataset with `ragbench dataset prepare --protocol-split {dev,test}`.
   `metadata.split` records the *upstream* split name and asserts nothing;
   `metadata.protocol_split` is this protocol's own declaration and only
   `"test"` clears a claim. Neither is ever inferred from the other — upstream
   "validation" is a dev split under another name, and upstream "test" does not
   mean *this* project held it out. Adapters therefore never emit the field,
   and it stays absent unless you pass the option: an unlabelled snapshot reads
   as "unsupported", not as "probably fine". Label `"test"` only if config,
   prompts, and chunk sizes were never chosen against that snapshot.
   Relabelling costs nothing — the dataset fingerprint covers documents,
   queries, and qrels only, so no suite needs re-freezing.
6. Flip `tier` to `claim_eligible` and record `config_frozen_at` (and
   `tuned_on_dataset`, if any) so a reader can see the config was frozen
   before the claim run rather than after seeing its results.

Step 5 is the one that cannot be automated away. Only the person who prepared
the dataset knows whether it has been tuned against, which is why an absent
`protocol_split` is treated as "unsupported", not as "probably fine".

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
- Correctness and faithfulness are two independent provider calls, so report
  and aggregate them independently: `correctness_judge_failure_rate` and
  `faithfulness_judge_failure_rate` alongside the blanket rate, and each mean
  averaged over the queries whose *own* judge call succeeded. A failure in one
  half must never shrink the sample size of the other.
- Calibrate an automated judge against a human-reviewed sample before using it
  for ranking. Blind the technique identity and randomize candidate ordering
  for pairwise review.
