# Research Workflow Contract

This repository is intentionally code-first: a paper is implemented in one
readable `techniques/<id>/pipeline.py`. The framework standardizes only the
boundaries required for fair experiments.

## Paper to result

1. Read the paper and state its fidelity level in `technique.yaml`.
2. Declare supported evaluation profiles and external requirements in
   `capabilities` / `requires`.
3. Implement `ingest()` and `query()` through `BasePipeline`.
4. Run `ragbench doctor --technique <id>` before an online or optional-model run.
5. Run the same frozen dataset/profile as the baseline.
6. Use `ragbench experiment` for repeated trials; inspect per-trial summaries
   and baseline deltas before making a claim.

## Evaluation profiles

- `retrieval`: labelled document/chunk retrieval only.
- `single_hop_rag`: one-evidence answer generation and abstention.
- `multi_hop_rag`: requires examples with multiple expected evidence IDs and
  reports complete, partial, and zero-evidence rates.
- `citation_rag`: requires expected citations and enables citation metrics.

Profiles are a compatibility contract, not a claim that a paper is faithfully
reproduced. A technique must declare each profile it supports.

## Artifact bundle

Each v4 artifact records the locked technique config, corpus fingerprint,
pipeline source fingerprint, relevant dependency versions, and checksums of
every persisted file. Embeddings are stored in a sibling `embeddings.npy`
(float32, node-order aligned) rather than inline in `nodes.json`. A technique
may write custom state (tree, graph, token index, checkpoint metadata) into
its artifact directory before `save_nodes()`;
the bundle inventory will include and validate it on load.

Use `ragbench artifacts inspect` to inspect the manifest. Do not silently load
an artifact after changing its index files or locked config.

## Experiment discipline

Compare only runs with the same dataset fingerprint, evaluation profile, mode,
and cutoff. `ragbench bench` emits deltas from the first successful baseline;
`ragbench experiment --trials N --seed S` keeps each trial in a separate
directory and writes `matrix.json`.

Provider-dependent techniques must be run in strict mode for benchmark claims.
Fallback implementations are for interactive exploration only and are recorded
in result metadata.
