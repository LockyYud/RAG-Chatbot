# Adding Techniques

Techniques are bundled Python packages under `techniques/<technique_id>/` and are discovered through package resources.

## Required files

```text
techniques/<technique_id>/
  __init__.py
  pipeline.py
  technique.yaml
  README.md
```

Copy `techniques/_template`, set an implementation level (`faithful_reproduction`, `paper_inspired`,
`production_pattern`, `concept_only`, or `baseline`), and implement a concrete `BasePipeline`.

Declare `capabilities.evaluation_profiles` in `technique.yaml` (`retrieval`, `single_hop_rag`,
`multi_hop_rag`, and/or `citation_rag`) and whether the technique writes `custom_artifacts`. Run
`raglab doctor --technique <id>` before a provider-dependent benchmark.

Every constructor parameter must be assigned to an attribute with the same name so `resolved_config()` can persist the
complete configuration in artifact v3. Declare `query_override_fields` explicitly. Only fields that do not change the
persisted index belong there; embedding model, chunking, enrichment, and store parameters must remain locked.

All three methods must use the shared contracts:

- `ingest()` calls `build_ingest_manifest(..., pipeline_config=self.resolved_config())` and saves artifact v3.
- Custom index state (for example a graph, tree, or token index) is written under the artifact directory before
  `save_nodes()`; v3 records and validates checksums for every file in that bundle.
- `load(artifact_path)` calls `self.load_artifact(artifact_path)` (validates config/corpus drift) and builds every
  retriever, reranker, vector store, or tool the technique needs, storing them on `self`. Finish with
  `self._mark_loaded(artifact_path, manifest, nodes)`. Anything expensive — a learned reranker's model load, an index
  built from `nodes` — belongs here, not in `query()`, since `load()` runs once per artifact while `query()` runs once
  per question.
- `query(question, mode)` calls `self._require_loaded()` first, then runs retrieval/generation/verification against the
  state `load()` built. It does **not** take `artifact_path` and does **not** re-read the artifact.
- `retrieval_only` performs retrieval/context construction only and returns `skipped_verification()`.
- Public citations are canonical document IDs; `[C1]` markers are prompt-local identifiers.
- Learned rerankers fail in strict mode. Demo fallback must be explicitly enabled and reported as the effective component.
- If a technique calls a chat model during retrieval itself (not just full-RAG answer synthesis — e.g. HyDE, RAG-Fusion),
  declare those constructor attribute names in `retrieval_time_models` so `doctor`/preflight require that provider key
  even in `retrieval_only` mode.

## Acceptance checklist

- `raglab techniques list` discovers the technique from an installed wheel and outside the checkout.
- Artifact mismatch, locked overrides, missing dependencies, and invalid modes fail with actionable messages.
- Offline tests inject fake providers; they do not download models or call APIs.
- The technique is compared with at least one baseline on the same frozen dataset and report schema.
- README states fidelity level, what is not reproduced, strong cases, weak cases, and expected cost.
