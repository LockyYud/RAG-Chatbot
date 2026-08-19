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
`ragbench doctor --technique <id>` before a provider-dependent benchmark.

Every constructor parameter must be assigned to an attribute with the same name so `resolved_config()` can persist the
complete configuration in artifact v5. Declare `query_override_fields` explicitly. Only fields that do not change the
persisted index belong there; embedding model, chunking, enrichment, and store parameters must remain locked.

All three methods must use the shared contracts:

- `ingest()` calls `build_ingest_manifest(..., pipeline_config=self.resolved_config())` and saves artifact v5.
- Custom index state (for example a graph, tree, or token index) is written under the artifact directory before
  `save_nodes()`; v5 records and validates checksums for every file in that bundle.
- If the technique embeds nodes, pass `store_backend=default_store_backend(len(nodes), has_embeddings=True)`
  (`ragbench.indexing.artifacts`) to `build_ingest_manifest(...)` and `store_spec={"type": store_backend}` to
  `save_nodes(...)` — this picks `json_memory` (numpy-vectorized exact search) or `faiss_local` based on corpus
  size instead of hardcoding a backend. Embeddings themselves are stored in a sibling `embeddings.npy`
  (float32, node-order aligned), not inline in `nodes.json`; `load_nodes()` reattaches them to
  `IndexedNode.embedding` transparently, so retrievers never need to know the storage format. Every node must
  end up embedded, or none — a partially-embedded node list is rejected at `save_nodes()`.
  See `techniques/naive_rag/pipeline.py` for the full ingest+load wiring, including passing the loaded
  `load_vector_store(artifact_path, nodes)` into the retriever at `load()` time.
- The FAISS threshold (default 2000 nodes) is overridable via the `RAGLAB_FAISS_NODE_THRESHOLD` env var, and
  automatically falls back to `json_memory` if `faiss` isn't installed (it's an optional dependency — the
  `vector` extra).
- `load(artifact_path)` calls `self.load_artifact(artifact_path)` (validates config/corpus drift) and builds every
  retriever, reranker, vector store, or tool the technique needs, storing them on `self`. Finish with
  `self._mark_loaded(artifact_path, manifest, nodes)`. Anything expensive — a learned reranker's model load, an index
  built from `nodes` — belongs here, not in `query()`, since `load()` runs once per artifact while `query()` runs once
  per question.
- `query(question, mode)` calls `self._require_loaded()` first, then runs retrieval/generation/verification against the
  state `load()` built. It does **not** take `artifact_path` and does **not** re-read the artifact.
- `query()` must not mutate `self` state beyond what's local to that call (build any per-question object — a
  policy, a controller, a running total — as a local variable, not `self.something`). This was always implied by
  `--latency-repetitions` calling `query()` repeatedly on the same instance, but `ragbench eval/bench --concurrency`
  additionally runs `query()` from multiple threads at once — shared mutable `self` state there is a data race,
  not just a repeatability bug. `techniques/agentic_rag_arag/pipeline.py` is the reference example: `load()` builds
  the shared, read-only `self._tools`/`self._reranker`, but `query()` builds a fresh policy/controller every call.
  **This rule is transitive**: it also applies to any retriever/reranker/verifier object `load()` stores on `self`
  and reuses across queries — a retriever that writes a per-call result onto its own `self.last_metadata` (or
  similar) and has the pipeline read it back afterward has exactly the same race, one level down. `HyDERetriever`/
  `RAGFusionRetriever` (`techniques/hyde_2022/pipeline.py`, `techniques/rag_fusion_2024/pipeline.py`) used to do
  this — a wide window opens between the retriever's internal LLM call and the pipeline reading its runtime
  metadata back, during which a second concurrent query could silently overwrite it. Fixed by returning
  `(results, runtime_metadata)` from `retrieve()` instead of stashing metadata on `self` — metadata now travels
  with the call instead of living in shared state. `techniques/self_rag_2023/pipeline.py`'s verifier has the same
  *shape* of code (`self.last_metadata`) but is safe today only because it's instantiated fresh inside `query()`
  rather than cached on `self` in `load()` — a fragile-by-convention pattern, not a guarantee; hoisting that
  construction into `load()` as a "perf optimization" would silently reintroduce the bug.
- `retrieval_only` performs retrieval/context construction only and returns `skipped_verification()`.
- Public citations are canonical document IDs; `[C1]` markers are prompt-local identifiers.
- Learned rerankers fail in strict mode. Demo fallback must be explicitly enabled and reported as the effective component.
- `CrossEncoderReranker` (`ragbench/inference/rerankers/cross_encoder.py`) supports two backends:
  `backend="local"` (default, a `sentence-transformers` model loaded once in `load()`) or
  `backend="api"` (a hosted rerank endpoint called per query via `LLMClient.create_rerank()`,
  e.g. `model="cohere/rerank-english-v3.0"`). A technique exposing this choice should declare a
  `reranker_backend` constructor attribute (query-overridable, same as `reranker_model`) and call
  `check_provider_ready(self.reranker_model)` in `load()` unconditionally — it is a no-op for a
  local model id (unknown prefix) and required for an API model id. `ragbench doctor` checks
  `sentence-transformers` availability for `backend="local"` or the provider API key for
  `backend="api"` automatically once the pipeline has a `reranker_model` attribute.
- If a technique calls a chat model during retrieval itself (not just full-RAG answer synthesis — e.g. HyDE, RAG-Fusion),
  declare those constructor attribute names in `retrieval_time_models` so `doctor`/preflight require that provider key
  even in `retrieval_only` mode.

## Acceptance checklist

- `ragbench techniques list` discovers the technique from an installed wheel and outside the checkout.
- Artifact mismatch, locked overrides, missing dependencies, and invalid modes fail with actionable messages.
- Offline tests inject fake providers; they do not download models or call APIs.
- The technique is compared with at least one baseline on the same frozen dataset and report schema.
- README states fidelity level, what is not reproduced, strong cases, weak cases, and expected cost.
