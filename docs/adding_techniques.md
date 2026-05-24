# Adding Techniques

Techniques live under `techniques/<technique_id>/` and should be runnable through the shared pipeline rather than custom
scripts.

## Required Files

```text
techniques/<technique_id>/
  README.md
  technique.yaml
  config.yaml
  custom/
    register.py
```

Use `custom/register.py` only when built-in registry components are not enough.

## Config Contract

Pipeline stages are configured by type and params:

```json
{
  "processing": {
    "parser": {"type": "markdown"},
    "chunker": {"type": "recursive", "params": {"chunk_size": 220, "overlap": 30}},
    "enrichers": [{"type": "section_title"}]
  },
  "indexing": {
    "embedding": {"type": "openai", "params": {"model": "${OPENAI_EMBEDDING_MODEL:-text-embedding-3-small}"}},
    "store": {"type": "json_memory"}
  },
  "inference": {
    "retriever": {"type": "openai_dense", "params": {"top_k": 5}},
    "reranker": {"type": "none"},
    "context_builder": {"type": "citation_context"},
    "generator": {"type": "openai_chat"},
    "verifier": {"type": "citation_coverage"}
  }
}
```

Supported vector stores:

- `json_memory`: zero-dependency default for small experiments.
- `faiss_local`: optional local backend for larger dense retrieval experiments.

## Acceptance Checklist

- Technique metadata states whether it is baseline, production pattern, paper-inspired, or concept-only.
- Config validates and runs through `raglab ingest` and `raglab eval`.
- Technique is compared against at least one baseline on the same QA set.
- README documents weak cases and what is not reproduced.
- Custom code records useful runtime metadata where relevant.
