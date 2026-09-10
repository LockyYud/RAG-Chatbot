# Agentic RAG (A-RAG-inspired, 2026)

## Source
- A-RAG: *Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces*, 2026 — <https://arxiv.org/abs/2602.03442>
- Survey: *Reasoning RAG via System 1 or System 2*, 2025 — <https://arxiv.org/abs/2506.10408>
- Background blog: [`docs/blogs/agentic_rag_2026.md`](../../docs/blogs/agentic_rag_2026.md)

## Core Idea
The 2026 shift in RAG is one of **control flow**: instead of a fixed
`retrieve → generate`, an LLM agent interleaves reasoning and retrieval — it
chooses *which* tool, *what* to query, and *when* it has enough evidence to
answer. This is the **training-free, inference-time** version (no RL, unlike
Search-R1 / AutoSearch), so it runs on any chat model.

## Stage Changed
Retrieval orchestration. A controller loop sits above the existing retrievers;
indexing, reranking, context construction and generation are reused unchanged.

## What This Repo Implements
- `AgenticRetrievalController` (`ragbench/inference/controllers/agentic.py`): a
  reusable, training-free loop with an **injectable policy** (LLM by default,
  scriptable for tests), `max_steps`/evidence guards, and a structured trace.
- Tools are the retrievers the repo already has:
  - `keyword` → `BM25Retriever`
  - `semantic` → `DenseRetriever`
  - `hybrid` → `RRFHybridRetriever`
  - `chunk_read` → expand a known node to its full parent section
- After the loop, evidence is reranked by the shared `CrossEncoderReranker` and
  the answer is synthesised by the standard `ChatGenerator` over a citation
  context — keeping retrieval/citation metrics comparable to other techniques.

## What This Repo Does Not Reproduce
- No RL training of the policy (that is Search-R1 / AutoSearch territory; it needs
  training infrastructure this lab intentionally avoids).
- No learned stopping criterion — the agent decides to stop via prompting, not a
  trained value head.

## Expected Strengths
- Multi-hop questions: the agent can issue follow-up queries with new terms found
  in earlier evidence.
- Picks the right retriever per step (keyword for codes, semantic for paraphrase).
- Emits an auditable `metadata.agent.trace` (steps, tools, subqueries, evidence).

## Expected Failure Modes
- Cost/latency: several LLM calls per question (one per step + generation).
- On simple single-fact lookups a single `hybrid` pass is cheaper and as good.
- Quality is bounded by `agent_model`'s tool-use/reasoning ability.

## Config
| Param | Default | Role |
| --- | --- | --- |
| `agent_model` | `gpt-4.1-mini` | LLM driving the retrieval loop |
| `max_steps` | `4` | Hard cap on reasoning/retrieval steps (guard) |
| `per_tool_top_k` | `5` | Results pulled per tool call |
| `rerank_top_k` | `6` | Final evidence kept after cross-encoder rerank |
| `embedding_model` | `text-embedding-3-small` | dense/hybrid embedding model |

## Benchmark Results
Compare against the single-pass hybrid baseline:

```bash
ragbench bench --techniques bm25_hybrid_rerank agentic_rag_arag \
  --docs datasets/sample/docs --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/agentic --mode full_rag
```

Inspect the loop on one question:

```bash
ragbench query --technique agentic_rag_arag --artifact artifacts/agentic \
  --query "..." --mode full_rag    # see answer.metadata.agent.trace
```

## Implementation Notes
- Requires embeddings + a chat model; the agent uses the LLM even in
  `retrieval_only` mode (that *is* the technique).
- The controller is engine-level and reusable: future CRAG / Adaptive-RAG /
  IRCoT techniques can share it by swapping the policy, per the roadmap's
  "add an orchestrator only when ≥2 techniques need it" rule.
