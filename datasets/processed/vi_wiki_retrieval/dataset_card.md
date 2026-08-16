# vi_wiki_retrieval

This processed dataset is an evaluation fixture for rag-pipeline-lab research benchmarks.
It is not part of the user document ingestion path.

## Summary

- Documents: 2490
- Queries: 2048
- Qrels: 4096
- Source: mteb/VieQuADRetrieval
- License: MIT according to Hugging Face dataset card
- Corpus policy: full_upstream_corpus

## Files

- `documents.jsonl`: canonical retrievable documents.
- `queries.jsonl`: canonical evaluation questions.
- `qrels.jsonl`: query-to-document relevance labels.
- `docs/`: markdown export used by ingest.
- `qa.jsonl`: evaluation export used by eval.
