# Golden Benchmark Implementation Notes

Status of the human-labelled golden benchmark work: Zalo Legal Retrieval and
UIT-ViQuAD full-RAG, on top of the existing synthetic benchmark track. This
supplements `docs/dataset_strategy.md` and `docs/benchmark_protocol_v1.md`
with the concrete adapter/provenance/validation decisions made while
implementing it; it does not restate the broader dataset survey in those
documents.

## Decision: Zalo AI Challenge 2021 for `vi_legal_retrieval`

`docs/dataset_strategy.md` recommends `YuITC/Vietnamese-Legal-Documents` for
the `vi_legal_retrieval` pack. The adapter actually implemented
(`ragbench/datasets/adapters/zalo_legal_retrieval.py`) uses
[`minhnguyent546/zalo-ai-legal-text-retrieval-2021`](https://huggingface.co/datasets/minhnguyent546/zalo-ai-legal-text-retrieval-2021)
instead, because it ships as a ready-made retrieval triplet (corpus/queries/qrels)
with human-labelled qrels already split into upstream `train`/`test` subsets —
no additional curation needed to get a held-out evaluation split. This is a
correction to `dataset_strategy.md`'s source recommendation for that pack, not
a new pack; that document's table should be updated separately to point at
this source.

## Adapter conventions established here

- **Revision pinning**: every HF-backed adapter should pass `revision=<pinned
  commit sha>` to `load_hf_dataset` and record it as `metadata.source_revision`,
  so a snapshot reproduces years later even if upstream repacks the repo. See
  `zalo_legal_retrieval.py` and `uit_viquad.py`.
- **Testable loaders**: an adapter exposes a small private loader function
  (`_load_zalo_triplet`, `_load_uit_viquad_rows`) that tests monkeypatch
  directly, so adapter unit tests never hit the network.
- **Document identity must key off content, not per-question ids**: UIT-ViQuAD
  rows are one-per-question, but many questions share a context paragraph.
  `_context_id()` hashes the context text itself (with `title` only as a
  readable prefix) — keying off `uit_id`/`id` instead would mint a duplicate
  "document" per question sharing a paragraph.
- **Unanswerable questions carry no ground-truth answer, defensively**: some
  upstream unanswerable rows still ship an `answers` key (e.g. `{"text": []}`),
  which naively passed through `answer_text()` becomes `""` rather than
  `None`. Adapters must force `ground_truth_answer=None` when
  `is_answerable` is `False` rather than deriving it from `answers`, since a
  non-`None` value there would leak into `exact_match`/`token_f1` scoring.
- **Provenance is a warning, not a gate**: `validate_processed_dataset`
  reports `annotation_type` and a `provenance_warnings` list (missing
  `source`/`source_revision`/`annotation_type`) without raising, so datasets
  prepared before these fields existed keep validating.

## Implementation status

Done (see `ragbench/datasets/adapters/zalo_legal_retrieval.py`,
`ragbench/datasets/adapters/uit_viquad.py`, `ragbench/datasets/schema.py`,
`ragbench/evaluation/metrics/qa.py`):

- Zalo Legal Retrieval adapter, registered as `zalo_legal_retrieval`.
- UIT-ViQuAD provenance cleanup: pinned revision, `annotation_type`,
  `task`, `supports_unanswerable`, document-identity fix, unanswerable
  ground-truth fix.
- `exact_match` / `token_f1` deterministic QA metrics for extractive items.
- Provenance warnings surfaced from `validate_processed_dataset`.

Not started: `vi_legal_retrieval_v1.yaml` / `vi_full_rag_v1.yaml` protocol
suites (blocked on actually running `dataset prepare` against the real HF
sources and committing the result — see `tests/test_committed_protocols.py`),
the 100-query pilot, human audit sampling, and the full benchmark run.
