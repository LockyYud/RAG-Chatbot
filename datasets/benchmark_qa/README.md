# Benchmark QA Dataset

Put the real evaluation set here once you have 50-100 labeled questions.

Expected JSONL format:

```json
{"question_id":"q001","question":"...","ground_truth_answer":"...","expected_doc_ids":["doc_id"],"expected_chunk_ids":[],"expected_citations":["doc_id:section"],"metadata":{"type":"FACTUAL"}}
```

The small sample dataset in `datasets/sample_qa/qa.jsonl` is only for smoke tests and demos. It is not statistically meaningful.
