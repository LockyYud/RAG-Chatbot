# `traceable_failure_taxonomy`: Phân Loại Lỗi RAG Có Thể Trace và Đo Lường

> **Pattern**: Taxonomy phân loại lỗi RAG có thể trace và đo lường trong production
> **Nguồn**: Production engineering pattern — tổng hợp từ thực tiễn và nhiều nguồn research
> **Loại**: `production_pattern`

RAG systems thất bại theo nhiều cách khác nhau, nhưng hầu hết teams chỉ track một số tổng hợp như "accuracy" hay "user satisfaction". Không có failure taxonomy rõ ràng, engineer không thể tối ưu đúng chỗ: accuracy thấp có thể do retriever không tìm được passage đúng, do context window ordering đẩy evidence vào middle bị bỏ qua, do LLM hallucinate ngay cả khi context đúng, hay do corpus không chứa thông tin. Taxonomy này cung cấp 6 failure classes có thể trace bằng logs, đo bằng metrics, và debug một cách systematic.

---

## 1. Bối cảnh và Động lực

### Tại sao "accuracy giảm 3%" không đủ thông tin

Giả sử RAG pipeline của bạn có accuracy 72% trên evaluation set và bạn muốn improve. Bạn có thể:
- Thay retriever
- Thay embedding model
- Tăng K (số passages retrieved)
- Thay LLM
- Thêm reranker
- Thay chunking strategy
- Thêm query rewriting

Mỗi quyết định tốn thời gian và tiền. Nếu bottleneck là retriever miss rate thì thay LLM không giúp. Nếu vấn đề là LLM hallucinate khi context đủ tốt thì tăng K không giúp.

**Failure taxonomy là diagnostic framework:** Biết chính xác failure đến từ đâu → invest optimization effort đúng chỗ.

### Failure modes chính trong RAG pipeline

```
Query
  │
  ▼
[RETRIEVER] ─── Failure 1: Retrieval Miss
  │
  ▼
[CONTEXT BUILDER] ─── Failure 2: Citation Miss (context ordering)
  │
  ▼
[LLM GENERATOR] ─── Failure 3: Unsupported Claim (hallucination with context)
                └─── Failure 4: Abstention Failure (shouldn't answer but does)
                └─── Failure 5: Retrieval Noise Propagation
                └─── Failure 6: Cross-Document Synthesis Failure
```

---

## 2. Sáu Failure Classes

### Class 1: Retrieval Miss

**Định nghĩa:** Relevant document *tồn tại* trong corpus nhưng *không có* trong top-K retrieved set.

**Root causes:**
- **Semantic miss:** Query và document dùng vocabulary khác nhau — embedding không bridge được gap. Ví dụ: query "what causes heart attack" nhưng document nói "myocardial infarction risk factors"
- **Coverage gap:** Document tồn tại nhưng không trong corpus (outdated data, missing sources)
- **Ranking failure:** Document retrieved nhưng bị đẩy xuống dưới rank K do noise

**Metrics:**
- **Recall@K:** Tỷ lệ relevant documents trong top-K (thường K = 5-10)
- **Evidence Coverage Rate:** % câu hỏi có ít nhất 1 relevant passage trong top-K

**Trace requirements:**
```
Log format for each query:
{
  "query_id": "q123",
  "query": "what causes heart attack",
  "retrieved_doc_ids": ["d1", "d5", "d12", "d8", "d3"],
  "retrieved_scores": [0.89, 0.84, 0.81, 0.79, 0.77],
  "relevant_doc_ids": ["d7", "d15"],  # from offline annotation
  "recall_at_5": 0  # neither d7 nor d15 in top-5
}
```

**Debug signal:** Recall@K thấp → investigate: embedding model, BM25 coverage, chunk size, query preprocessing.

---

### Class 2: Citation Miss

**Định nghĩa:** Relevant document *có* trong retrieved context nhưng LLM *không dùng* nó khi generate answer.

**Root causes:**
- **Lost-in-middle effect:** Passage relevant nằm ở giữa context window bị under-attended (xem [lost_in_middle_context_ordering.md](./lost_in_middle_context_ordering.md))
- **Relevance dilution:** Quá nhiều passages, LLM bị overwhelmed bởi noise
- **Format mismatch:** Passage chứa evidence nhưng ở format khó parse (bảng, list, code)

**Metrics:**
- **Citation Recall:** Số relevant passages được cite / số relevant passages trong context
- **Citation Precision:** Số cited passages thực sự relevant / tổng số cited passages

**Trace requirements:**
```
Log format:
{
  "query_id": "q123",
  "context_doc_ids_ordered": ["d7", "d3", "d15", "d1", "d5"],
  "generated_answer": "...",
  "cited_doc_ids": ["d3", "d1"],  # extracted from answer citations or attribution
  "relevant_doc_ids_in_context": ["d7", "d15"],
  "citation_recall": 0.0,  # d7 and d15 retrieved but not cited
  "position_of_relevant": {"d7": 0, "d15": 2}  # positions in context
}
```

**Debug signal:** Citation Recall thấp mặc dù Recall@K cao → investigate: context ordering, context length, position of relevant passages, reranking.

---

### Class 3: Unsupported Claim

**Định nghĩa:** LLM generate claim *không có* trong retrieved context — hallucination mặc dù context được cung cấp.

**Phân loại:**

| Sub-type | Mô tả |
|---------|-------|
| **Confabulation** | Claim hoàn toàn không có trong context, sai hoàn toàn |
| **Over-generalization** | Context nói "X trong trường hợp A", LLM generate "X nói chung" |
| **Entity substitution** | Sai tên, số liệu, ngày tháng — hallucinate specific facts |
| **Unsupported inference** | Suy luận không justified bởi context (mặc dù context đúng) |

**Metrics:**
- **Faithfulness Score:** % atomic claims trong answer được support bởi context
- **Hallucination Rate:** % answers có ít nhất 1 unsupported claim

**Measurement approach (aligned với [ragas_eval.md](./ragas_eval.md)):**
```
For each generated answer:
1. Decompose answer thành atomic claims: ["X is Y", "Z happened in 2020", ...]
2. For each claim, verify support trong retrieved context
3. Faithfulness = supported_claims / total_claims
```

**Trace requirements:**
```
{
  "query_id": "q123",
  "answer": "...",
  "atomic_claims": ["claim1", "claim2", "claim3"],
  "claim_support": {
    "claim1": {"supported": true, "source_passage": "d7", "quote": "..."},
    "claim2": {"supported": false, "type": "confabulation"},
    "claim3": {"supported": true, "source_passage": "d15", "quote": "..."}
  },
  "faithfulness": 0.67
}
```

**Debug signal:** Faithfulness thấp mặc dù citation recall cao → investigate: context window too long (dilutes focus), LLM instruction following, temperature.

---

### Class 4: Abstention Failure

**Định nghĩa:** LLM *nên* từ chối trả lời (vì corpus không chứa thông tin) nhưng lại generate câu trả lời confident-sounding có thể sai.

**Ví dụ:**
- Query về sự kiện xảy ra sau knowledge cutoff của corpus
- Query về domain-specific knowledge không có trong corpus
- Query mơ hồ cần clarification

**Metrics:**
- **Abstention Rate trên OOC (Out-of-Corpus) queries:** % queries không có answer trong corpus mà model từ chối (desired behavior)
- **False Confidence Rate:** % OOC queries mà model generate sai answer với high confidence

**Tạo OOC test set:**
```
Sampling strategy:
1. Random sample queries → verify NOT answerable từ corpus
2. Create adversarial queries về recent events beyond corpus date
3. Create queries về tangential topics (close domain but not in corpus)
```

**Trace requirements:**
```
{
  "query_id": "q456",
  "query": "...",
  "is_in_corpus": false,  # ground truth: answer không tồn tại trong corpus
  "answer": "Based on the provided information, ...",  # model answered confidently
  "abstained": false,  # should have abstained
  "abstention_failure": true
}
```

**Debug signal:** Abstention Failure rate cao → investigate: system prompt về uncertainty/abstention, LLM tendency toward overconfidence, retrieval score threshold cho "no answer found".

---

### Class 5: Retrieval Noise Propagation

**Định nghĩa:** Irrelevant retrieved passages *làm confuse* LLM, dẫn đến wrong answer ngay cả khi correct passage cũng có trong context.

**Đây là distinct từ Citation Miss:** Trong Citation Miss, relevant passage bị ignored. Trong Noise Propagation, irrelevant passage được "consumed" và interferes with correct answer.

**Ví dụ:**
Query: "Công ty X thành lập năm nào?"
- Passage 1 (relevant): "Công ty X được thành lập năm 2010 bởi..."
- Passage 2 (noise): "Công ty X đã được tái cơ cấu năm 2015, sau khi..." (về cùng topic nhưng confusing)
- Answer: "2015" — sai, bị confuse bởi noise passage

**Metrics:**
- **Noise Sensitivity:** Accuracy drop khi thêm irrelevant passages so với clean context
- Đo bằng controlled experiment: run same queries với (a) only relevant passages, (b) relevant + K noise passages

**Trace requirements:**
```
{
  "query_id": "q789",
  "gold_answer": "2010",
  "answer_with_clean_context": "2010",  # correct
  "answer_with_full_context": "2015",   # wrong - noise interference
  "noise_passages": ["d_noise1", "d_noise2"],
  "noise_propagation_detected": true
}
```

**Debug signal:** Noise Sensitivity cao → investigate: reranking quality (noise passages shouldn't be top-K), context length (fewer passages = less noise), reranker với high precision.

---

### Class 6: Cross-Document Synthesis Failure

**Định nghĩa:** Mỗi retrieved passage *đúng và relevant*, nhưng LLM *không synthesize* chúng correctly để produce final answer.

**Khác với Retrieval Miss:** Evidence có trong context.  
**Khác với Unsupported Claim:** Claims có thể supported bởi individual passages, nhưng tổng hợp sai.

**Ví dụ:**
Query: "Tổng doanh thu của 3 sản phẩm A, B, C?"
- Passage 1: "Sản phẩm A doanh thu 100M"
- Passage 2: "Sản phẩm B doanh thu 150M"
- Passage 3: "Sản phẩm C doanh thu 200M"
- Expected: "450M"
- Wrong answer: "350M" (missed C) hoặc "200M" (only max) — synthesis failure

**Metrics:**
- **Multi-Document Accuracy:** Accuracy trên questions specifically requiring synthesis từ nhiều passages (e.g., aggregation, comparison, enumeration)

**Trace requirements:**
```
{
  "query_id": "q012",
  "question_type": "aggregation",  # multi-hop, comparison, enumeration
  "required_passages": ["d1", "d2", "d3"],
  "all_passages_retrieved": true,
  "all_passages_cited": true,
  "synthesis_correct": false,
  "expected_answer": "450M",
  "generated_answer": "350M"
}
```

**Debug signal:** Cross-document synthesis failure rate cao → investigate: context ordering của relevant passages, LLM instruction về aggregation/comparison, structured output format.

---

## 3. Debugging Workflow

Khi accuracy RAG degraded, follow workflow này theo thứ tự ưu tiên:

```
Step 1: Check Recall@K trên validation set
  → Thấp (< target): RETRIEVER problem
    Debug: embedding model quality, BM25 coverage, 
           chunk size, hybrid retrieval, query expansion

Step 2: Check Citation Recall
  (assumption: Recall@K đã OK)
  → Thấp: CONTEXT UTILIZATION problem
    Debug: context ordering (best-first), context length,
           number of passages (reduce K?), lost-in-middle

Step 3: Check Faithfulness Score
  (assumption: Recall@K và Citation Recall đã OK)
  → Thấp: GENERATION problem
    Debug: LLM selection, temperature, system prompt,
           context length (shorter = more focused), 
           instruction following quality

Step 4: Check Abstention Rate trên OOC set
  → False confidence rate cao: GROUNDING problem
    Debug: system prompt về uncertainty, 
           retrieval score threshold,
           "no answer found" instruction

Step 5: Check Noise Sensitivity
  → Cao: PRECISION problem
    Debug: reranker quality, K value, 
           chunk relevance filtering

Step 6: Check Cross-Document Accuracy trên multi-hop set
  → Thấp: SYNTHESIS problem
    Debug: explicit synthesis instruction,
           structured output format,
           multi-hop retrieval strategy (IRCoT, HippoRAG)
```

---

## 4. Implementation: Logging Schema

### Minimum Viable Logging

Mọi RAG request nên log:

```python
{
  "request_id": str,
  "timestamp": str,
  "query": str,
  
  # Retrieval
  "retrieved_doc_ids": List[str],
  "retrieved_scores": List[float],
  
  # Generation
  "generated_answer": str,
  "cited_doc_ids": List[str],  # nếu có citation mechanism
  
  # Metadata
  "latency_ms": int,
  "llm_model": str,
  "retriever_model": str
}
```

### Extended Logging (cho failure analysis)

```python
{
  # ... minimum fields ...
  
  # Offline annotation (batch job)
  "relevant_doc_ids": List[str],          # từ human annotation hoặc auto-judge
  "recall_at_k": float,                   # computed offline
  "citation_recall": float,               # computed offline
  "faithfulness_score": float,            # computed với LLM judge (RAGAS/ARES)
  "answer_correctness": float,            # if ground truth available
  
  # Failure flags (computed batch)
  "retrieval_miss": bool,                 # recall_at_k == 0
  "citation_miss": bool,                  # citation_recall < threshold
  "unsupported_claim_detected": bool,     # faithfulness < threshold
  "noise_propagation_suspected": bool,    # answer changes với subset context
}
```

---

## 5. Phân tích Phê bình

### Overlapping và ambiguous failure classes

Failure classes không luôn mutually exclusive:
- Unsupported claim có thể xảy ra đồng thời với citation miss
- Noise propagation và cross-document synthesis failure có overlap khi noise passage là from same domain

**Mitigation:** Prioritize debugging theo failure chain (retrieval → citation → generation), không cố classify mọi failure vào một class duy nhất.

### Cost của logging và annotation

Extended logging schema và offline annotation (đặc biệt faithfulness scoring) có real cost:
- Storage: mỗi request thêm ~2-5KB logs
- Compute: offline faithfulness scoring với LLM judge = LLM API costs
- Privacy: query logs có thể chứa PII — cần data handling policy

**Mitigation:** Sample-based evaluation (không cần annotate 100% requests), anonymization, retention policy.

### Ground truth availability

Failure detection chính xác đòi hỏi:
- Recall@K: cần biết relevant documents (manual annotation hoặc weak supervision)
- Faithfulness: cần LLM judge hoặc human annotation
- OOC detection: cần labeled OOC test set

Nhiều teams không có những labels này sẵn có. **Practical approach:** Start với proxy metrics (retrieval score distribution, citation rate, answer length distribution) trước khi đầu tư vào full annotation pipeline.

---

## 6. Liên kết với Các Kỹ Thuật Trong Series

| Failure Class | Technique giải quyết |
|--------------|---------------------|
| Retrieval Miss | [bm25_hybrid_rerank.md](./bm25_hybrid_rerank.md) (hybrid recall), [dpr_2020.md](./dpr_2020.md) (dense retrieval), [rag_fusion_2024_v2.md](./rag_fusion_2024_v2.md) (multi-query) |
| Citation Miss | [lost_in_middle_context_ordering.md](./lost_in_middle_context_ordering.md) (position-aware ordering) |
| Unsupported Claim | [ragas_eval.md](./ragas_eval.md) (faithfulness metric), [ares_eval.md](./ares_eval.md) (judge models), [self_rag_2023.md](./self_rag_2023.md) ([IsSUP] token) |
| Abstention Failure | [self_rag_2023.md](./self_rag_2023.md) ([IsUSE] token), [crag_2024.md](./crag_2024.md) (evaluator confidence) |
| Retrieval Noise | [crag_2024.md](./crag_2024.md) (decompose-then-recompose), [adaptive_rag_2024.md](./adaptive_rag_2024.md) (complexity routing) |
| Synthesis Failure | [ircot_2022.md](./ircot_2022.md) (multi-hop), [hipporag_2024.md](./hipporag_2024.md) (PPR), [raptor_2024.md](./raptor_2024.md) (hierarchy) |

---

## 7. Takeaway

**Failure taxonomy là prerequisite cho systematic improvement:** Không có taxonomy, optimization là trial-and-error. Với taxonomy, engineering team có thể: (1) measure baseline mỗi failure class, (2) implement một fix, (3) re-measure, (4) xác nhận đúng failure class đã improve. Scientific approach thay vì intuition-driven.

**Different failure classes cần different fixes:** Không có single solution cho "RAG accuracy thấp". Retrieval miss → invest in retriever/embedding. Citation miss → fix context ordering. Unsupported claim → fix generation instructions hay LLM selection. Điều này có nghĩa là failure diagnosis phải đến trước solution selection.

**Logging investment trả về theo thời gian:** Chi phí upfront của implementing full logging schema và offline evaluation pipeline đáng kể. Nhưng khi production RAG system evolve — new corpus, new LLM, new features — having traceable failure metrics là what allows confident changes với quicker debugging cycles.

---

## Nguồn

- Liu, N. F., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. *TACL 2024*. [lost_in_middle_context_ordering.md](./lost_in_middle_context_ordering.md)
- Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024*. [ragas_eval.md](./ragas_eval.md)
- Saad-Falcon, J., et al. (2023). ARES: An Automated Evaluation Framework for RAG Systems. *NAACL 2024*. [ares_eval.md](./ares_eval.md)
- Asai, A., et al. (2023). Self-RAG. *ICLR 2024*. [self_rag_2023.md](./self_rag_2023.md)
- Yan, S.-Q., et al. (2024). Corrective RAG (CRAG). *ICLR 2024*. [crag_2024.md](./crag_2024.md)
- Trivedi, H., et al. (2022). IRCoT. *ACL 2023*. [ircot_2022.md](./ircot_2022.md)
