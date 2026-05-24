# Roadmap Nghiên Cứu Và Triển Khai RAG

Tài liệu này liệt kê các paper và hướng nghiên cứu đáng để học, đánh giá và triển khai trong `rag-pipeline-lab`.
Mục tiêu không phải là sưu tầm paper, mà là biến các ý tưởng RAG quan trọng thành những technique có thể chạy được,
cấu hình được, benchmark được và so sánh được trên cùng một protocol ingest, retrieval, generation, verification và
evaluation.

Bộ blog chi tiết cho từng paper/kỹ thuật nằm tại [`docs/blogs/`](./blogs/README.md).

## Nguyên Tắc Roadmap

- Ưu tiên các kỹ thuật tác động rõ vào một stage của RAG: indexing, retrieval, reranking, context construction,
  generation, verification hoặc evaluation.
- Luôn có baseline mạnh trước khi thêm biến thể phức tạp hơn.
- Mỗi technique phải ghi rõ mức độ triển khai: `faithful_reproduction`, `paper_inspired`, `production_pattern` hoặc
  `concept_only`.
- Mỗi technique phải chạy qua cùng benchmark harness và sinh report có thể so sánh.
- Tránh thêm paper chỉ khác nhau ở prompt wording nếu không tạo ra một chiến lược reusable.

## Hiện Trạng Repo

Các hướng đã có trong repo:

| Nhóm | Technique hiện có | Trạng thái |
| --- | --- | --- |
| Baseline RAG đơn giản | `naive_rag` | local baseline chạy được |
| Parent-child retrieval | `parent_child` | production pattern chạy được |
| RAG-Sequence | `rag_sequence_2020` | paper-inspired |
| HyDE | `hyde_2022` | paper-inspired |
| RAG-Fusion | `rag_fusion_2024` | paper-inspired |
| Self-RAG critique | `self_rag_2023` | concept-only verifier |

Các khoảng trống chính:

- Chưa có production baseline mạnh kiểu lexical + dense + reranker.
- Chưa có họ retriever late-interaction như ColBERT.
- Chưa có adaptive/corrective retrieval controller.
- Chưa có long-document hierarchy ngoài parent-child chunking.
- Chưa có graph-based retrieval.
- Chưa có multi-hop hoặc agentic retrieval loop.
- Evaluation đã tốt hơn, nhưng chưa align đầy đủ với các RAG evaluation framework phổ biến.

## Các Wave Ưu Tiên

### Wave 0: Baseline Làm Nền Cho Mọi Kết Quả Sau Này

Đây không hẳn đều là paper mới, nhưng là phần bắt buộc nếu muốn các so sánh sau này đáng tin.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P0 | `bm25_hybrid_rerank` | BM25 + dense + cross-encoder production pattern | Baseline thực tế mạnh; tránh so sánh technique mới với baseline quá yếu | BM25, dense retrieval, score fusion, optional reranker |
| P0 | `dpr_2020` | [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) | Nền tảng của dense retrieval cho open-domain QA | Dense retriever paper-inspired dùng embedding hiện đại |
| P0 | `fid_2020` | [Fusion-in-Decoder](https://arxiv.org/abs/2007.01282) | Cách tiếp cận kinh điển: retrieve nhiều passage rồi generate với nhiều evidence | Context builder/generator giữ tách biệt từng passage |
| P0 | `lost_in_middle_context_ordering` | [Lost in the Middle](https://arxiv.org/abs/2307.03172) | Bài học thực tế về thứ tự context trong prompt dài | Thử nghiệm context ordering: best-first, edge-biased, interleaved |

Tiêu chí hoàn thành:

- Tất cả baseline chạy được không cần script riêng.
- Benchmark table có local baseline, hybrid baseline và một dense baseline.
- Mỗi report có recall, MRR, citation accuracy, latency và cost.

### Wave 1: Nâng Chất Lượng Retrieval Và Query Transformation

Wave này cải thiện recall và độ robust của retrieval trước khi thêm reasoning phức tạp.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P1 | `hyde_2022_v2` | [HyDE](https://arxiv.org/abs/2212.10496) | Đã có bản đầu, nhưng cần dễ tune và ablation hơn | Thêm ablation cho sample count, temperature, query embedding vs hypothetical embedding |
| P1 | `rag_fusion_2024_v2` | [RAG-Fusion](https://arxiv.org/abs/2402.03367) | Đã có bản đầu, nhưng nên hỗ trợ lexical/dense/hybrid sub-retrievers | Tổng quát hóa RRF trên nhiều retriever backend |
| P1 | `rewrite_retrieve_read_2023` | [Rewrite-Retrieve-Read](https://arxiv.org/abs/2305.14283) | Query rewriting đơn giản, thực dụng, dễ dùng trong production | Thêm query rewriter stage trước retriever |
| P1 | `replug_2023` | [REPLUG](https://arxiv.org/abs/2301.12652) | Retrieval-augmented black-box LM strategy | Passage scoring và weighted context construction |
| P1 | `colbert_2020` | [ColBERT](https://arxiv.org/abs/2004.12832) | Kiến trúc retrieval lớn với late interaction | Optional ColBERT-style retriever hoặc integration wrapper |
| P1 | `colbertv2_2022` | [ColBERTv2](https://arxiv.org/abs/2112.01488) | Late-interaction retrieval hiệu quả hơn | Chỉ làm sau khi basic ColBERT path thật sự hữu ích |

Tiêu chí hoàn thành:

- Query transform techniques ghi lại generated queries/hypotheticals trong `retrieval_runtime.debug_payload`.
- Tất cả retrieval method chạy được ở mode `retrieval_only`.
- RRF/fusion nhận được nhiều result list và normalize rank ổn định.

### Wave 2: Adaptive Và Corrective RAG

Wave này thêm controller để quyết định khi nào retrieval đã đủ, khi nào cần sửa, và khi nào cần lặp lại.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P2 | `self_rag_2023_fuller` | [Self-RAG](https://arxiv.org/abs/2310.11511) | Repo hiện chỉ có critique verifier; bản đầy đủ hơn cần retrieve/critique decisions | Concept-only controller với retrieve-needed, support và utility judgments |
| P2 | `crag_2024` | [Corrective RAG](https://arxiv.org/abs/2401.15884) | Luồng sửa retrieval thực tế khi evidence ban đầu kém | Retrieval evaluator, fallback web/local expansion interface, correction workflow |
| P2 | `adaptive_rag_2024` | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | Route theo độ phức tạp của câu hỏi | Query complexity classifier và route configs |
| P2 | `flare_2023` | [FLARE](https://arxiv.org/abs/2305.06983) | Active retrieval trong lúc generation | Generator loop kích hoạt retrieval cho span thiếu tự tin hoặc forward-looking |

Tiêu chí hoàn thành:

- Chỉ thêm layer `controller` hoặc `orchestrator` khi có ít nhất hai technique cần dùng chung.
- Evaluation report phải tách được retrieval failure và generation failure.
- Adaptive methods phải ghi lại route decisions cho từng question.

### Wave 3: Long-Document, Hierarchical Và Graph Retrieval

Wave này nhắm tới tài liệu dài, corpus lớn hoặc câu hỏi mà flat top-k retrieval không đủ.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P3 | `raptor_2024` | [RAPTOR](https://arxiv.org/abs/2401.18059) | Ý tưởng hierarchy mạnh cho long-document QA | Tree index với recursive summaries và multi-level retrieval |
| P3 | `graphrag_2024` | [GraphRAG](https://arxiv.org/abs/2404.16130) | Quan trọng cho global questions trên corpus lớn | Entity/community extraction, local/global query modes |
| P3 | `lightrag_2024` | [LightRAG](https://arxiv.org/abs/2410.05779) | Graph-style RAG đơn giản hơn, overhead thấp hơn | Lightweight graph index và dual-level retrieval |
| P3 | `hipporag_2024` | [HippoRAG](https://arxiv.org/abs/2405.14831) | Retrieval kiểu memory cho multi-hop association | Graph/memory retrieval experiment, nên bắt đầu ở mức concept-only |

Tiêu chí hoàn thành:

- Chỉ thêm graph/hierarchy artifact sau khi manifest versioning hỗ trợ rõ.
- Graph/hierarchical indices phải inspect được qua CLI.
- Report phải có index build stats: nodes, edges/summaries, build latency và estimated cost.

### Wave 4: Multi-Hop Và Agentic Retrieval

Wave này chỉ nên làm sau khi repo đã có retrieval và evaluation foundation đáng tin.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P4 | `ircot_2022` | [IRCoT](https://arxiv.org/abs/2212.10509) | Xen kẽ reasoning và retrieval cho multi-hop QA | Reasoning step -> retrieval -> evidence accumulation loop |
| P4 | `react_rag` | [ReAct](https://arxiv.org/abs/2210.03629) | Pattern reasoning/action phổ biến cho agentic workflow | Tool-using RAG agent với trace schema chặt chẽ |
| P4 | `least_to_most_rag` | [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) | Pattern decomposition hữu ích cho câu hỏi phức tạp | Query decomposition và sub-answer synthesis |
| P4 | `plan_and_solve_rag` | [Plan-and-Solve Prompting](https://arxiv.org/abs/2305.04091) | Planning trước retrieval, dễ ứng dụng trong pipeline | Planner stage và retrieval cho từng plan item |

Tiêu chí hoàn thành:

- Multi-hop methods phải emit structured trace: steps, subqueries, retrieved evidence, intermediate answers.
- Evaluation phải có multi-hop dataset hoặc synthetic multi-section QA.
- Agent loop phải có max-step guard và cost guard.

### Wave 5: Evaluation Research

Wave này giúp các claim benchmark thuyết phục và defensible hơn.

| Ưu tiên | Technique ID | Nguồn | Vì sao quan trọng | Mục tiêu triển khai |
| --- | --- | --- | --- | --- |
| P1 | `ragas_eval` | [RAGAS](https://arxiv.org/abs/2309.15217) | Bộ chiều đo RAG evaluation được tham chiếu rộng rãi | Optional metric adapter: faithfulness, answer relevance, context precision/recall |
| P2 | `ares_eval` | [ARES](https://arxiv.org/abs/2311.09476) | Framework tự động đánh giá RAG systems | Judge prompt templates và synthetic validation sets |
| P2 | `traceable_failure_taxonomy` | Internal production pattern | Cần cho report có chất lượng engineering | Phân loại lỗi: retrieval miss, citation miss, unsupported answer, abstention miss |

Tiêu chí hoàn thành:

- Giữ internal metrics ổn định kể cả khi external libraries là optional.
- Judge metrics phải có prompt/version metadata.
- Report nên hỗ trợ so sánh hai run và liệt kê regressions.

## Thứ Tự Triển Khai Đề Xuất

1. `bm25_hybrid_rerank`
2. `dpr_2020`
3. `lost_in_middle_context_ordering`
4. `rewrite_retrieve_read_2023`
5. `rag_fusion_2024_v2`
6. `ragas_eval`
7. `crag_2024`
8. `adaptive_rag_2024`
9. `raptor_2024`
10. `graphrag_2024`
11. `ircot_2022`
12. `react_rag`

Thứ tự này giữ repo đi theo hướng đáng tin: baseline mạnh trước, sau đó cải thiện recall, tiếp theo là evaluation,
rồi mới đến adaptive, hierarchical và agentic methods.

## Tên Thư Mục Technique Đề Xuất

```text
techniques/
  bm25_hybrid_rerank/
  dpr_2020/
  fid_2020/
  lost_in_middle_context_ordering/
  rewrite_retrieve_read_2023/
  replug_2023/
  colbert_2020/
  colbertv2_2022/
  crag_2024/
  adaptive_rag_2024/
  flare_2023/
  raptor_2024/
  graphrag_2024/
  lightrag_2024/
  hipporag_2024/
  ircot_2022/
  react_rag/
```

## Template Ghi Chú Nghiên Cứu

Mỗi README của technique nên có:

```markdown
# Technique Name

## Source
- Paper:
- Authors:
- Year:
- URL:

## Core Idea

## Stage Changed

## What This Repo Implements

## What This Repo Does Not Reproduce

## Expected Strengths

## Expected Failure Modes

## Config

## Benchmark Results

## Implementation Notes
```

Giữ tên heading trong template bằng tiếng Anh cũng được, vì đây là metadata kỹ thuật dễ grep và nhất quán với các
README technique hiện tại. Phần nội dung bên dưới có thể viết tiếng Việt hoặc song ngữ.

## Danh Sách Nguồn

Foundational và retrieval:

- Lewis et al., 2020, Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks: <https://arxiv.org/abs/2005.11401>
- Karpukhin et al., 2020, Dense Passage Retrieval: <https://arxiv.org/abs/2004.04906>
- Izacard and Grave, 2020, Fusion-in-Decoder: <https://arxiv.org/abs/2007.01282>
- Guu et al., 2020, REALM: <https://arxiv.org/abs/2002.08909>
- Gao et al., 2022, HyDE: <https://arxiv.org/abs/2212.10496>
- Khattab and Zaharia, 2020, ColBERT: <https://arxiv.org/abs/2004.12832>
- Santhanam et al., 2022, ColBERTv2: <https://arxiv.org/abs/2112.01488>
- Shi et al., 2023, REPLUG: <https://arxiv.org/abs/2301.12652>
- Ma et al., 2023, Rewrite-Retrieve-Read: <https://arxiv.org/abs/2305.14283>
- Rackauckas, 2024, RAG-Fusion: <https://arxiv.org/abs/2402.03367>

Adaptive, corrective và active retrieval:

- Asai et al., 2023, Self-RAG: <https://arxiv.org/abs/2310.11511>
- Yan et al., 2024, Corrective Retrieval Augmented Generation: <https://arxiv.org/abs/2401.15884>
- Jeong et al., 2024, Adaptive-RAG: <https://arxiv.org/abs/2403.14403>
- Jiang et al., 2023, FLARE: <https://arxiv.org/abs/2305.06983>

Long-document, graph và memory:

- Liu et al., 2023, Lost in the Middle: <https://arxiv.org/abs/2307.03172>
- Sarthi et al., 2024, RAPTOR: <https://arxiv.org/abs/2401.18059>
- Edge et al., 2024, GraphRAG: <https://arxiv.org/abs/2404.16130>
- Guo et al., 2024, LightRAG: <https://arxiv.org/abs/2410.05779>
- Gutiérrez et al., 2024, HippoRAG: <https://arxiv.org/abs/2405.14831>

Multi-hop và agentic:

- Trivedi et al., 2022, IRCoT: <https://arxiv.org/abs/2212.10509>
- Yao et al., 2022, ReAct: <https://arxiv.org/abs/2210.03629>
- Zhou et al., 2022, Least-to-Most Prompting: <https://arxiv.org/abs/2205.10625>
- Wang et al., 2023, Plan-and-Solve Prompting: <https://arxiv.org/abs/2305.04091>

Evaluation:

- Es et al., 2023, RAGAS: <https://arxiv.org/abs/2309.15217>
- Saad-Falcon et al., 2023, ARES: <https://arxiv.org/abs/2311.09476>
