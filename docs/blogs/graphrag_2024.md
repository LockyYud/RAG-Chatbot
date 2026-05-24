# `graphrag_2024`: Graph Index và Community Summaries cho Global Query-Focused Summarization

> **Paper**: From Local to Global: A Graph RAG Approach to Query-Focused Summarization
> **Tác giả**: Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson (Microsoft Research)
> **Venue**: arXiv:2404.16130, 2024
> **Loại**: `paper_inspired`

GraphRAG xây dựng knowledge graph từ corpus bằng LLM, phân cụm thành communities qua thuật toán Leiden, và dùng community summaries kết hợp Map-Reduce để trả lời các câu hỏi mang tính tổng quan toàn bộ tài liệu — loại câu hỏi mà flat top-k chunk retrieval về nguyên lý không thể xử lý đầy đủ.

---

## 1. Bối cảnh và Động lực

### Query-Focused Summarization là gì?

Query-focused summarization (QFS) là bài toán tạo ra một bản tóm tắt tập trung vào một chủ đề hoặc câu hỏi cụ thể, thay vì tóm tắt toàn bộ tài liệu. Ví dụ: với corpus gồm hàng trăm báo cáo nội bộ, câu hỏi "Các rủi ro vận hành chính trong năm vừa qua là gì?" yêu cầu hệ thống tổng hợp thông tin từ nhiều tài liệu, không phải tìm một đoạn cụ thể.

### Standard RAG thất bại ở đâu?

RAG truyền thống dựa trên top-k semantic retrieval: embed query, tìm k chunks gần nhất trong vector store, đưa vào context window của LLM. Cách này hoạt động tốt cho factual lookup:

- "Điều kiện xét tuyển đại học là gì?" → retrieve đúng section trong tài liệu tuyển sinh.
- "API endpoint để tạo user là gì?" → retrieve đúng đoạn documentation.

Nhưng với global questions, top-k retrieval có vấn đề cấu trúc:

> *"What are the main themes in this entire document corpus?"*

Không có k chunks nào đủ đại diện cho toàn bộ corpus. Flat retrieval chỉ lấy được những fragments xung quanh các keywords trong query, bỏ qua cấu trúc tổng thể, các chủ đề xuất hiện rải rác, hay các mối quan hệ tiềm ẩn giữa entities.

### Câu hỏi nghiên cứu

> Liệu có thể xây dựng một index structure — không phải flat vector store — giúp LLM trả lời global summarization questions chính xác và toàn diện hơn?

---

## 2. Đóng góp Chính

Paper đề xuất bốn đóng góp có thể phân biệt rõ ràng:

1. **LLM-based entity và relation extraction pipeline**: Thay vì dùng NER truyền thống, tác giả dùng LLM để trích xuất entities, relationships, và covariate claims trực tiếp từ text chunks. Điều này cho phép trích xuất ngữ nghĩa phong phú hơn, kể cả các quan hệ ẩn.

2. **Hierarchical community detection bằng Leiden algorithm**: Sau khi build graph, tác giả áp dụng Leiden algorithm để phát hiện communities ở nhiều resolution level. Mỗi level cho một góc nhìn khác nhau về cấu trúc corpus.

3. **Community summarization bằng LLM**: Mỗi community được LLM tóm tắt thành một đoạn văn ngắn, trở thành "unit of knowledge" cho global queries. Đây là sự khác biệt kiến trúc cốt lõi so với chunk-based retrieval.

4. **Hai query modes riêng biệt — Local và Global**: Local mode dùng entity-centric graph traversal cho câu hỏi về entity cụ thể. Global mode dùng Map-Reduce trên community summaries cho câu hỏi tổng quan.

---

## 3. Phương pháp Chi tiết

### Phase 1 — Indexing: Entity và Relation Extraction

Input là các source documents được chunk thành text units (mặc định ~600 tokens, có overlap). Với mỗi text unit, LLM được prompt để extract:

- **Entities**: tên, loại (person, organization, location, concept...), mô tả ngắn gọn.
- **Relationships**: cặp (source entity, target entity, mô tả quan hệ, strength score).
- **Covariates** (optional): claims hoặc events gắn với entity.

Prompt design theo dạng few-shot với ví dụ cụ thể, yêu cầu LLM output dưới dạng structured tuples. Sau đó entities trùng tên được merge bằng cách summarize descriptions, và relationships được aggregated theo weight (số lần xuất hiện).

Kết quả là một **undirected weighted graph** G = (V, E) trong đó:
- V là tập entities với descriptions.
- E là tập relationships với weight và description.

### Phase 2 — Indexing: Community Detection bằng Leiden Algorithm

Sau khi có graph, bước tiếp theo là phân cụm thành communities. Tác giả chọn **Leiden algorithm** (Traag et al., 2019) thay vì các thuật toán khác như Louvain hay spectral clustering vì:

- **Tính bảo đảm về chất lượng partition**: Leiden đảm bảo các communities đều *well-connected* bên trong, tránh vấn đề "disconnected communities" mà Louvain mắc phải.
- **Hierarchical resolution**: Leiden hỗ trợ tối ưu modularity ở nhiều resolution parameter γ, tạo ra cây phân cấp communities từ granular (nhiều communities nhỏ) đến coarse (ít communities lớn). Paper dùng các levels C₀, C₁, C₂, C₃ với C₀ là fine-grained nhất.
- **Scalability**: Leiden chạy gần tuyến tính theo số edges, phù hợp với graphs lớn từ real-world corpora.

Mỗi resolution level cho một cách nhìn khác nhau:
- Level 0: communities nhỏ, chi tiết về sub-topics hẹp.
- Level 2-3: communities lớn, đại diện cho major themes của corpus.

### Community Summarization

Với mỗi community ở mỗi level, LLM được prompt để viết một summary tổng hợp từ: entity descriptions, relationship descriptions, và covariate claims của tất cả nodes trong community đó.

Community summary phải bao gồm:
- Những entities chính trong community là gì.
- Các mối quan hệ quan trọng bên trong.
- Các findings hoặc claims đáng chú ý.
- Source references (chunk IDs).

### Query Time — Local Mode

Local mode phù hợp với câu hỏi về entity cụ thể hoặc tìm kiếm trong neighborhood của graph:

```
query
  → extract query entities
  → find matching graph nodes (entity linking)
  → traverse neighborhood: entities + relations + covariates
  → rank candidates bởi relevance score
  → fill context window theo priority
  → generate answer
```

Local mode giống với entity-augmented RAG hơn là global summarization.

### Query Time — Global Mode (Map-Reduce)

Global mode là đóng góp kiến trúc chính của paper:

```
global query
  → chọn community level phù hợp (thường level 1-2)
  → shuffle community summaries thành random batches
  → MAP: với mỗi batch, LLM generate một partial answer và rate usefulness (0-100)
  → lọc partial answers có usefulness score thấp
  → REDUCE: LLM tổng hợp tất cả partial answers có điểm cao thành final answer
```

**Lý do dùng Map-Reduce**: Tổng số community summaries thường vượt quá context window của một LLM call. Map-Reduce cho phép xử lý parallel và chỉ giữ lại những partial answers thực sự hữu ích. Đây là adaptation trực tiếp từ paradigm MapReduce của distributed computing.

**Sơ đồ toàn hệ thống:**

```
Documents
    │
    ▼
Text Chunks (600 tokens, overlap)
    │
    ▼ LLM extraction (few-shot prompt)
Entity Graph G = (V, E)
    │
    ▼ Leiden algorithm (multi-resolution)
Community Tree (Level 0 → Level 3)
    │
    ▼ LLM summarization per community
Community Summaries
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
Local Query Mode                Global Query Mode
(entity traversal)              (Map-Reduce on summaries)
    │                                  │
    └──────────────┬───────────────────┘
                   ▼
             Final Answer
```

---

## 4. Thực nghiệm và Kết quả

### Setup Đánh giá

Paper dùng hai corpora thực tế:
- **Podcast transcripts**: ~1 triệu tokens từ các episode podcast (dataset công khai).
- **News articles**: ~1.7 triệu tokens từ bộ tin tức.

Thay vì dùng ground truth cố định, tác giả dùng **LLM-as-judge** với GPT-4 Turbo để so sánh cặp câu trả lời theo ba tiêu chí:

| Metric | Định nghĩa |
|--------|-----------|
| **Comprehensiveness** | Câu trả lời bao phủ đầy đủ các khía cạnh của câu hỏi không? |
| **Diversity** | Câu trả lời có đưa ra nhiều góc nhìn, quan điểm khác nhau không? |
| **Empowerment** | Câu trả lời có giúp người đọc hiểu và đưa ra quyết định không? |

Mỗi câu hỏi là global question được tạo từ corpus, ví dụ: "What are the most important topics in this podcast series?"

### Kết quả Chính — Win Rates so với Naive RAG

**Trên Podcast Corpus:**

| Condition | Comprehensiveness | Diversity | Empowerment |
|-----------|:-----------------:|:---------:|:-----------:|
| GraphRAG Global C2 vs Naive RAG | **72%** | **62%** | 57% |
| GraphRAG Global C1 vs Naive RAG | **72%** | **65%** | 57% |
| GraphRAG Local vs Naive RAG | 54% | 56% | 57% |
| Community Summaries (C2) vs Naive RAG | **73%** | **65%** | 57% |

**Trên News Corpus:**

| Condition | Comprehensiveness | Diversity | Empowerment |
|-----------|:-----------------:|:---------:|:-----------:|
| GraphRAG Global C2 vs Naive RAG | **80%** | **73%** | 66% |
| GraphRAG Global C1 vs Naive RAG | **82%** | **74%** | 66% |

Kết quả nổi bật: GraphRAG global mode vượt trội naive RAG đáng kể về comprehensiveness và diversity, nhưng lợi thế về empowerment ít rõ ràng hơn (~57-66% win rate, gần mức 50/50).

### Chi phí Token — Indexing vs Querying

Đây là điểm then chốt của paper. GraphRAG yêu cầu **indexing upfront cost** rất lớn:

| Phase | Token usage (Podcast corpus, ~1M tokens) |
|-------|------------------------------------------|
| Entity/relation extraction | ~3.3× corpus size |
| Community summarization (C0) | ~2.0× corpus size |
| **Total indexing** | **~26 tokens per source token** |

So sánh, naive RAG (standard chunked retrieval) có indexing cost gần như bằng 0 (chỉ embedding).

Ở query time, global mode tiêu thụ nhiều tokens hơn local mode và naive RAG vì phải process tất cả community summaries qua Map-Reduce. Community level coarser (ít communities hơn) giảm token usage nhưng cũng giảm detail.

### Khi GraphRAG Kém Hơn

Paper thừa nhận rõ: với câu hỏi local factual ("What year was X founded?"), **naive RAG thường tốt hơn hoặc ngang bằng GraphRAG**. GraphRAG không được thiết kế để thay thế chunk retrieval cho mọi loại query.

---

## 5. Phân tích Phê bình

### Rủi ro Hallucination trong Graph Construction

Graph được xây dựng hoàn toàn từ LLM extraction. LLM có thể:
- Tạo ra relationships không tồn tại trong source text (relation hallucination).
- Merge entities khác nhau vì tên tương tự (coreference errors).
- Bỏ qua entities ít prominent dù quan trọng về mặt ngữ nghĩa.

Sai số trong extraction sẽ propagate sang community detection và cuối cùng ảnh hưởng chất lượng community summaries. Không có bước verification tự động nào trong pipeline gốc.

### Chi phí Indexing Không Incremental

Indexing cost (~26× source tokens) là cố định và phải chạy lại nếu corpus thay đổi. Với corpus 1M tokens dùng GPT-4, chi phí indexing có thể lên tới hàng chục đến hàng trăm USD. Paper không đề xuất incremental update strategy — thêm một tài liệu mới về lý thuyết cần rebuild toàn bộ graph hoặc ít nhất merge subgraph mới vào community structure cũ.

### LLM-as-Judge Bias

Evaluation hoàn toàn dựa vào GPT-4 Turbo làm judge. Đây là methodological limitation đã được ghi nhận rộng rãi:
- LLM judges có xu hướng favor longer, more detailed answers — GraphRAG global mode tự nhiên sinh ra answers dài hơn.
- Self-preference bias: dùng GPT-4 để judge answers cũng được tạo bởi GPT-4 có thể có systematic bias.
- Không có human evaluation để calibrate.

### GraphRAG Kém cho Local Questions

Local factual queries là điểm yếu rõ ràng. Với câu hỏi "Ai là CEO của công ty X?", top-k chunk retrieval nhanh hơn, rẻ hơn và chính xác hơn entity traversal của GraphRAG. Triển khai GraphRAG cho mọi loại query là lãng phí tài nguyên và có thể giảm chất lượng.

### Reproducibility

Microsoft đã open-source code tại [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag), nhưng pipeline nặng về LLM calls và khó reproduce chính xác vì:
- Kết quả phụ thuộc vào model version và temperature.
- Chi phí thực tế để chạy full experiment trên 1M+ token corpus khá cao.
- Community structure thay đổi tùy theo Leiden random seed.

---

## 6. Vị trí trong Landscape

### So sánh với các phương pháp khác

| Phương pháp | Global Questions | Local Questions | Indexing Cost | Query Latency |
|-------------|:----------------:|:---------------:|:-------------:|:-------------:|
| **Naive RAG** | Kém | Tốt | Thấp (embed only) | Thấp |
| **HyDE** | Kém | Khá tốt | Thấp | Thấp-Trung |
| **Parent-Child** | Kém | Tốt | Trung | Thấp |
| **RAPTOR** | Tốt | Khá tốt | Cao (hierarchical summaries) | Trung |
| **GraphRAG Local** | Trung | Tốt | Rất cao | Trung |
| **GraphRAG Global** | Rất tốt | Kém | Rất cao | Cao |

### RAPTOR vs GraphRAG

Cả hai đều hướng đến hierarchical indexing cho global questions, nhưng khác biệt ở kiến trúc:
- **RAPTOR** dùng recursive clustering + summarization của text chunks, không có explicit entity graph.
- **GraphRAG** build explicit entity-relation graph, cho phép entity-centric traversal ở local mode.

RAPTOR rẻ hơn về indexing; GraphRAG mạnh hơn khi cần entity-level reasoning hoặc quan hệ giữa entities.

### Khi nào GraphRAG đáng chi phí?

GraphRAG indexing cost hợp lý khi:
1. Corpus ổn định (không update thường xuyên).
2. Người dùng chủ yếu hỏi global/thematic questions.
3. Entity relationships là trung tâm của domain (ví dụ: knowledge base về nhân vật lịch sử, mạng lưới công ty, hệ sinh thái công nghệ).
4. Corpus đủ lớn để top-k retrieval bỏ lỡ thông tin quan trọng (>100K tokens).

---

## 7. Takeaway

**1. Graph index là một loại index khác về mặt cấu trúc, không phải optimization của flat retrieval.**
Community summaries encode thông tin về *cấu trúc* của corpus, không chỉ *nội dung* của từng chunk. Đây là lý do GraphRAG có thể trả lời global questions mà RAG truyền thống không thể — không phải vì nó "retrieve tốt hơn", mà vì nó có một dạng pre-computed synthesis.

**2. Leiden algorithm giải quyết vấn đề cụ thể của Louvain: disconnected communities.**
Khi community detection fails (communities nội bộ không connected), community summaries sẽ incoherent. Việc chọn Leiden không phải ngẫu nhiên — đây là quyết định engineering có động lực rõ ràng từ đặc điểm của knowledge graphs thực tế.

**3. Chi phí indexing là trade-off chính, không phải bug.**
~26× source tokens cho indexing là cái giá của pre-computation. Giống như building an inverted index để enable fast full-text search, graph indexing là upfront investment cho global QA capability. Quyết định "có nên dùng GraphRAG không" phụ thuộc vào query distribution — nếu >50% queries là global questions, investment có thể đáng; nếu hầu hết là local factual lookup, naive RAG hiệu quả hơn nhiều.

---

## Nguồn

- Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., & Larson, J. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. *arXiv:2404.16130*. https://arxiv.org/abs/2404.16130
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports*, 9(1), 5233. https://doi.org/10.1038/s41598-019-41695-z
- Microsoft GraphRAG open-source implementation: https://github.com/microsoft/graphrag
- Edge et al. (2024) evaluation details and token cost analysis: arXiv:2404.16130, Section 4 (Evaluation) và Appendix C (Token Costs).
