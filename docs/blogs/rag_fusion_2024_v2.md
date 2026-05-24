# `rag_fusion_2024`: Query Diversification và Rank Fusion cho Retrieval Robust hơn

> **Nguồn gốc**: Zackary Rackauckas (2023), blog post trên Medium/Towards Data Science; kết hợp với RRF (Cormack, Clarke & Buettcher, SIGIR 2009)
> **Lưu ý quan trọng**: RAG-Fusion là một blog post và open-source implementation — **không phải peer-reviewed paper**. Đây là điểm cần ghi nhớ khi đánh giá độ tin cậy của các claim.
> **Loại**: `paper_inspired`

RAG-Fusion giải quyết sự giòn vỡ (brittleness) của single-query retrieval bằng cách để LLM tạo ra nhiều reformulation của câu hỏi gốc, sau đó hợp nhất các ranked list riêng biệt bằng Reciprocal Rank Fusion (RRF). Kỹ thuật này ghép nối hai ý tưởng có nguồn gốc khác nhau: query diversification từ truyền thống IR, và RRF từ Cormack et al. (SIGIR 2009) — một đóng góp có peer-reviewed backing thực sự.

---

## 1. Bối cảnh và Động lực

### Single-query retrieval và vấn đề brittleness

Trong hệ thống RAG tiêu chuẩn, toàn bộ chất lượng của context phụ thuộc vào một embedding vector duy nhất của query gốc. Điều này tạo ra một điểm thất bại cụ thể: nếu người dùng dùng từ ngữ khác với cách corpus được viết, retriever có thể trả về ít hoặc không có document phù hợp — dù thông tin thực sự tồn tại trong corpus. Đây là vấn đề vocabulary mismatch, đã được nghiên cứu kỹ trong Information Retrieval từ thời BM25.

Ví dụ: query "cách giảm overfitting trong neural net" và "regularization techniques for deep learning" có nghĩa gần giống nhau nhưng có thể không match tốt với nhau nếu dùng dense retrieval thuần túy trên corpus được viết theo phong cách thuật ngữ học thuật.

### Prior work: Query expansion và HyDE

**Query expansion truyền thống** (Rocchio 1971, PRF — Pseudo-Relevance Feedback) mở rộng query bằng cách thêm terms từ top-k documents đầu tiên. Cách này không cần LLM nhưng bị phụ thuộc vào chất lượng của lần retrieve đầu.

**HyDE** (Gao et al., 2022 — *Precise Zero-Shot Dense Retrieval without Relevance Labels*, arXiv:2212.10496) dùng LLM để sinh ra một *hypothetical document* cho câu hỏi, rồi embed document đó thay vì embed query. Ý tưởng là document hypothetical sẽ nằm gần hơn với document thật trong embedding space so với query ngắn. HyDE giải quyết vocabulary mismatch theo chiều "kéo query về phía document".

**RAG-Fusion** tiếp cận từ góc khác: thay vì biến đổi một query, nó tạo ra **nhiều query đa dạng** và khai thác redundancy qua RRF fusion.

### Điều RAG-Fusion thêm vào

RAG-Fusion kết hợp ba thành phần:
1. LLM làm query reformulator (thay vì rule-based expansion).
2. Multiple independent retrievals (thay vì một lần retrieve).
3. RRF làm score-free rank combination (thay vì weighted average).

---

## 2. Đóng góp Chính

RAG-Fusion có hai contribution thực dụng đáng chú ý:

**Thứ nhất — LLM-based query diversification**: Dùng LLM để sinh ra N reformulation của query gốc, thay vì dựa vào query expansion dựa trên term co-occurrence. LLM có thể paraphrase, đặt lại góc nhìn, hoặc đặt câu hỏi phụ liên quan — tạo ra diversity thực sự ở mức semantic.

**Thứ hai — RRF cho score-free rank combination**: Đây thực chất là đóng góp của Cormack et al. (SIGIR 2009), không phải của RAG-Fusion. Tuy nhiên, việc áp dụng RRF vào bài toán multi-query RAG là đóng góp thực dụng của Rackauckas. RRF không yêu cầu các ranked list phải dùng cùng score scale — điều này rất quan trọng khi fuse kết quả từ BM25 và dense retriever.

> **Cần ghi nhớ**: Bản thân Rackauckas không claim novelty về thuật toán. Blog post mang tính engineering/recipe nhiều hơn là đề xuất thuật toán mới.

---

## 3. Phương pháp Chi tiết

### 3.1 Query Diversification

LLM nhận query gốc và được prompt để sinh ra N câu hỏi liên quan nhưng có góc nhìn khác nhau. Một prompt mẫu điển hình:

```
You are a helpful assistant. Generate {N} different versions of the following
question to retrieve relevant documents from a vector database. Provide these
alternative questions separated by newlines.

Original question: {query}
```

Mục tiêu là đạt được **semantic coverage** rộng: mỗi reformulation phải cover một khía cạnh khác nhau của query space. Nếu tất cả N queries đều paraphrase gần như giống nhau, lợi ích của fusion sẽ gần như bằng không.

### 3.2 Independent Retrieval

Với mỗi query (query gốc + N-1 variants), chạy một retrieval riêng biệt và lấy top-K documents. Kết quả là N ranked list riêng biệt, mỗi list có thể có overlap một phần với các list khác.

### 3.3 Reciprocal Rank Fusion (RRF)

RRF được đề xuất bởi Cormack, Clarke & Buettcher tại SIGIR 2009 trong paper *"Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods"*. Công thức:

$$\text{score}(d) = \sum_{i=1}^{N} \frac{1}{k + \text{rank}_i(d)}$$

Trong đó:
- $d$ là một document/chunk
- $N$ là số ranked list (bằng số queries)
- $\text{rank}_i(d)$ là vị trí của $d$ trong ranked list thứ $i$ (bắt đầu từ 1); nếu $d$ không xuất hiện trong list $i$, bỏ qua hoặc gán rank rất cao
- $k = 60$ là hằng số được Cormack et al. xác định qua thực nghiệm trên TREC collections

**Tại sao $k = 60$?** Cormack et al. thử nhiều giá trị và nhận thấy $k = 60$ ổn định nhất trên nhiều collection khác nhau. Hằng số này đóng vai trò **damping**: nó làm giảm penalty cực đoan cho documents xếp cuối trong một list, tránh việc một document bị bỏ qua hoàn toàn chỉ vì xếp hạng thấp ở một trong N lists.

**Tại sao RRF tốt hơn weighted average of scores?**
- Không cần score normalization: BM25 score và cosine similarity có scale khác nhau hoàn toàn.
- Robust với outliers: một retriever có confidence rất cao không "overpower" các retriever khác.
- Đơn giản để implement và không có hyperparameter cần tune (ngoài $k$, vốn gần như luôn dùng $60$).

### 3.4 Pseudocode Pipeline

```
function rag_fusion(original_query, N, k=60, top_K=10):
    # Bước 1: Query diversification
    queries = [original_query]
    queries += llm_generate_variants(original_query, count=N-1)

    # Bước 2: Independent retrievals
    ranked_lists = []
    for q in queries:
        results = retriever.search(q, top_k=top_K)
        ranked_lists.append(results)  # list of (doc_id, score) ordered by rank

    # Bước 3: RRF fusion
    rrf_scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Bước 4: Sort by fused score
    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Bước 5: Assemble context
    context = fetch_documents([doc_id for doc_id, _ in fused_results[:top_K]])
    return generator.generate(original_query, context)
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 RRF gốc: Cormack et al. SIGIR 2009

Paper gốc của Cormack et al. đánh giá RRF trên TREC collections (TREC Robust 2004, TREC Web Track, v.v.) và so sánh với:
- Condorcet-fuse
- CombMNZ
- Individual rank learning methods

Kết quả chính: RRF đạt MAP (Mean Average Precision) cao hơn hoặc ngang bằng Condorcet và CombMNZ trên hầu hết collections, trong khi đơn giản hơn đáng kể. Đặc biệt, RRF với $k=60$ thể hiện sự ổn định tốt across collections — không cần tuning per-collection.

Đây là bằng chứng peer-reviewed duy nhất trong toàn bộ stack của RAG-Fusion về hiệu quả của phương pháp fusion.

### 4.2 RAG-Fusion: Thiếu hụt đánh giá học thuật nghiêm ngặt

**Không có peer-reviewed benchmark nào đánh giá RAG-Fusion một cách cô lập.** Blog post của Rackauckas không bao gồm controlled experiments với statistical significance testing. Các con số được trích dẫn trong cộng đồng thường đến từ:
- Anecdotal reports của practitioners.
- Leaderboard không chính thức trên datasets nhỏ.
- Comparisons không có ablation control.

Một số papers sau này (2024-2025) đề cập đến query diversification trong RAG evaluation nhưng thường không isolate RAG-Fusion cụ thể khỏi các kỹ thuật khác.

### 4.3 Bằng chứng thực tế từ cộng đồng

Practitioners báo cáo lợi ích của RAG-Fusion chủ yếu trong các trường hợp:
- Corpus đa dạng về vocabulary (technical + lay language cùng tồn tại).
- Câu hỏi có nhiều sub-aspects cần cover.
- Khi dùng kết hợp BM25 và dense retrieval (hybrid fusion qua RRF).

Không có evidence nhất quán về việc RAG-Fusion outperform trong mọi trường hợp.

### 4.4 Ablation: Số lượng queries

Dựa trên lý thuyết và các observation thực tế:

| Số queries | Nhận xét |
|---|---|
| 1 (baseline) | Standard RAG, không có fusion |
| 2 | Recall cải thiện nhỏ, overhead thấp |
| 4 | Sweet spot thực tế — diversity đủ, latency chấp nhận được |
| 8+ | Diminishing returns; LLM có xu hướng generate queries quá giống nhau |

Điểm inflection thường nằm ở 3–5 queries, sau đó marginal gain giảm mạnh trong khi cost tăng tuyến tính.

---

## 5. Phân tích Phê bình

### 5.1 Vấn đề về academic rigor

Đây là giới hạn quan trọng nhất: **RAG-Fusion thiếu peer-reviewed evaluation**. Điều này có nghĩa là:
- Không có standardized benchmark để so sánh công bằng.
- Claim về improvement không được kiểm chứng bởi independent researchers.
- Không rõ trong điều kiện nào RAG-Fusion thực sự tốt hơn single-query RAG.

Practitioners nên thực hiện ablation trên dataset của riêng họ thay vì tin vào các con số từ blog post.

### 5.2 Query diversity risk

LLM không đảm bảo tạo ra queries đa dạng. Nếu LLM generate ra $N$ paraphrases gần như giống nhau, các retrieved lists sẽ có overlap cao và RRF chỉ cộng điểm cho các documents đã được rank cao trong single-query retrieval — không có lợi ích về recall. Đây là một failure mode âm thầm, khó detect nếu không monitor diversity metrics.

Có thể đo bằng: average cosine similarity giữa query embeddings, hoặc Jaccard similarity giữa top-K result sets.

### 5.3 Latency và chi phí

RAG-Fusion tốn:
- 1 LLM call để generate N queries (có thể batch).
- N retrieval calls (có thể parallel nếu infrastructure cho phép).
- 1 RRF computation (rất rẻ — O(N × K)).

Trong production, latency tăng tuyến tính với N nếu retrievals không parallel. Với N=4 và p99 latency 200ms/retrieval, overhead là khoảng 800ms — đáng kể với real-time applications.

### 5.4 RRF chỉ hữu ích khi lists khác nhau

RRF mang lại giá trị khi các ranked lists có **significant disagreement**. Nếu mọi query đều trả về gần như cùng documents theo cùng thứ tự, RRF score chỉ là reweighting không thay đổi gì. Điều này có nghĩa là hiệu quả của RAG-Fusion phụ thuộc nhiều vào chất lượng của query diversification.

### 5.5 So sánh với HyDE

| Tiêu chí | HyDE | RAG-Fusion |
|---|---|---|
| Cơ chế chính | Hypothetical document embedding | Multiple query reformulations |
| Số LLM calls | 1 (generate hypothetical doc) | 1 (generate N queries) |
| Số retrieval calls | 1 | N |
| Xử lý vocabulary mismatch | "Kéo" embedding về phía document | Diversify query space |
| Peer-reviewed? | Có (arXiv:2212.10496, NAACL 2023) | Không |
| Failure mode chính | Hallucination trong hypothetical doc | Queries không đủ diverse |

---

## 6. Vị trí trong Landscape

Bảng so sánh các kỹ thuật xử lý retrieval failure:

| Kỹ thuật | Giả định về nguyên nhân thất bại | Cơ chế | Peer-reviewed? |
|---|---|---|---|
| Standard RAG | — (baseline) | Single query, single retrieval | — |
| BM25 + Dense Hybrid | Mỗi retriever capture khía cạnh khác nhau | Score/rank fusion | Có (nhiều papers) |
| HyDE | Query quá ngắn, không đủ vocabulary | LLM hypothetical document | Có (NAACL 2023) |
| RAG-Fusion | Một formulation bỏ lỡ relevant docs | Multi-query + RRF | Không (blog post) |
| Rewrite-Retrieve-Read | Query gốc suboptimal | LLM rewrite single query | Có (Ma et al. 2023) |
| Step-Back Prompting | Query quá specific, cần abstraction | LLM abstraction query | Có (Zheng et al. 2023) |

Điểm chung của HyDE, RAG-Fusion và Rewrite-Retrieve-Read: tất cả đều dùng LLM để biến đổi query trước khi retrieve, nhưng khác nhau ở chiến lược — một query tốt hơn, một document giả thuyết, hay nhiều queries đa dạng.

---

## 7. Takeaway

1. **RRF là phần học thuật vững chắc nhất của stack**: Cormack et al. (SIGIR 2009) cung cấp peer-reviewed evidence rằng rank fusion với công thức $\frac{1}{k+\text{rank}}$ ($k=60$) outperform các fusion methods phức tạp hơn. Đây là lý do nên tin dùng RRF. Phần còn lại của RAG-Fusion — việc dùng LLM để generate queries — là engineering decision chưa được validate nghiêm ngặt.

2. **Hiệu quả thực tế phụ thuộc hoàn toàn vào diversity of queries**: Nếu LLM không generate được queries đủ khác nhau, toàn bộ overhead (N retrievals, latency, cost) không mang lại lợi ích. Cần monitor diversity metrics trong production. Sweet spot thực nghiệm là N=3–4.

3. **Cần tự benchmark trên dataset của mình**: Không có con số benchmark đáng tin từ peer-reviewed sources cho RAG-Fusion cụ thể. Trước khi deploy, nên chạy ablation: single-query vs 2-query vs 4-query fusion, đo Recall@5/10 và MRR, và kiểm tra xem fusion có thực sự improve hay chỉ add overhead.

---

## Nguồn

- **Cormack, G. V., Clarke, C. L. A., & Buettcher, S.** (2009). Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. *Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2009)*, pp. 758–759. ACM.

- **Rackauckas, Z.** (2023). RAG-Fusion: a New Take on Retrieval-Augmented Generation. Blog post (Medium / Towards Data Science). Implementation: [github.com/Raudaschl/rag-fusion](https://github.com/Raudaschl/rag-fusion). *(Không phải peer-reviewed paper.)*

- **Gao, L., Ma, X., Lin, J., & Callan, J.** (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. arXiv:2212.10496. Published at NAACL 2023. *(HyDE — để so sánh.)*

- **Ma, X., et al.** (2023). Query Rewriting for Retrieval-Augmented Large Language Models. arXiv:2305.14283. *(Rewrite-Retrieve-Read — để so sánh.)*

- **Zheng, H., et al.** (2023). Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models. arXiv:2310.06117. *(Step-Back Prompting — để so sánh.)*
