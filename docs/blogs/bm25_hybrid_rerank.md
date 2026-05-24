# `bm25_hybrid_rerank`: Tại Sao Hybrid Retrieval và Reranking Là Baseline Không Thể Bỏ Qua

> **Pattern**: Production pattern tổng hợp từ nhiều nghiên cứu
> **Nguồn gốc**: BM25 (Robertson & Zaragoza, 2009), DPR (Karpukhin et al., 2020), RRF (Cormack et al., 2009)
> **Loại**: `production_pattern`

Không có retriever đơn lẻ nào đủ tốt cho mọi truy vấn: BM25 nắm bắt exact keyword một cách đáng tin cậy nhưng bỏ lỡ paraphrase, trong khi dense retrieval xử lý tốt ngữ nghĩa nhưng dễ bị mờ bởi các ký hiệu kỹ thuật hay tên định danh hiếm. Pattern `bm25_hybrid_rerank` kết hợp hai tín hiệu bổ sung nhau qua rank fusion, sau đó dùng cross-encoder reranker để cải thiện precision — tạo ra một baseline đủ mạnh mà bất kỳ kỹ thuật RAG nâng cao nào cũng phải vượt qua mới có giá trị thực.

---

## 1. Bối cảnh và Động lực

### Sparse Retrieval và Dense Retrieval: Thế Mạnh Khác Nhau

Trong lịch sử information retrieval, BM25 (Best Match 25) là sparse lexical model được Robertson & Zaragoza (2009) hệ thống hoá từ dòng nghiên cứu Probabilistic Relevance Framework trải dài hàng chục năm. BM25 không cần training, index nhanh bằng inverted index, và đặc biệt mạnh khi query chứa exact term: mã điều khoản pháp lý ("Điều 12a"), mã sản phẩm ("SKU-7842"), tên viết tắt ("ISO 27001"), hay ngày tháng cụ thể.

Dense retrieval — tiêu biểu là DPR của Karpukhin et al. (2020) — ánh xạ câu hỏi và đoạn văn vào cùng một không gian vector thông qua dual-encoder được train trên question-passage pairs. Khi người dùng hỏi "Làm thế nào để tăng tốc code?" nhưng tài liệu viết "Các kỹ thuật tối ưu hoá hiệu suất phần mềm", dense retrieval nắm bắt sự tương đồng ngữ nghĩa mà BM25 không thể làm được.

### Bằng Chứng Thực Nghiệm Về Sự Bổ Sung

Karpukhin et al. (2020) chỉ ra rằng DPR vượt BM25 đáng kể trên Natural Questions (top-20 accuracy: 79.4% so với 59.1%) và TriviaQA (top-20: 78.8% so với 66.9%). Tuy nhiên, cùng bài báo thừa nhận BM25 vẫn cạnh tranh hoặc thắng trên các tập dữ liệu có nhiều named entity như WebQuestions và CuratedTREC, nơi exact match mang lại lợi thế rõ ràng.

Quan trọng hơn, Thakur et al. (2021) trên BEIR benchmark — bao gồm 18 tập dữ liệu đa dạng từ y tế đến luật pháp — cho thấy không retriever đơn lẻ nào thống trị toàn bộ: BM25 thắng trên TREC-COVID và Signal-1M, DPR thắng trên NQ và HotpotQA, còn hybrid thường đạt điểm cao nhất trung bình. Đây là lý do cơ bản để xây dựng hybrid pipeline.

---

## 2. Đóng góp Chính (của pattern này)

Pattern `bm25_hybrid_rerank` tổng hợp ba insight quan trọng từ tài liệu học thuật:

**Recall-precision separation**: Retriever (cả BM25 lẫn dense) được tối ưu cho recall cao — lấy về nhiều candidate có liên quan. Reranker sau đó được tối ưu cho precision — chọn ra những candidate thực sự hữu ích nhất. Tách hai mục tiêu này cho phép mỗi thành phần chuyên biệt hóa.

**RRF cho score normalization**: BM25 trả về TF-IDF score dạng log probability, trong khi dense retriever trả về cosine similarity trong khoảng [−1, 1]. Hai không gian score này không thể cộng trực tiếp. Cormack et al. (2009) đề xuất Reciprocal Rank Fusion — chỉ dùng thứ hạng (rank), không dùng score gốc — giải quyết bài toán normalization một cách gọn gàng.

**Staged pipeline**: Pipeline gồm nhiều tầng (retrieval → fusion → reranking) cho phép kiểm soát latency theo từng bước và dễ dàng A/B test từng thành phần độc lập.

---

## 3. Phương pháp Chi tiết

### 3.1 BM25: Công Thức và Tham Số

BM25 score cho document `d` với query `q` gồm các term `t₁, t₂, ..., tₙ`:

```
BM25(d, q) = Σ IDF(tᵢ) · (f(tᵢ, d) · (k₁ + 1)) / (f(tᵢ, d) + k₁ · (1 - b + b · |d|/avgdl))
```

Trong đó:

- `f(tᵢ, d)`: tần suất xuất hiện của term `tᵢ` trong document `d`
- `|d|`: độ dài document (tính theo số từ)
- `avgdl`: độ dài trung bình của các document trong corpus
- `IDF(tᵢ) = log((N - n(tᵢ) + 0.5) / (n(tᵢ) + 0.5) + 1)` với `N` là tổng số document và `n(tᵢ)` là số document chứa term `tᵢ`

Ba tham số cần chú ý:

| Tham số | Vai trò | Giá trị mặc định |
| --- | --- | --- |
| `k₁` | Kiểm soát ảnh hưởng của term frequency — giá trị cao hơn ưu tiên document nhắc đến term nhiều lần | 1.2–2.0 |
| `b` | Kiểm soát length normalization — `b=1` normalize hoàn toàn, `b=0` bỏ qua độ dài | 0.75 |
| `k₃` | Term frequency saturation trong query (ít dùng) | 0–1000 |

Khi `k₁ = 0`, BM25 suy biến thành IDF-only model. Khi `b = 0`, BM25 bỏ qua chuẩn hoá độ dài document. Giá trị `k₁ = 1.5` và `b = 0.75` hoạt động tốt trên hầu hết corpus văn bản tiếng Anh và tiếng Việt.

### 3.2 Dense Retrieval: Dual-Encoder Architecture

DPR (Karpukhin et al., 2020) sử dụng hai BERT encoder riêng biệt:

```
E_Q(q) : query → ℝ^d
E_P(p) : passage → ℝ^d

sim(q, p) = E_Q(q)ᵀ · E_P(p)
```

Query encoder và passage encoder được train bằng contrastive learning: maximize dot product giữa query và positive passage, minimize với negative passages. Sau khi train, toàn bộ passage corpus được encode offline và lưu vào FAISS index. Lúc inference, chỉ cần encode query rồi tìm nearest neighbors qua approximate nearest neighbor search.

Điểm mạnh: capture ngữ nghĩa qua paraphrase. Điểm yếu: exact token có thể bị "hoà tan" trong vector space, đặc biệt với named entities ít xuất hiện trong training data.

### 3.3 Reciprocal Rank Fusion (RRF)

Cormack et al. (2009) đề xuất RRF để kết hợp nhiều danh sách kết quả ranked mà không cần normalize score gốc:

```
RRF_score(d) = Σᵢ 1 / (k + rankᵢ(d))
```

Trong đó:

- `rankᵢ(d)`: thứ hạng của document `d` trong danh sách thứ `i` (bắt đầu từ 1)
- `k`: hằng số smoothing, mặc định `k = 60`
- Nếu document `d` không xuất hiện trong danh sách thứ `i`, không cộng thêm gì

Ví dụ: document `d` đứng hạng 3 trong BM25 và hạng 7 trong dense retrieval:

```
RRF_score(d) = 1/(60 + 3) + 1/(60 + 7) = 1/63 + 1/67 ≈ 0.0159 + 0.0149 = 0.0308
```

Tại sao `k = 60`? Cormack et al. thử nghiệm nhiều giá trị và `k = 60` tạo ra một "dead zone" — document xếp hạng thấp (rank > 60) đóng góp rất ít, tránh nhiễu từ những kết quả kém chất lượng. Document xếp hạng cao được ưu tiên rõ ràng hơn.

Ưu điểm cốt lõi: RRF hoạt động tốt ngay cả khi các retriever có score scale rất khác nhau, không yêu cầu hyperparameter tuning ngoài `k`.

### 3.4 Cross-Encoder Reranking

Sau khi RRF tạo ra một danh sách hợp nhất (thường top-20 đến top-50 candidate), cross-encoder reranker chấm điểm lại từng cặp `(query, passage)` cùng nhau:

```
score(q, p) = CrossEncoder([CLS] q [SEP] p [SEP])
```

Khác với dual-encoder (encode query và passage độc lập), cross-encoder xử lý query và passage cùng lúc qua full attention — cho phép mô hình nắm bắt interaction chi tiết giữa query token và passage token. Kết quả là chính xác hơn đáng kể, nhưng không thể pre-compute passage offline như dense retrieval, nên latency tăng theo tuyến tính với số candidate.

### 3.5 Pipeline Tổng Thể

```
Query
  ├─→ BM25 retriever ──────────────────────┐
  │    (top-K₁ candidates, lexical signal) │
  │                                        ├─→ RRF Fusion (k=60) → top-N merged list
  └─→ Dense retriever ─────────────────────┘
       (top-K₂ candidates, semantic signal)
                                                │
                                                ▼
                                        Cross-encoder reranker
                                        (top-M final list, M << N)
                                                │
                                                ▼
                                        Citation-aware context builder
                                                │
                                                ▼
                                            Generator
```

Giá trị điển hình: `K₁ = K₂ = 20`, `N = 30` (sau dedup), `M = 5–10`.

---

## 4. Thực nghiệm và Kết quả

### 4.1 BM25 vs DPR: Natural Questions và TriviaQA

Karpukhin et al. (2020) báo cáo top-20 passage retrieval accuracy:

| Phương pháp | Natural Questions | TriviaQA | WebQuestions | CuratedTREC |
| --- | --- | --- | --- | --- |
| BM25 (Lucene) | 59.1% | 66.9% | 55.0% | 70.9% |
| DPR (single) | 78.4% | 79.4% | 73.2% | 79.4% |
| DPR (multi) | **79.4%** | **78.8%** | **75.0%** | **89.1%** |
| BM25 + DPR hybrid | 76.6% | 78.0% | 73.9% | 88.2% |

Kết quả đáng chú ý: hybrid không phải lúc nào cũng thắng DPR đơn lẻ trên NQ — nhưng trên WebQuestions và CuratedTREC (nhiều named entity), BM25 góp phần đáng kể. Điều này nhấn mạnh rằng lợi ích của hybrid phụ thuộc nhiều vào đặc điểm corpus và query distribution.

### 4.2 RRF vs Các Phương Pháp Fusion Khác

Cormack et al. (2009) so sánh RRF với CombMNZ, CombSUM và các rank learning methods trên TREC benchmark:

| Phương pháp | TREC MAP trung bình |
| --- | --- |
| Best single ranker | 0.347 |
| CombMNZ | 0.352 |
| Condorcet fusion | 0.361 |
| RRF (k=60) | **0.367** |
| RankBoost (supervised) | 0.356 |

RRF — không cần training, không cần score normalization — vượt cả supervised learning method trên tập test. Đây là lý do RRF trở thành lựa chọn mặc định trong hybrid retrieval pipeline thực tế.

### 4.3 Tác Động Của Cross-Encoder Reranking

Nogueira & Cho (2019) — bài báo đề xuất dùng BERT làm passage reranker — báo cáo trên MS MARCO passage ranking:

| Phương pháp | MRR@10 |
| --- | --- |
| BM25 baseline | 0.167 |
| BM25 + BERT reranker | **0.365** |
| Cải thiện tương đối | +119% |

Hofstätter et al. (2020) trên MS MARCO với TAS-B bi-encoder:

| Phương pháp | MRR@10 | Latency (ms/query) |
| --- | --- | --- |
| BM25 | 0.184 | 55 |
| Bi-encoder (TAS-B) | 0.344 | 42 |
| Hybrid + cross-encoder | **0.390** | 180 |

Cross-encoder cải thiện MRR@10 đáng kể so với bi-encoder đơn lẻ, nhưng latency tăng khoảng 4× — điều này giải thích tại sao cross-encoder chỉ áp dụng cho candidate pool nhỏ (top-20 đến top-50), không phải toàn corpus.

### 4.4 BEIR: Kết Quả Tổng Quát Hoá

Thakur et al. (2021) đo nDCG@10 trung bình trên 18 tập dữ liệu BEIR:

| Phương pháp | nDCG@10 (trung bình 18 datasets) |
| --- | --- |
| BM25 | 0.428 |
| DPR (trained on NQ) | 0.226 |
| ANCE | 0.336 |
| TAS-B | 0.397 |
| Hybrid (BM25 + dense) | **~0.44–0.46** (phụ thuộc implementation) |

Kết quả DPR thấp do overfitting vào NQ training distribution — một cảnh báo quan trọng: dense retriever train trên một domain có thể generalise kém sang domain khác. BM25, không cần training, generalise tốt hơn đáng ngạc nhiên trên zero-shot settings.

---

## 5. Phân tích Phê bình

### Khi Hybrid Không Giúp Ích

Hybrid retrieval không phải lúc nào cũng tốt hơn. Một số trường hợp hybrid không cải thiện hoặc thậm chí giảm chất lượng:

- **Corpus đồng nhất về từ vựng**: Nếu tất cả tài liệu dùng cùng thuật ngữ kỹ thuật chuẩn, BM25 đã đủ mạnh và dense retrieval thêm vào chủ yếu là noise.
- **Query luôn là exact keyword**: Trong enterprise search với mã sản phẩm, số hiệu văn bản, BM25 thường thắng dense retrieval. Hybrid có thể làm loãng signal.
- **Dense retriever không được train đúng domain**: Như BEIR cho thấy, DPR train trên NQ đạt nDCG@10 chỉ 0.226 — thấp hơn BM25. Hybrid với một dense retriever yếu có thể làm hỏng kết quả BM25.

### Độ Nhạy Của Score Fusion

Nếu dùng linear interpolation thay vì RRF — `score = α · BM25_norm + (1-α) · dense_score` — tham số `α` rất nhạy với đặc điểm corpus. Giá trị `α = 0.5` hợp lý cho NQ có thể thất bại hoàn toàn trên corpus pháp lý hay y tế. RRF tránh được vấn đề này bằng cách không dùng score gốc.

### Reranker Recall Loss

Cross-encoder chỉ thấy candidate pool đầu vào. Nếu RRF đã bỏ sót document liên quan (false negative ở tầng retrieval), reranker không thể cứu vớt. Đây là lý do candidate pool trước rerank phải đủ lớn — trade-off giữa recall của pool và latency của reranker.

### Chi Phí Vận Hành

Hybrid pipeline yêu cầu duy trì hai index song song: inverted index cho BM25 (BM25Okapi, Elasticsearch, Lucene) và vector index cho dense retrieval (FAISS, Qdrant, Weaviate). Đây là gánh nặng operational đáng kể: hai quá trình indexing, hai quá trình update khi corpus thay đổi, hai infrastructure cần monitor.

---

## 6. Vị trí trong Landscape

| Phương pháp | Điểm mạnh | Điểm yếu | Khi nào dùng |
| --- | --- | --- | --- |
| BM25-only | Nhanh, không cần training, generalise tốt | Không hiểu paraphrase, bỏ lỡ ngữ nghĩa | Corpus kỹ thuật nhiều exact keyword, zero-shot |
| Dense-only | Hiểu ngữ nghĩa, paraphrase | Yếu với named entity hiếm, cần training data | Corpus ngôn ngữ tự nhiên, có fine-tuning data |
| Hybrid (BM25 + Dense) | Bổ sung điểm yếu của nhau | Phức tạp hơn, hai index | Production baseline cho corpus đa dạng |
| Hybrid + Cross-encoder | Precision cao nhất | Latency cao, cần GPU cho reranker | Khi precision quan trọng hơn latency |
| ColBERT late-interaction | Tốc độ tốt hơn cross-encoder, chính xác hơn bi-encoder | Index lớn hơn (lưu token embeddings) | Khi cần cân bằng speed và accuracy |

ColBERT (Khattab & Zaharia, 2020) là hướng thú vị: thay vì encode cả passage thành một vector (bi-encoder) hay tính attention đầy đủ lúc inference (cross-encoder), ColBERT encode từng token độc lập và tính MaxSim giữa query tokens và passage tokens — cho phép pre-compute passage representation trong khi vẫn giữ được token-level interaction. Tuy nhiên, ColBERT đòi hỏi index lớn hơn đáng kể so với bi-encoder.

---

## 7. Takeaway

**Hybrid retrieval cải thiện recall bằng cách tổng hợp hai tín hiệu bổ sung nhau**: BM25 bắt exact lexical match đáng tin cậy, dense retrieval bắt semantic similarity. Không phương pháp nào thống trị trên mọi loại corpus — BEIR benchmark là bằng chứng thực nghiệm rõ ràng nhất.

**RRF là lựa chọn fusion mặc định tốt**: Không cần training, không cần tune score normalization, vượt nhiều supervised fusion method trong thực nghiệm của Cormack et al. (2009). Hằng số `k = 60` hoạt động tốt trên hầu hết corpus mà không cần điều chỉnh.

**Reranking cải thiện precision đáng kể nhưng có chi phí**: Cross-encoder tăng MRR@10 lên hơn 2× so với BM25 đơn lẻ (Nogueira & Cho, 2019), nhưng latency tăng 3–5×. Pattern này chỉ hợp lý khi candidate pool đủ lớn (≥20 candidates) để reranker có đủ nguyên liệu, và khi ứng dụng có thể chấp nhận thêm vài trăm milliseconds.

---

## Nguồn

- Robertson, S., & Zaragoza, H. (2009). **The Probabilistic Relevance Framework: BM25 and Beyond**. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.-T. (2020). **Dense Passage Retrieval for Open-Domain Question Answering**. *EMNLP 2020*. [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
- Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). **Reciprocal rank fusion outperforms condorcet and individual rank learning methods**. *SIGIR 2009*.
- Nogueira, R., & Cho, K. (2019). **Passage Re-ranking with BERT**. *arXiv:1901.04085*. [https://arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085)
- Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., & Gurevych, I. (2021). **BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models**. *NeurIPS 2021*. [https://arxiv.org/abs/2104.08663](https://arxiv.org/abs/2104.08663)
- Hofstätter, S., Lin, S.-C., Yang, J.-H., Lin, J., & Hanbury, A. (2021). **Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling**. *SIGIR 2021*.
- Khattab, O., & Zaharia, M. (2020). **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**. *SIGIR 2020*. [https://arxiv.org/abs/2004.12832](https://arxiv.org/abs/2004.12832)
