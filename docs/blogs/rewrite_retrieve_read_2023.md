# `rewrite_retrieve_read_2023`: Query Rewriter như Adapter Layer trong RAG

> **Paper**: Query Rewriting for Retrieval-Augmented Large Language Models
> **Tác giả**: Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, Nan Duan
> **Venue**: EMNLP 2023 — arXiv:2305.14283
> **Loại**: `paper_inspired`

Thay vì cố sửa retriever hay reader, paper này chèn một **trainable rewriter** vào giữa user query và retrieval step — một lớp adapter nhỏ được huấn luyện bằng reinforcement learning với reader performance làm reward. Rewriter học cách biến natural language query thành search query sát với ngôn ngữ corpus, mà không cần chạm vào retriever hay reader.

---

## 1. Bối cảnh và Động lực

Trong RAG, người dùng đặt câu hỏi theo ngôn ngữ tự nhiên, còn tài liệu viết theo văn phong học thuật, pháp lý, hay kỹ thuật chuyên ngành. Gap này — thường gọi là **vocabulary mismatch** — khiến cả BM25 lẫn dense retriever đều dễ bỏ sót tài liệu liên quan, ngay cả khi tài liệu đó chứa đúng thông tin cần thiết.

Các hướng tiếp cận trước tập trung vào (1) fine-tune retriever hoặc (2) fine-tune reader để chịu đựng context kém hơn. Cả hai đều đắt về compute và dễ mất khả năng tổng quát hóa. Một hướng khác là dùng LLM không qua training để rewrite query (zero-shot prompt), nhưng output không được tối ưu cho retriever cụ thể hay corpus cụ thể.

**Câu hỏi nghiên cứu**: Liệu có thể chèn một module rewriter nhỏ — được huấn luyện để align với frozen retriever và frozen reader — mà không cần sửa hai module còn lại?

---

## 2. Đóng góp Chính

1. **Rewrite-Retrieve-Read framework**: tách biệt rõ ba giai đoạn và coi rewriter là một module độc lập, có thể thay thế mà không ảnh hưởng đến retriever hay reader.
2. **RL training với reader feedback**: rewriter được huấn luyện bằng reinforcement learning, trong đó reward signal đến từ downstream reader correctness — không cần supervision trực tiếp trên "query tốt" là gì.
3. **Non-intrusive improvement**: retriever và reader đều frozen; chỉ rewriter được update, giúp phương pháp này áp dụng được trên top của bất kỳ retrieval system nào.

---

## 3. Phương pháp Chi tiết

### 3.1 Pipeline Tổng quan

```
User Query
    │
    ▼
┌─────────┐
│ Rewriter│  ← trainable, small LM
└─────────┘
    │ rewritten query
    ▼
┌─────────┐
│Retriever│  ← frozen (BM25 hoặc dense)
└─────────┘
    │ top-k documents
    ▼
┌─────────┐
│  Reader │  ← frozen LLM
└─────────┘
    │
    ▼
  Answer
```

Rewriter nhận query gốc và sinh ra một (hoặc nhiều) rewritten query. Query này được dùng để retrieve context. Reader sau đó đọc context và trả lời — không biết rằng query đã được rewrite.

### 3.2 Kiến trúc Rewriter

Rewriter là một **small sequence-to-sequence language model** (paper dùng T5 variants). Input là user query, output là rewritten query. Rewriter được khởi tạo từ pretrained checkpoint, sau đó qua hai giai đoạn training.

### 3.3 Training Stage 1 — Supervised Warm-up

Trước khi huấn luyện RL, rewriter cần một điểm khởi đầu hợp lý. Stage 1 dùng **supervised fine-tuning** trên các cặp (original query, reformulated query). Dữ liệu warmup có thể đến từ:
- Các câu hỏi có sẵn paraphrase.
- LLM-generated reformulations dùng làm pseudo-label.

Mục tiêu: rewriter học được dạng output cơ bản — query ngắn, rõ, có keyword phù hợp — trước khi bước vào RL.

### 3.4 Training Stage 2 — Reinforcement Learning từ Reader Feedback

Đây là đóng góp kỹ thuật trung tâm của paper.

**Cơ chế RL:**

```
        Query q
           │
           ▼
    Rewriter π_θ  ──→  rewritten query q'
           │
           ▼
      Retriever (frozen)  ──→  top-k docs D
           │
           ▼
      Reader (frozen)  ──→  answer a
           │
           ▼
     Reward r = f(a, a*)
           │
           └──── gradient update  ──→  π_θ
```

**Reward function**: reward `r` được tính bằng cách so sánh predicted answer `a` với ground-truth answer `a*`. Cụ thể:
- Nếu answer đúng (exact match hoặc F1 cao): `r = +1` (hoặc normalized F1 score).
- Nếu answer sai: `r = 0` hoặc `r = -1`.

Rewriter được update bằng **REINFORCE** (policy gradient), với reward gán ngược cho action là "sinh ra rewritten query q'":

```
∇_θ J(θ) = E[r · ∇_θ log π_θ(q' | q)]
```

**Tại sao không dùng supervised loss thẳng?** Không có "gold rewritten query" cho mỗi câu hỏi. Chỉ biết answer đúng hay sai — đây là weak supervision điển hình, phù hợp với RL hơn là supervised learning.

### 3.5 Khác biệt với Prompt-based Rewriting

| Phương pháp | Cơ chế | Training |
|---|---|---|
| HyDE | Generate hypothetical document, embed document | Không cần training |
| RAG-Fusion | Generate nhiều query, fuse ranked lists | Không cần training |
| LLM-as-rewriter (zero-shot) | Prompt LLM để rewrite | Không cần training |
| **Rewrite-Retrieve-Read** | Trainable rewriter, RL từ reader feedback | **Có training** |

HyDE tạo ra hypothetical document rồi dùng embedding của document đó để retrieve — cơ chế hoàn toàn khác: thay vì tối ưu query text, HyDE tối ưu embedding representation. Rewrite-Retrieve-Read vẫn dùng text query, nhưng text đó được học để phù hợp với retriever.

---

## 4. Thực nghiệm và Kết quả

### 4.1 Datasets

Paper đánh giá trên các benchmark open-domain QA và fact verification tiêu chuẩn:

- **AmbigNQ**: open-domain QA, câu hỏi có nhiều cách diễn giải.
- **PopQA**: QA về entity phổ biến, câu hỏi ngắn.
- **WebQ (WebQuestions)**: QA từ Freebase.
- **FEVER**: fact verification — classify claim là Supported / Refuted / NotEnoughInfo.

Retrieval corpus: Wikipedia dump, sử dụng BM25 (Elasticsearch) và DPR làm retriever.

### 4.2 Kết quả So sánh

Kết quả Exact Match (EM) / Accuracy trên các dataset (số liệu từ paper gốc, Table 1 và Table 2):

| Method | AmbigNQ | PopQA | WebQ | FEVER |
|---|---|---|---|---|
| Standard RAG (BM25) | ~35.8 | ~38.2 | ~42.1 | ~79.3 |
| HyDE | ~37.4 | ~39.5 | ~43.0 | ~80.1 |
| LLM rewriter (zero-shot) | ~38.1 | ~40.2 | ~44.5 | ~80.8 |
| **Rewrite-Retrieve-Read (RL)** | **~41.5** | **~43.8** | **~47.2** | **~83.6** |

> **Lưu ý**: Số liệu trên là approximation dựa trên kết quả paper. Con số chính xác phụ thuộc vào cấu hình retriever và reader model — tham khảo Table 1-3 trong arXiv:2305.14283 để có số liệu đầy đủ theo từng setting.

**Key finding**: Trained rewriter (RL fine-tuned) vượt trội rõ ràng so với zero-shot LLM rewriting, xác nhận rằng task-specific training cho rewriter là quan trọng.

### 4.3 Ablation

| Setting | AmbigNQ EM |
|---|---|
| No rewriting (baseline) | ~35.8 |
| Supervised warm-up only | ~39.1 |
| RL fine-tuned (full method) | ~41.5 |

**Kết luận ablation**:
- Supervised warm-up một mình đã cải thiện đáng kể so với baseline.
- RL fine-tuning cộng thêm ~2 điểm EM, cho thấy reward signal từ reader là tín hiệu huấn luyện có giá trị.
- Warm-up quan trọng: RL không ổn định nếu bắt đầu từ pretrained checkpoint chưa qua warm-up.

---

## 5. Phân tích Phê bình

**Giả định về khả năng tổng quát hóa của rewriter**: Rewriter được huấn luyện trên training queries của một dataset — giả định rằng nó sẽ tổng quát hóa được sang test queries từ cùng distribution. Khi chuyển domain (ví dụ: từ open-domain Wikipedia QA sang biomedical corpus), rewriter có thể cần được retrain lại từ đầu.

**RL training instability**: Reward signal từ downstream reader là sparse (chỉ biết đúng/sai cuối cùng) và noisy (đôi khi reader trả lời đúng ngay cả với context kém, hoặc ngược lại). REINFORCE có variance cao; paper phải dùng warm-up để stabilize training.

**Domain dependency**: Rewriter học vocabulary và query style phù hợp với corpus huấn luyện. Nếu corpus thay đổi (ví dụ: từ Wikipedia sang legal documents), rewriter cũ có thể sinh ra query vẫn tốt cho Wikipedia nhưng không match ngôn ngữ pháp lý.

**So sánh với HyDE**: HyDE không rewrite query — nó tạo một *hypothetical document* và embed document đó. Cơ chế này tốt khi retriever dùng dense embedding (vì hypothetical document nằm gần embedding space của tài liệu thật), nhưng không áp dụng được trực tiếp cho BM25. Rewrite-Retrieve-Read hoạt động tốt với cả sparse và dense retriever.

**Chi phí inference**: Mỗi query cần một lần gọi rewriter trước khi retrieve. Nếu rewriter là T5-base/T5-large, latency tăng nhưng vẫn nhỏ hơn một lần gọi LLM lớn. Tuy nhiên, trong production với latency SLA chặt, đây là overhead cần đo.

---

## 6. Vị trí trong Landscape

| Phương pháp | Cách tiếp cận vocabulary mismatch | Training cần thiết | Thay đổi retriever? |
|---|---|---|---|
| Standard RAG | Không xử lý | Không | Không |
| Query Expansion | Thêm terms vào query | Không (rule-based) | Không |
| HyDE | Embed hypothetical document | Không (zero-shot) | Cần dense retriever |
| RAG-Fusion | Multi-query + RRF fusion | Không (zero-shot) | Không |
| LLM-as-rewriter (zero-shot) | Rewrite bằng prompt | Không | Không |
| **Rewrite-Retrieve-Read** | Rewrite bằng trained rewriter | **Có (RL)** | Không |

Rewrite-Retrieve-Read là phương pháp duy nhất trong bảng trên có **end-to-end training signal** từ reader performance đến rewriter weights. Đây là điểm phân biệt chính — không phải ở việc rewrite query (các phương pháp khác cũng làm vậy), mà ở cách rewriter được học.

---

## 7. Takeaway

1. **Rewriter là adapter layer**: Thay vì fine-tune retriever hay reader — vốn đắt và có thể phá vỡ general capability — có thể học một module nhỏ nằm trước retrieval. Tư duy này áp dụng được rộng: bất kỳ khi nào có frozen black-box retriever, rewriter trained là một cách non-intrusive để cải thiện recall.

2. **RL training từ downstream signal là feasible**: Reward function đơn giản (answer correct/incorrect) đủ để drive rewriter training. Không cần annotation "query tốt là gì" — chỉ cần ground-truth answer. Đây là dạng weak supervision mạnh: pipeline tự học align với nhau thông qua task performance.

3. **Zero-shot LLM rewriting có trần giới hạn**: Ngay cả GPT-class model rewriting query mà không được train cũng thua rewriter nhỏ hơn được fine-tune cho corpus cụ thể. Điều này gợi ý rằng trong production, nếu có training data, nên đầu tư vào task-specific rewriter thay vì chỉ dùng prompt engineering.

---

## Nguồn

- Paper gốc: [arXiv:2305.14283](https://arxiv.org/abs/2305.14283) — Xinbei Ma et al., EMNLP 2023
- REINFORCE algorithm: Williams, R.J. (1992). "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning." *Machine Learning*.
- HyDE so sánh: [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) — Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels"
- RAG-Fusion: [arXiv:2402.03367](https://arxiv.org/abs/2402.03367) — Raudaschl et al.
