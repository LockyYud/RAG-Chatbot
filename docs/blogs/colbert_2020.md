# `colbert_2020`: Late Interaction — Khi Mỗi Token Đều Được Lắng Nghe

> **Paper**: ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT
> **Tác giả**: Omar Khattab, Matei Zaharia
> **Venue**: SIGIR 2020 — arXiv:2004.12832
> **Loại**: `paper_reproduction`

ColBERT giải quyết tension cốt lõi của information retrieval hiện đại: bi-encoder nhanh nhưng kém chính xác vì encode query và passage độc lập thành một vector duy nhất, trong khi cross-encoder chính xác nhưng chậm vì phải tính attention đầy đủ mỗi lần có query. Late interaction — tính MaxSim trên tất cả token embeddings — cho phép pre-compute passage representations như bi-encoder nhưng vẫn giữ được token-level interaction, đạt accuracy gần cross-encoder với latency gần bi-encoder.

---

## 1. Bối cảnh và Động lực

### Hai thế giới không tương thích

Retrieval system hiện đại đứng trước lựa chọn khó: bi-encoder (dual-encoder) hay cross-encoder?

**Bi-encoder** (DPR, SBERT): encode query và passage thành hai vector độc lập, tính cosine/dot-product similarity. Toàn bộ passage corpus có thể được pre-computed và lưu vào FAISS index — query chỉ cần một lần encode. Kết quả: retrieval trong mili-giây. Nhưng khi encode riêng biệt, bi-encoder mất hoàn toàn token-level interaction: từ "bank" trong query "river bank" không biết đang ở cạnh "financial bank" hay "river bank" trong passage.

**Cross-encoder** (BERT reranker): concatenate query và passage, chạy full BERT — mọi query token được attend với mọi passage token qua many-to-many self-attention. Chính xác hơn đáng kể, nhưng không thể pre-compute passage representations. Với corpus 9 triệu passages như MS MARCO, cross-encoder cần 9M lần BERT forward pass mỗi query — thực tế là không khả thi.

**Late interaction** là câu trả lời của Khattab và Zaharia: "tại sao phải chọn một trong hai?"

### Khoảng trống về cơ chế

Điểm yếu cơ bản của bi-encoder là buộc nén toàn bộ meaning của passage vào một vector. Với [CLS] token representation, câu hỏi "Who invented the telephone?" và một passage nói về "Alexander Graham Bell invented the telephone in 1876" có thể có high cosine similarity nếu embedding model tốt. Nhưng câu hỏi "What did Alexander Bell invent?" và "Bell invented the telephone" — trong một sentence embedding, sự khác biệt về subject/object có thể bị mờ đi.

Cross-encoder không có vấn đề này vì nó giữ toàn bộ token-level context. Câu hỏi là: **làm sao giữ token-level matching mà vẫn pre-computable?**

### Research question

Liệu có thể thiết kế một retrieval model cho phép pre-compute passage token representations, nhưng vẫn tính similarity ở token level thay vì vector level — đạt được accuracy của cross-encoder với efficiency của bi-encoder?

---

## 2. Đóng góp Chính

**Contribution 1 — Late Interaction paradigm**: ColBERT tách rõ *encoding* (offline, pre-computable) khỏi *interaction* (online, per-query). Passage token embeddings được tính một lần và lưu vào index. Query token embeddings được tính online. Interaction function (MaxSim) kết hợp hai tập embeddings mà không cần re-encode.

**Contribution 2 — MaxSim operator**: Thay vì một số duy nhất đại diện cho similarity giữa query và passage, ColBERT tính: với mỗi query token, tìm passage token có similarity cao nhất (max), rồi cộng tất cả lại (sum). Operator này đơn giản, differentiable, và phản ánh trực quan về matching: query được "trả lời" nếu mỗi phần của nó tìm được passage token match tốt.

**Contribution 3 — Efficient index và query serving**: Tất cả passage token vectors được lưu trong FAISS index. Retrieval diễn ra trong hai bước: (1) approximate nearest neighbor search từ query token vectors để tìm candidate passages, (2) exact MaxSim rescoring trên candidates. Approach này cho phép retrieval trên MS MARCO 9M passages trong ~45ms — 75x nhanh hơn cross-encoder.

---

## 3. Phương pháp Chi tiết

### 3.1 Query và Passage Encoding

ColBERT dùng hai BERT encoder với weights riêng biệt (hoặc shared — paper test cả hai, riêng biệt thường tốt hơn một chút):

**Query encoder** $E_Q$:
- Input: `[Q]` token + query tokens (dùng `[MASK]` padding đặc biệt — xem bên dưới)
- Output: ma trận $\mathbf{H}_q \in \mathbb{R}^{m \times d}$ với $m$ là số query tokens, $d$ là embedding dimension
- Bước cuối: L2-normalize mỗi row

**Passage encoder** $E_P$:
- Input: `[D]` token + passage tokens
- Output: ma trận $\mathbf{H}_p \in \mathbb{R}^{n \times d}$ với $n$ là số passage tokens
- Bước cuối: L2-normalize mỗi row

ColBERT dùng dimension $d = 128$ thay vì BERT's full 768 — projection layer giảm từ 768 xuống 128, giảm 6x index size mà accuracy giảm không đáng kể.

**Query augmentation với [MASK] tokens**: Query thường ngắn hơn passage nhiều. Để query có đủ "expressiveness", ColBERT pad query đến fixed length bằng `[MASK]` tokens. Những `[MASK]` này được contextualized qua BERT attention — về mặt lý thuyết chúng có thể "học" cách expand query terms. Trong practice, đây là engineering trick đơn giản giúp cải thiện recall một chút.

### 3.2 MaxSim Interaction

Với query $q$ và passage $p$, ColBERT tính similarity score:

$$S(q, p) = \sum_{i=1}^{|q|} \max_{j=1}^{|p|} \mathbf{H}_q[i] \cdot \mathbf{H}_p[j]^T$$

Diễn đạt bằng ngôn ngữ tự nhiên: với **mỗi query token** $i$, tìm passage token $j$ nào có dot product cao nhất với nó (đó là max over $j$), rồi **cộng tất cả các max scores** đó lại.

```
Query: [Q] "who founded apple"
      → [e_Q, e_who, e_founded, e_apple, e_MASK, e_MASK]
                ↕ MaxSim ↕
Passage: [D] "Steve Jobs co-founded Apple in 1976"
       → [e_D, e_Steve, e_Jobs, e_co-founded, e_Apple, e_in, e_1976]

Score = max(e_Q × each passage token)    → 0.1 (weak match)
      + max(e_who × each passage token)  → 0.6 (match với e_Steve hoặc e_Jobs)
      + max(e_founded × each passage token) → 0.9 (match với e_co-founded)
      + max(e_apple × each passage token)   → 0.95 (match với e_Apple)
      + max(e_MASK₁ × ...)               → 0.3
      + max(e_MASK₂ × ...)               → 0.2
                                          ──────
Total score                              → 3.05
```

MaxSim có một tính chất quan trọng: **mỗi query token ảnh hưởng đến score độc lập**. Nếu passage chứa tất cả query terms thì mỗi term đều đóng góp cao. Nếu passage chỉ match một phần, tổng score thấp hơn. Điều này khác với bi-encoder — ở đó một vector trung bình có thể bị dominated bởi phần lớn của câu mà miss important term.

### 3.3 Training

Training dùng pairwise softmax cross-entropy loss:

$$\mathcal{L} = -\log \frac{e^{S(q, p^+)}}{\sum_{p' \in \{p^+\} \cup \mathcal{N}} e^{S(q, p')}}$$

Trong đó:
- $p^+$: positive passage (ground truth relevant)
- $\mathcal{N}$: set of negative passages

Negatives gồm hai loại:
1. **In-batch negatives**: passages của các queries khác trong cùng batch — cheap
2. **BM25 hard negatives**: passages xếp cao bởi BM25 nhưng không phải ground truth — quan trọng cho quality

Paper tạo training data từ MS MARCO triples (query, positive passage, hard negative passage). Fine-tune từ BERT-base-uncased.

### 3.4 Index và Retrieval Pipeline

```
Offline (indexing):
  Với mỗi passage p trong corpus:
    compute H_p = E_P(p)           # matrix [n × 128]
    store H_p in FAISS IVF index
    store passage metadata (text, ID)

Online (retrieval):
  Input: query q
  
  Step 1 — Candidate generation:
    compute H_q = E_Q(q)           # matrix [m × 128]
    for each query token e_qi in H_q:
      run ANN search in FAISS       # find ~nprobe × k passage tokens
    union all candidate passage IDs → C (set of ~1000 passages)
  
  Step 2 — Exact scoring:
    for each passage p in C:
      load H_p from index
      score = MaxSim(H_q, H_p)
    sort C by score descending
    return top-k
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 MS MARCO Passage Ranking (main benchmark)

MS MARCO Development Set — MRR@10 (Mean Reciprocal Rank tại 10 kết quả):

| Method | MRR@10 | Ghi chú |
|--------|--------|---------|
| BM25 (Lucene) | 18.4 | Lexical baseline |
| FastText bi-encoder | 20.5 | Shallow dense |
| BERT bi-encoder | 33.8 | DPR-style |
| ColBERT | **34.9** | Late interaction |
| BERT cross-encoder (reranker) | 37.1 | Upper bound, không scalable |

ColBERT thu hẹp khoảng cách giữa bi-encoder và cross-encoder từ 3.3 điểm xuống còn 2.2 điểm (34.9 vs 37.1), trong khi latency chỉ là ~45ms so với ~3400ms của cross-encoder.

### 4.2 TREC 2019 Passage Ranking — NDCG@10

| Method | NDCG@10 |
|--------|---------|
| BM25 | 50.6 |
| BERT bi-encoder | 59.2 |
| ColBERT | 69.3 |
| BERT cross-encoder | 72.4 |

Trên TREC 2019, gap của ColBERT so với cross-encoder là nhỏ (2.3 điểm) trong khi gap với bi-encoder là 10.1 điểm — chứng tỏ late interaction thực sự tạo ra chất lượng khác biệt.

### 4.3 End-to-End Latency và Storage

| System | Latency (ms/query) | Index Size (9M passages) |
|--------|-------------------|-----------------------|
| BM25 | ~50 | ~10GB |
| Bi-encoder (FAISS) | ~25 | ~26GB |
| **ColBERT** | **~45** | **~154GB** |
| Cross-encoder | ~3400 | N/A (no index) |

ColBERT chậm hơn bi-encoder một chút (2 stage retrieval) nhưng 75x nhanh hơn cross-encoder. Index lớn hơn đáng kể (6x so với bi-encoder) do lưu tất cả token vectors.

### 4.4 Ablation Study

| Variant | MRR@10 |
|---------|--------|
| ColBERT (full) | 34.9 |
| Không có query augmentation ([MASK]) | 34.1 |
| Shared encoder (Q=P weights) | 34.3 |
| Dimension 768 (không project) | 35.0 |
| Dimension 64 | 34.2 |

Dimension 128 là sweet spot: 768 tốt hơn nhẹ nhưng index 6x lớn hơn; 64 rẻ hơn nhưng quality drop rõ.

---

## 5. Phân tích Phê bình

**MaxSim bị ảnh hưởng bởi query token frequency**: Sum over all query tokens có nghĩa là queries dài hơn tự nhiên có score cao hơn. Worse, stopwords trong query cũng đóng góp vào score — token "the" hay "of" vẫn tìm được passage token có max similarity. Paper dùng query augmentation và L2 normalization để giảm thiểu, nhưng không eliminate hoàn toàn. Điều này có thể dẫn đến over-retrieval cho verbose queries.

**Index size là bottleneck thực tế**: 154GB cho MS MARCO (9M passages) là thực sự lớn cho deployment. Một bi-encoder dùng 26GB — 6x nhỏ hơn. Với corpus lớn hơn (100M documents), ColBERT index sẽ cần ~1.7TB, so với ~290GB của bi-encoder. Paper không thảo luận về production deployment constraints này.

**Evaluation chủ yếu trên MS MARCO (English)**: MS MARCO là question-answer pairs có distribution đặc thù — queries ngắn, passages từ web. Generalization sang domain khác (legal, medical, code) hay ngôn ngữ khác chưa được explored trong paper gốc. BEIR benchmark (2021, sau ColBERT) sẽ cho thấy câu chuyện phức tạp hơn.

**Hard negative selection chưa được explore**: Paper gốc chỉ dùng BM25 hard negatives. Sau đó ColBERTv2 sẽ chứng minh rằng hard negatives từ ColBERT v1 chính nó (deceptive negatives) improve training đáng kể. Đây là limitation của thiết kế training trong paper gốc.

**Two-stage retrieval complexity**: Candidate generation bằng ANN trên token vectors rồi exact MaxSim rescoring là engineering phức tạp. Nprobe parameter (số clusters FAISS scan) affect recall/latency tradeoff và cần tuning cho mỗi corpus.

---

## 6. Vị trí trong Landscape

### So sánh các paradigm retrieval

| Paradigm | Representation | Matching | Pre-compute passage? | Accuracy | Latency | Index Size |
|----------|---------------|---------|---------------------|----------|---------|------------|
| BM25 | TF-IDF inverted index | Exact term | Có | Thấp (lexical only) | ~50ms | Thấp |
| Bi-encoder (DPR) | 1 vector/text | Dot product | Có | Trung bình | ~25ms | Thấp |
| **ColBERT (late interaction)** | **n vectors/text** | **MaxSim** | **Có** | **Cao** | **~45ms** | **Cao** |
| Cross-encoder | Token joint | Full attention | Không | Rất cao | ~3400ms | N/A |
| SPLADE | Sparse token weights | Sparse dot product | Có | Cao | ~30ms | Trung bình |

ColBERT mở ra một paradigm mới: retrieval models không phải là "one representation per text" mà là "bag of token representations". SPLADE (Formal et al., 2021) tiếp cận bài toán khác — sparse token weighting — và cũng nằm trong khoảng giữa bi-encoder và cross-encoder.

### Khi nào ColBERT đáng dùng

**Nên dùng khi:**
- Corpus ổn định và có thể pre-index (không update quá thường)
- Có storage budget cho index lớn
- Cần accuracy gần cross-encoder nhưng không có latency của cross-encoder
- Câu hỏi phức tạp, nhiều tokens quan trọng (ColBERT bắt multi-term matching tốt)

**Không nên dùng khi:**
- Storage là constraint nghiêm ngặt (dùng bi-encoder)
- Corpus update liên tục (re-indexing tốn kém hơn bi-encoder)
- Đã có cross-encoder reranker trên candidate pool nhỏ (gap không đáng)
- Query rất ngắn (1-2 tokens): MaxSim benefit giảm đáng kể

ColBERT v1 → ColBERT v2 (Santhanam et al., 2022) sẽ giải quyết vấn đề index size bằng residual compression và cải thiện accuracy bằng distillation — xem [colbertv2_2022.md](./colbertv2_2022.md).

---

## 7. Takeaway

**MaxSim là inductive bias đúng cho retrieval**: Bi-encoder giả định meaning của cả passage có thể được nén vào một vector — đây là giả định quá mạnh cho heterogeneous passages. MaxSim giả định rằng "passage liên quan" nghĩa là "mỗi phần của query đều tìm được match trong passage" — trực quan và đúng hơn về mặt thông tin.

**Late interaction giải phóng encoder khỏi bottleneck compression**: Khi passage encoder không phải map cả passage vào một điểm trong vector space, nó có thể giữ lại thông tin chi tiết ở token level. Đây là lý do ColBERT outperform bi-encoder ngay cả với encoder cùng size — không phải vì model mạnh hơn mà vì representation richer.

**Index size là giá thực sự của late interaction**: 6x index size so với bi-encoder không phải là chi tiết nhỏ. Trong production, đây là quyết định infrastructure quan trọng. ColBERTv2 giải quyết được phần lớn vấn đề này (xuống ~25GB), nhưng tradeoff vẫn tồn tại. Luôn benchmark accuracy gain vs storage cost trước khi adopt.

---

## Nguồn

- Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR 2020*. https://arxiv.org/abs/2004.12832
- Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*. https://arxiv.org/abs/2004.04906
- Nogueira, R., & Cho, K. (2019). Passage Re-ranking with BERT. arXiv:1901.04085
- Formal, T., et al. (2021). SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking. *SIGIR 2021*.
- Santhanam, K., et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction. *NAACL 2022*. https://arxiv.org/abs/2112.01488
