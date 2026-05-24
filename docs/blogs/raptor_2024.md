# `raptor_2024`: Hierarchical Tree Indexing cho Long-Document Retrieval

> **Paper**: RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
> **Tác giả**: Parth Sarthi, Salman Abdullah, Aditi Sehrawat, Srividya Gollapalli, Anirudh Baggari, Christopher D. Manning
> **Venue**: ICLR 2024 — arXiv:2401.18059
> **Loại**: `paper_inspired`

RAPTOR giải quyết giới hạn cốt lõi của flat chunk retrieval: khi câu hỏi yêu cầu tổng hợp thông tin trải rộng nhiều đoạn văn bản, top-k cosine similarity không đủ để tìm được evidence. Bằng cách kết hợp Gaussian Mixture Model clustering với abstractive summarization đệ quy, RAPTOR xây dựng một cây chỉ mục đa tầng cho phép retrieval ở cả mức chi tiết lẫn mức tổng quan — và thực nghiệm cho thấy collapsed tree retrieval trên cây đó vượt trội rõ so với các baseline flat retrieval.

---

## 1. Bối cảnh và Động lực

### Giới hạn của flat chunk retrieval

Phần lớn hệ thống RAG hiện tại hoạt động theo mô hình flat: chia tài liệu thành chunks cố định, embed từng chunk, lưu vào vector store, rồi khi query đến thì lấy top-k chunks gần nhất theo cosine similarity. Cách này hoạt động tốt cho câu hỏi lookup cục bộ ("Tên CEO của công ty X là gì?") nhưng thất bại với câu hỏi tổng hợp:

- **Câu hỏi so sánh**: "Tác phẩm nào trong cuốn sách thể hiện chủ đề cô đơn rõ nhất?" — cần đọc và so sánh nhiều chương.
- **Câu hỏi về cấu trúc**: "Luận điểm chính của toàn bài nghiên cứu là gì?" — không nằm trong một chunk cụ thể.
- **Câu hỏi suy luận đa bước**: cần thông tin từ nhiều vị trí không liền kề trong tài liệu.

Với flat retrieval, kể cả khi tăng k lên rất lớn, các chunks trả về vẫn là những đoạn cục bộ ngẫu nhiên, không có thông tin về mối quan hệ và cấu trúc tổng thể của tài liệu.

### Tại sao parent-child chunking chưa đủ

Parent-child retrieval (tìm child chunk chính xác, trả về parent chunk rộng hơn) giải quyết được vấn đề bối cảnh cục bộ nhưng vẫn chỉ là hai tầng cứng. Nó không tạo ra representation của toàn bộ document hay representation của một nhóm section liên quan. Cây RAPTOR thực sự đệ quy và abstractive — không chỉ mở rộng window mà tạo ra nội dung mới ở mỗi tầng.

### Câu hỏi nghiên cứu

Liệu có thể xây dựng một cấu trúc chỉ mục đa tầng, trong đó mỗi tầng đại diện cho một mức trừu tượng khác nhau của cùng một corpus, và retrieval trên cấu trúc đó vượt trội so với flat retrieval cho cả câu hỏi cục bộ lẫn câu hỏi toàn cục?

---

## 2. Đóng góp Chính

**Contribution 1 — GMM clustering đệ quy tại indexing time**: Thay vì k-means hay hierarchical clustering cứng, RAPTOR dùng Gaussian Mixture Model để nhóm các text chunks theo không gian embedding. Việc này cho phép mỗi chunk thuộc về nhiều cluster với các xác suất khác nhau (soft assignment), phản ánh tốt hơn thực tế rằng một đoạn văn bản có thể liên quan đến nhiều chủ đề.

**Contribution 2 — Abstractive summarization tại mỗi tầng**: Mỗi cluster được tóm tắt bằng LLM để tạo ra một summary node. Summary này không phải extraction (copy-paste câu từ các chunks) mà là abstraction — LLM phải tổng hợp lại ý chính. Summary nodes được embed và đưa vào vòng lặp tiếp theo.

**Contribution 3 — Hai retrieval mode trên cây đã xây**: (a) *Collapsed tree*: phẳng hóa toàn bộ các node ở mọi tầng thành một pool, rồi dùng embedding similarity để retrieve — đơn giản nhưng hiệu quả; (b) *Tree traversal*: bắt đầu từ root, greedy descent xuống node con phù hợp nhất — phức tạp hơn nhưng kiểm soát được ngữ cảnh tại mỗi bước.

---

## 3. Phương pháp Chi tiết

### 3.1 Giai đoạn Indexing

**Bước 1 — Phân đoạn và embed leaf chunks**

Tài liệu gốc được chia thành các chunks với độ dài khoảng 100 token (leaf nodes). Mỗi chunk được embed bằng một text embedding model (trong paper gốc dùng SBERT / `sentence-transformers`). Kết quả là tập hợp vector $\{e_1, e_2, \ldots, e_n\}$ trong không gian chiều cao.

**Bước 2 — Giảm chiều và GMM clustering**

Do curse of dimensionality, khoảng cách cosine trong không gian embedding chiều cao thường không phân biệt rõ các cluster. RAPTOR dùng UMAP để giảm chiều xuống còn 2D trước khi fit GMM.

GMM clustering ưu việt hơn k-means vì ba lý do quan trọng:

| Tiêu chí | k-means | GMM |
|---|---|---|
| Gán cluster | Hard — mỗi điểm chỉ vào 1 cluster | Soft — mỗi điểm có xác suất thuộc mỗi cluster |
| Hình dạng cluster | Giả định cầu tròn đều nhau | Gaussian bất kỳ — hình ellipse, kích thước khác nhau |
| Xử lý chủ đề giao thoa | Không — buộc phải chọn 1 | Có — đoạn văn thuộc nhiều chủ đề được đưa vào nhiều summary |
| Ổn định với outlier | Kém | Tốt hơn (outlier có probability thấp đều) |

Cụ thể, với soft assignment, một đoạn văn bản nói về "kinh tế chính trị" có thể được đưa vào cả cluster "kinh tế" lẫn cluster "chính trị" nếu xác suất của nó ở cả hai cluster đều đủ cao (vượt ngưỡng). Ngưỡng thông thường là 0.5. Điều này đảm bảo thông tin cross-topic không bị mất.

Số cluster tối ưu được xác định bằng BIC (Bayesian Information Criterion) — RAPTOR không yêu cầu người dùng chỉ định k trước.

**Bước 3 — Abstractive summarization mỗi cluster**

Tất cả các chunks trong một cluster được nối lại và đưa vào LLM (paper dùng GPT-3.5/GPT-4) với prompt tóm tắt. Output là một summary node đại diện cho nội dung chung của cluster đó. Đây là bước quan trọng nhất: thay vì chỉ giữ metadata về cluster, RAPTOR tạo ra nội dung mới có thể được retrieve sau này.

**Bước 4 — Lặp đệ quy**

Summary nodes tiếp tục được embed, đưa vào UMAP + GMM, summarize lại. Quá trình này lặp cho đến khi tập hợp nodes còn lại đủ nhỏ để gom vào một cluster duy nhất (root node). Số tầng điển hình là 2–3 tầng với corpus vừa phải.

```
Tầng 0 (Leaf):  [c1] [c2] [c3] [c4] [c5] [c6] [c7] [c8]
                  \___/         \___/    \___/    \___/
Tầng 1 (L1):    [s1-2]         [s3-4]   [s5-6]  [s7-8]
                    \______________/       \_______/
Tầng 2 (L2):       [s(1-2)+(3-4)]        [s(5-8)]
                          \_________________/
Tầng 3 (Root):            [s_root]

Ghi chú:
  c = leaf chunk gốc
  s = summary node (do LLM generate)
  Mỗi tầng, GMM nhóm các node của tầng dưới
  và LLM tóm tắt mỗi nhóm thành summary node mới
```

### 3.2 Giai đoạn Query — Hai Retrieval Mode

**Mode 1: Collapsed Tree Retrieval**

Tất cả các nodes ở tất cả các tầng — leaf chunks, summary tầng 1, summary tầng 2, root — được đưa vào một flat retrieval pool duy nhất. Query được embed, rồi tính cosine similarity với tất cả nodes, lấy top-k.

```
Query embedding
     |
     v
Flat pool: {leaf_1..n} ∪ {summary_L1_1..m} ∪ {summary_L2_1..p} ∪ {root}
     |
     v
Cosine similarity ranking
     |
     v
Top-k nodes (có thể là hỗn hợp leaf + summary)
```

Ưu điểm: đơn giản, không cần biết cấu trúc cây, dễ tích hợp với bất kỳ vector store nào. Thực nghiệm cho thấy mode này thường đạt kết quả tốt hơn hoặc tương đương tree traversal.

**Mode 2: Tree Traversal Retrieval**

Bắt đầu từ root node, ở mỗi tầng tính cosine similarity giữa query và tất cả các node con trực tiếp, chọn top-k node con (thông thường k=1 hoặc k=2), rồi mở rộng xuống tầng tiếp theo, cho đến khi đến leaf. Đây là greedy beam search trên cây.

```
Query embedding
     |
     v
Root node → cosine sim với các L2 nodes → chọn top-1 L2 node
     |
     v
L2 node đã chọn → cosine sim với các L1 nodes con → chọn top-1 L1 node
     |
     v
L1 node đã chọn → cosine sim với các leaf nodes con → chọn top-k leaves
```

Nhược điểm: nếu greedy step sai ở tầng trên thì không thể phục hồi. Root node sau nhiều lần summarize có thể mất thông tin cụ thể, dẫn đến greedy step ở tầng đầu không chính xác cho các câu hỏi chi tiết. Đây là lý do tại sao collapsed tree thường vượt trội hơn tree traversal trong thực nghiệm.

**Lý do collapsed tree thường tốt hơn tree traversal**

Collapsed tree cho phép retriever chọn node ở bất kỳ tầng nào tùy theo query. Câu hỏi tổng hợp tự nhiên "kéo" về summary nodes, câu hỏi chi tiết tự nhiên "kéo" về leaf nodes — không cần greedy path từ trên xuống. Tree traversal bị ràng buộc bởi thứ tự top-down và không có khả năng "nhảy" tầng.

---

## 4. Thực nghiệm và Kết quả

### 4.1 Datasets và Setup

Paper đánh giá trên ba benchmark:

- **QASPER**: QA trên các paper nghiên cứu NLP, yêu cầu tổng hợp thông tin từ nhiều section.
- **QuALITY**: Multiple-choice QA trên các bài văn dài (truyện ngắn, bài báo), có đáp án cần suy luận.
- **NarrativeQA**: QA trên toàn bộ novel và movie scripts, câu hỏi về plot và nhân vật.

Baseline gồm: BM25, DPR (Dense Passage Retrieval), standard top-k chunk retrieval (flat), và retrieval chỉ dùng document-level summary.

### 4.2 Kết quả Chính

**Trên QASPER** (metric: F1 answer):

| Method | F1 |
|---|---|
| BM25 | 25.1 |
| DPR | 22.6 |
| Flat chunk retrieval (top-k) | 36.6 |
| RAPTOR — Tree Traversal | 36.8 |
| RAPTOR — Collapsed Tree | **36.6** |
| RAPTOR + GPT-4 (full pipeline) | **53.1** |

**Trên QuALITY** (metric: accuracy):

| Method | Accuracy |
|---|---|
| Flat chunk retrieval | 66.3 |
| RAPTOR — Collapsed Tree | **72.0** |
| RAPTOR + GPT-4 | **82.6** |

**Trên NarrativeQA** (metric: F1):

| Method | F1 |
|---|---|
| Flat chunk retrieval | 13.5 |
| RAPTOR — Collapsed Tree | **14.3** |
| RAPTOR + GPT-4 | **21.0** |

> **Lưu ý**: Các con số trên được trích từ paper (Bảng 1 và 2 trong arXiv:2401.18059). Kết quả RAPTOR + GPT-4 là kết hợp của RAPTOR retrieval và GPT-4 generation, không phải chỉ retrieval. Để so sánh fair giữa retrieval methods, so sánh các dòng cùng backbone LLM.

### 4.3 Ablation Study

**Collapsed tree vs tree traversal**: Trên QuALITY, collapsed tree đạt 72.0% trong khi tree traversal đạt 70.1%. Khoảng cách này nhỏ hơn trên QASPER (câu hỏi factoid hơn) nhưng rõ hơn trên NarrativeQA (câu hỏi toàn cục hơn).

**Số tầng summarization**: Tăng từ 1 tầng lên 2 tầng cho cải thiện đáng kể (~3-4% trên QuALITY). Từ 2 tầng lên 3 tầng cải thiện ít hơn và đôi khi không có lợi — vì summary quá trừu tượng ở tầng 3 mất chi tiết cụ thể.

**Mixed retrieval (leaf + summary)**: Việc retriever có thể chọn cả leaf nodes lẫn summary nodes trong collapsed tree là nguồn cải thiện chính. Nếu chỉ dùng summary-only retrieval, recall ở câu hỏi chi tiết giảm mạnh.

---

## 5. Phân tích Phê bình

**Giả định về chất lượng summary**: Toàn bộ pipeline phụ thuộc vào việc LLM tạo ra summaries chính xác và faithful. Nếu summary hallucinate thông tin hoặc bỏ sót chi tiết quan trọng, lỗi đó lan truyền lên tầng trên và không thể phục hồi. Đây là rủi ro hệ thống, không phải lỗi ngẫu nhiên.

**Chi phí build index**: Mỗi cluster ở mỗi tầng cần một lần gọi LLM. Với corpus N chunks và T tầng, số lần gọi LLM tỷ lệ với N (vì mỗi tầng giảm khoảng 10x số nodes). Với corpus lớn (hàng trăm nghìn chunks), chi phí này là đáng kể, đặc biệt khi dùng GPT-4.

**Incremental update khó**: Khi thêm tài liệu mới vào corpus, cây phải được rebuild từ tầng bị ảnh hưởng trở lên. Không thể chỉ thêm leaf nodes mới mà không ảnh hưởng đến cấu trúc cluster và summaries ở tầng trên.

**Hallucination lan truyền theo cây**: Một summary sai ở tầng 1 sẽ được include vào cluster tầng 2, ảnh hưởng summary tầng 2. Không có cơ chế nào trong RAPTOR để phát hiện hay sửa lỗi này sau khi đã build.

**Phạm vi đánh giá**: Ba datasets trong paper đều là literary/academic text (tiểu thuyết, paper NLP, bài báo dài). Hiệu quả của RAPTOR trên technical documentation, code, hay structured data (bảng, JSON) chưa được đánh giá trong paper gốc. Domain khác nhau có thể cho kết quả rất khác.

**UMAP là non-deterministic**: Việc dùng UMAP trước GMM thêm một nguồn randomness vào pipeline. Hai lần build trên cùng corpus có thể tạo ra cây khác nhau, ảnh hưởng reproducibility.

---

## 6. Vị trí trong Landscape

### So sánh các phương pháp hierarchical retrieval

| Method | Cấu trúc chỉ mục | Retrieval | LLM at indexing? | Incremental update | Chi phí build |
|---|---|---|---|---|---|
| Flat chunking | Flat list | cosine similarity | Không | Dễ — chỉ thêm | Thấp |
| Parent-child chunking | 2-level cứng | child → expand parent | Không | Dễ | Thấp |
| RAPTOR | N-level tree (3–5) | Collapsed tree hoặc tree traversal | Có (nhiều lần) | Khó — rebuild subtree | Cao |
| GraphRAG (Microsoft) | Knowledge graph | Graph traversal + community summary | Có (rất nhiều lần) | Rất khó | Rất cao |

### Khi nào hierarchical indexing đáng chi phí

RAPTOR đáng đầu tư khi:
- Corpus ổn định và ít thay đổi (sách, báo cáo, tài liệu nghiên cứu).
- Câu hỏi thường yêu cầu tổng hợp nhiều phần (global questions chiếm >30% workload).
- Có sẵn LLM API đủ rẻ để summarize (GPT-3.5 cho indexing là hợp lý).

RAPTOR không nên dùng khi:
- Corpus update liên tục (news feed, log, real-time data).
- Câu hỏi chủ yếu là lookup cục bộ — flat retrieval đã đủ.
- Budget LLM API eo hẹp và corpus rất lớn.

---

## 7. Takeaway

**Insight 1 — Soft assignment là chìa khóa của GMM**: Với k-means, một đoạn văn bản ở biên giới hai chủ đề sẽ bị gán cứng vào một cluster và biến mất khỏi cluster kia. GMM với soft assignment đảm bảo thông tin cross-topic được đưa vào nhiều summaries, giảm information loss khi xây cây.

**Insight 2 — Collapsed tree đơn giản hơn và thường tốt hơn tree traversal**: Tree traversal nghe có vẻ "thông minh" hơn vì nó tận dụng cấu trúc cây, nhưng thực ra greedy descent là fragile. Collapsed tree không cần biết cấu trúc cây và cho phép retriever tự tìm đúng mức trừu tượng cho từng query — điều này phù hợp hơn với cách embedding similarity hoạt động.

**Insight 3 — Hierarchical indexing giải quyết query type mà flat retrieval không thể**: Cải thiện rõ nhất của RAPTOR là ở QuALITY và NarrativeQA — datasets yêu cầu hiểu cấu trúc tổng thể của văn bản. Trên QASPER (factoid hơn), gap nhỏ hơn. Điều này cho thấy hierarchical indexing là giải pháp đúng cho đúng vấn đề — nó không phải là "flat retrieval tốt hơn" mà là giải pháp cho một loại câu hỏi khác.

---

## Nguồn

- Sarthi, P., Abdullah, S., Sehrawat, A., Gollapalli, S., Baggari, A., & Manning, C. D. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. *ICLR 2024*. https://arxiv.org/abs/2401.18059
- QASPER dataset: Dasigi et al. (2021). A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers. *NAACL 2021*.
- QuALITY benchmark: Pang et al. (2022). QuALITY: Question Answering with Long Input Texts, Yes! *NAACL 2022*.
- NarrativeQA: Kočiský et al. (2018). The NarrativeQA Reading Comprehension Challenge. *TACL 2018*.
