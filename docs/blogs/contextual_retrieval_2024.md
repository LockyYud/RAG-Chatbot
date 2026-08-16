# `contextual_retrieval_2024`: Cứu Ngữ Cảnh Bị Mất Khi Chunking

> **Nguồn**: *Introducing Contextual Retrieval*, Anthropic, 2024
> **Tác giả**: Anthropic
> **Venue**: Anthropic Engineering Blog, 09/2024 — <https://www.anthropic.com/news/contextual-retrieval>
> **Loại**: `paper_inspired`

Chunking để retrieval vô tình phá hủy chính thứ làm cho một đoạn văn bản có thể tìm được: ngữ cảnh xung quanh nó. Contextual Retrieval sửa lỗi này bằng một bước rẻ và đơn giản — dùng LLM viết một câu ngữ cảnh cho mỗi chunk *dựa trên toàn bộ tài liệu*, rồi prepend vào chunk trước khi index. Không đổi kiến trúc retrieval, chỉ đổi cái được đưa vào index.

---

## 1. Bối cảnh và Động lực

RAG tiêu chuẩn cắt tài liệu thành chunk nhỏ (vài trăm token) để embedding sắc nét và để vừa context window. Nhưng mỗi chunk được index *độc lập*, mất liên kết với tài liệu gốc.

Ví dụ kinh điển của Anthropic: một chunk ghi "The company's revenue grew by 3% over the previous quarter." Chunk này không nói *công ty nào*, *quý nào*. Khi người dùng hỏi "ACME Corp tăng trưởng doanh thu Q2 2023 bao nhiêu?", cả BM25 (thiếu keyword "ACME", "Q2 2023") lẫn dense retrieval (vector mờ nghĩa, không neo được vào thực thể cụ thể) đều dễ trượt chunk này.

Câu hỏi nghiên cứu: làm sao khôi phục ngữ cảnh tài liệu cho từng chunk *mà không* phải từ bỏ chiến lược chunk nhỏ?

---

## 2. Đóng góp Chính

- **Contextual Embeddings**: trước khi embedding, prepend vào mỗi chunk một đoạn ngữ cảnh 50–100 token do LLM sinh ra, giải thích chunk nằm ở đâu trong tài liệu.
- **Contextual BM25**: dùng đúng chunk đã được prepend ngữ cảnh để xây inverted index — nên cả tín hiệu lexical cũng giàu thực thể/định danh hơn.
- **Kết hợp với hybrid + rerank**: Anthropic chỉ ra contextualization cộng dồn lợi ích với RRF fusion và cross-encoder reranking, không thay thế chúng.

---

## 3. Phương pháp Chi tiết

### Sinh ngữ cảnh cho từng chunk

Với mỗi chunk, gọi LLM một lần với cả tài liệu và chunk:

```
<document> {WHOLE_DOCUMENT} </document>
Here is the chunk we want to situate within the whole document:
<chunk> {CHUNK} </chunk>
Give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

Kết quả (~50–100 token) được prepend vào chunk:

```
contextualized_chunk = context + "\n\n" + original_chunk
```

`contextualized_chunk` được dùng cho **cả** embedding **và** BM25 index. Văn bản đưa cho generator vẫn nên là chunk gốc — ngữ cảnh tổng hợp là phụ trợ cho retrieval, không phải nội dung để trích dẫn.

### Vì sao prepend thay vì các lựa chọn khác

- *Vì sao không tóm tắt cả tài liệu rồi index?* Tóm tắt mất chi tiết; chunk vẫn cần giữ nguyên văn để trả lời chính xác.
- *Vì sao dùng LLM thay vì chỉ thêm tiêu đề mục?* Tiêu đề mục (như `SectionTitleEnricher` trong repo) chỉ nắm cấu trúc; LLM nắm được quan hệ ngữ nghĩa ("đây là số liệu Q2 2023 của ACME trong phần báo cáo tài chính").

### Cost trick: prompt caching

Document được lặp lại cho mọi chunk của nó. Anthropic dùng **prompt caching** để cache phần document, nên chỉ trả tiền đọc document một lần mỗi tài liệu thay vì mỗi chunk — đưa chi phí xuống mức ~1 USD/triệu chunk token.

---

## 4. Thực nghiệm và Kết quả

**Metric**: tỉ lệ retrieval *thất bại* ở top-20 (chunk liên quan không nằm trong 20 kết quả đầu) — càng thấp càng tốt.

### Kết quả Chính (Anthropic báo cáo)

| Cấu hình | Top-20 failure rate | Giảm tương đối |
|---|---|---|
| Embeddings thường (baseline) | 5.7% | — |
| Contextual Embeddings | 3.7% | −35% |
| Contextual Embeddings + Contextual BM25 | 2.9% | −49% |
| + Reranking | **1.9%** | **−67%** |

### Ablation quan trọng

Hai trục cộng dồn rõ rệt: (1) thêm BM25 ngữ cảnh hóa vào embeddings ngữ cảnh hóa kéo failure từ 3.7%→2.9%; (2) thêm reranker kéo tiếp xuống 1.9%. Đây là lý do technique này được dựng *trên nền* `bm25_hybrid_rerank` thay vì thay thế nó.

---

## 5. Phân tích Phê bình

**Assumption ẩn**: chất lượng ngữ cảnh phụ thuộc hoàn toàn vào năng lực của `context_model`. Một model yếu có thể sinh ngữ cảnh sai/nhiễu, làm *giảm* recall — Anthropic không định lượng độ nhạy này.

**Limitation Anthropic thừa nhận**: chi phí ingest tăng (một lần gọi LLM/chunk); họ giảm bằng prompt caching nhưng vẫn tốn hơn chunking thường.

**Limitation không nêu**: với chunk vốn đã tự đủ nghĩa (mục FAQ, định nghĩa độc lập), ngữ cảnh thêm vào hầu như không giúp gì mà vẫn tốn tiền. Lợi ích tỉ lệ thuận với độ "phụ thuộc ngữ cảnh" của corpus.

**Reproducibility**: đây là blog kỹ thuật kèm cookbook code công khai, không phải paper peer-reviewed; con số là nội bộ Anthropic trên tập đánh giá riêng, nên coi là *chỉ dấu xu hướng*, cần benchmark lại trên corpus của mình.

---

## 6. Vị trí trong Landscape

| Method | Ý tưởng lõi | Khi nên dùng | Hạn chế chính |
|---|---|---|---|
| Contextual Retrieval | LLM viết ngữ cảnh/chunk trước khi index | Tài liệu dài, chunk mất ngữ cảnh | Tốn LLM lúc ingest |
| Parent-Child / small-to-big | Index chunk nhỏ, trả về parent lớn | Tài liệu có heading rõ | Không thêm ngữ cảnh ngữ nghĩa mới |
| RAPTOR | Cây tóm tắt đệ quy nhiều tầng | QA tài liệu rất dài, câu hỏi global | Xây index nặng |
| HyDE | Sinh câu trả lời giả để embed query | Mismatch từ vựng phía query | Không sửa phía index |

Contextual Retrieval nằm ở *phía index* và bổ trợ chứ không cạnh tranh với các kỹ thuật phía query (HyDE) hay phía cấu trúc (RAPTOR, parent-child).

---

## 7. Takeaway

1. **Ngữ cảnh bị mất khi chunk là một failure mode có thật và rẻ để sửa**: một câu LLM/chunk kéo top-20 failure giảm tới một nửa.
2. **Cùng một prepend phục vụ cả dense lẫn lexical**: vì BM25 và embedding đều đọc văn bản chunk, contextualization "miễn phí" áp cho cả hai tín hiệu.
3. **Đây là lớp bổ trợ, không phải thay thế**: lợi ích cộng dồn với hybrid fusion và reranking — nên benchmark nó *trên cùng* backbone hybrid để cô lập đúng phần nó đóng góp.

---

## Nguồn

- Anthropic (2024). *Introducing Contextual Retrieval*. <https://www.anthropic.com/news/contextual-retrieval>
- Anthropic Cookbook — Contextual Embeddings (code mẫu). <https://github.com/anthropics/anthropic-cookbook>
- Liên quan trong repo: [`bm25_hybrid_rerank`](./bm25_hybrid_rerank.md) (backbone retrieval), [`raptor_2024`](./raptor_2024.md) (hierarchy phía index)
