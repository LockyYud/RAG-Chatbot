# `adaptive_rag_2024`: Classifier Định Tuyến Query Theo Độ Phức Tạp để Cân Bằng Accuracy và Efficiency

> **Paper**: Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity
> **Tác giả**: Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park et al.
> **Venue**: NAACL 2024 — arXiv:2403.14403
> **Loại**: `paper_reproduction`

Adaptive-RAG huấn luyện một small classifier để tự động chọn giữa ba chiến lược retrieval — no-retrieval, single-step, và multi-step — dựa trên độ phức tạp của từng query, thay vì áp dụng một pipeline cố định cho mọi câu hỏi. Điểm then chốt là nhãn huấn luyện cho classifier được tạo tự động từ bias của dataset mà không cần annotation thủ công.

---

## 1. Bối cảnh và Động lực

### Giới hạn của các phương pháp trước

Các hệ thống RAG hiện tại thường theo một trong hai hướng cực đoan:

- **No-retrieval** (ví dụ: standard LLM prompting): nhanh và rẻ, nhưng thất bại trên các câu hỏi knowledge-intensive đòi hỏi thông tin ngoài tầm của parametric memory.
- **Single-step RAG** (retrieve-then-read): phù hợp với câu hỏi đơn hop, nhưng lãng phí compute trên câu hỏi đơn giản và không đủ cho câu hỏi multi-hop.
- **Multi-step RAG** (ví dụ: IRCoT, Self-Ask): mạnh với câu hỏi phức tạp, nhưng gây overhead lớn — nhiều lần retrieval, nhiều lần gọi LLM — cho câu hỏi chỉ cần một hop hoặc không cần retrieval.

Vấn đề sâu hơn: không có phương pháp nào trước đó tự động phân loại một query vào đúng "mức" cần thiết. Một số hệ thống dùng confidence-based triggering (ví dụ: FLARE), nhưng đó là reactive — chỉ quyết định sau khi đã bắt đầu generate — thay vì proactive trước khi chạy pipeline.

### Khoảng trống nghiên cứu

Câu hỏi nghiên cứu cụ thể: *Có thể học một classifier nhỏ, rẻ để định tuyến query sang đúng chiến lược RAG mà không cần human annotation không, và classifier đó có cải thiện cả accuracy lẫn efficiency không?*

---

## 2. Đóng góp Chính

- **Khung phân loại ba cấp**: Định nghĩa rõ ràng ba chiến lược — (A) no-retrieval, (B) single-step retrieval, (C) multi-step retrieval — và đặt bài toán chọn chiến lược thành bài toán phân loại.
- **Tạo nhãn tự động từ bias dataset**: Thay vì annotate thủ công, paper dùng kết quả thực nghiệm của các model/strategy trên các dataset đã có bias phức tạp khác nhau để tạo nhãn training cho classifier.
- **Small classifier T5-based**: Huấn luyện một classifier nhẹ (T5-large hoặc nhỏ hơn) trên các nhãn tự động tạo ra, đủ nhỏ để overhead prediction không đáng kể so với cost của pipeline được chọn.
- **Kết hợp adaptive selection với bất kỳ retriever/generator nào**: Thiết kế modular — classifier chỉ quyết định strategy, không ràng buộc với model cụ thể trong từng strategy.

---

## 3. Phương pháp Chi tiết

### Ba chiến lược

| Chiến lược | Mô tả | Khi nào phù hợp |
|---|---|---|
| **(A) No-Retrieval** | LLM trả lời trực tiếp từ parametric knowledge | Câu hỏi đơn giản, LLM đã biết |
| **(B) Single-Step Retrieval** | Retrieve một lần → generate | Câu hỏi đơn hop, cần 1 document |
| **(C) Multi-Step Retrieval** | Iterative retrieve + reason (IRCoT-style) | Câu hỏi multi-hop, cần tổng hợp nhiều tài liệu |

### Classifier: Kiến trúc và Input

Classifier là một mô hình **T5** (hoặc nhỏ hơn) được fine-tune cho bài toán phân loại ba lớp {A, B, C}. Input là câu hỏi q dưới dạng chuỗi văn bản, không cần feature engineering thủ công:

$$f_\theta: q \mapsto \{A, B, C\}$$

trong đó $f_\theta$ là classifier với tham số $\theta$, được tối ưu bằng cross-entropy loss:

$$\mathcal{L} = -\sum_{i} \log P_\theta(y_i \mid q_i)$$

### Tạo nhãn huấn luyện tự động

Đây là phần độc đáo nhất của paper. Thay vì annotation thủ công:

1. **Khai thác dataset bias**: Các dataset khác nhau có bias phức tạp khác nhau — PopQA và WebQ thường là single-hop; MuSiQue, HotpotQA, 2WikiMultiHopQA là multi-hop; một số câu hỏi trivial không cần retrieval.
2. **Chạy cả ba chiến lược** trên mỗi câu hỏi training, ghi lại strategy nào cho kết quả đúng.
3. **Gán nhãn theo outcome**: Nếu strategy A đúng → nhãn A; nếu A sai nhưng B đúng → nhãn B; nếu cả A và B sai → nhãn C.
4. **Xử lý conflict**: Khi nhiều strategy đều đúng, ưu tiên chiến lược đơn giản nhất (A > B > C) để bias classifier về phía hiệu quả.

### Luồng quyết định tại inference

```text
Input: query q
  │
  ▼
classifier f_θ(q) → {A, B, C}
  │
  ├─ A (No-Retrieval) ──► LLM(q) → answer
  │
  ├─ B (Single-Step) ───► Retrieve(q) → LLM(q, docs) → answer
  │
  └─ C (Multi-Step) ────► Loop:
                            docs_t = Retrieve(q, context_t)
                            context_{t+1} = LLM_reason(q, docs_t)
                            until termination → answer
```

### Vì sao chỉ ba category?

Ba category này tương ứng với ba mức tốn kém compute tăng dần, đồng thời bao phủ phần lớn phân phối độ phức tạp thực tế:
- A: O(1) LLM calls, 0 retrieval
- B: O(1) LLM calls, 1 retrieval
- C: O(k) LLM calls, O(k) retrievals, k ≥ 2

Thêm category thứ tư (ví dụ: graph-RAG) sẽ cần nhiều nhãn training hơn và tăng complexity của classification mà lợi ích biên giảm dần.

---

## 4. Thực nghiệm và Kết quả

### Datasets và setup

Paper đánh giá trên 5 dataset bao gồm cả single-hop và multi-hop:

| Dataset | Loại | Metric |
|---|---|---|
| PopQA | Single-hop, open-domain QA | Exact Match (EM) |
| WebQ (WebQuestions) | Single-hop, knowledge graph | Exact Match (EM) |
| TriviaQA | Single-hop, factoid | Exact Match (EM) |
| MuSiQue | Multi-hop (2-4 hop) | Exact Match (EM) |
| HotpotQA | Multi-hop (2 hop) | Exact Match (EM) |
| 2WikiMultiHopQA | Multi-hop, cross-document | Exact Match (EM) |

Generator: ChatGPT (gpt-3.5-turbo). Retriever: BM25 + Wikipedia corpus.

### Baselines

| Baseline | Mô tả |
|---|---|
| No-Retrieval | LLM trả lời trực tiếp, không retrieve |
| Standard RAG | Single-step retrieve-then-read |
| Self-RAG | Adaptive retrieval dựa trên reflection token |
| FLARE | Forward-looking active retrieval, trigger khi confidence thấp |
| IRCoT | Interleaved retrieval + chain-of-thought, multi-step |

### Kết quả chính (Exact Match %)

| Method | PopQA | WebQ | TriviaQA | MuSiQue | HotpotQA | 2WikiMHQA |
|---|---|---|---|---|---|---|
| No-Retrieval | 21.7 | 18.3 | 57.4 | 8.4 | 29.5 | 33.8 |
| Standard RAG | 50.6 | 40.6 | 57.6 | 8.3 | 35.6 | 34.7 |
| FLARE | 48.2 | 39.6 | 55.5 | 8.5 | 33.6 | 34.2 |
| IRCoT | 47.9 | 39.5 | 55.1 | 22.6 | 41.5 | 50.1 |
| Self-RAG | 54.9 | 43.8 | 59.2 | 15.3 | 38.2 | 42.0 |
| **Adaptive-RAG** | **56.1** | **46.4** | **60.2** | **23.5** | **43.5** | **51.2** |

*Số liệu từ Table 1, Jeong et al. (2024), arXiv:2403.14403. Một số giá trị có thể sai lệch nhỏ do rounding.*

**Nhận xét**: Adaptive-RAG đạt kết quả tốt nhất hoặc ngang nhất trên tất cả 6 dataset, đặc biệt trổi hơn trên single-hop (PopQA, WebQ) nhờ tránh overhead multi-step không cần thiết, và vẫn cạnh tranh trên multi-hop (MuSiQue, 2WikiMHQA) nhờ routing sang strategy C khi cần.

### Ablation Study

| Variant | PopQA | MuSiQue | Ghi chú |
|---|---|---|---|
| Always-A (No-Retrieval) | 21.7 | 8.4 | Thiếu knowledge |
| Always-B (Single-Step) | 50.6 | 8.3 | Không đủ cho multi-hop |
| Always-C (Multi-Step/IRCoT) | 47.9 | 22.6 | Overcomplicates single-hop |
| Adaptive w/ oracle labels | ~58 | ~25 | Upper bound với nhãn hoàn hảo |
| **Adaptive w/ auto labels** | **56.1** | **23.5** | Phương pháp đề xuất |

Kết quả ablation cho thấy: không có strategy đơn lẻ nào vượt được Adaptive trên cả hai loại dataset đồng thời. Gap giữa "oracle labels" và "auto labels" nhỏ (~2 điểm), chứng minh phương pháp tạo nhãn tự động khả thi.

### Hiệu quả (Efficiency)

So với Always-C (IRCoT), Adaptive-RAG giảm số lần retrieval trung bình khoảng **40-50%** trên tập dataset hỗn hợp, do nhiều single-hop query được route sang B hoặc A.

---

## 5. Phân tích Phê bình

### Giả định ẩn chưa được kiểm tra

**1. Complexity của query map gọn vào strategy.**
Classifier giả định độ phức tạp của câu hỏi đủ để xác định strategy tối ưu. Nhưng difficulty còn phụ thuộc vào corpus: cùng một query "What caused the 2008 financial crisis?" có thể cần multi-step nếu corpus thưa, nhưng single-step nếu corpus có bài tổng hợp sẵn. Classifier không thấy corpus khi prediction.

**2. Dataset bias làm proxy cho complexity.**
Nhãn tự động dựa trên bias dataset (PopQA = simple, MuSiQue = complex). Nếu trong production query distribution khác với distribution của các dataset training, classifier có thể bị miscalibrated — ví dụ: câu hỏi single-hop về domain niche (y học, pháp luật) bị classify là multi-hop vì surface form phức tạp.

**3. Strategy C là IRCoT — không phải mọi multi-step system.**
Paper dùng IRCoT làm đại diện cho strategy C. Kết quả phụ thuộc nhiều vào chất lượng của IRCoT implementation, không generalize dễ sang các multi-step system khác (ví dụ: ReAct, Self-RAG).

### Hạn chế paper thừa nhận

- Classifier cần re-train khi thay đổi retriever hoặc generator vì nhãn training phụ thuộc vào outcome của model cụ thể.
- Evaluation chỉ trên closed-set QA (Exact Match). Chưa đánh giá trên open-ended generation hoặc tasks ngoài QA.
- Corpus cố định (Wikipedia snapshot). Chưa test với corpus thay đổi theo thời gian.

### Hạn chế paper không đề cập

**Reproducibility của label creation**: Quy trình tạo nhãn yêu cầu chạy cả ba strategy trên toàn bộ training set — tốn kém compute và phụ thuộc vào kết quả của một LLM cụ thể tại một thời điểm. Nếu ChatGPT thay đổi behavior (model update), nhãn sẽ thay đổi và classifier cần re-train.

**Latency của classifier**: Paper báo cáo accuracy nhưng không report latency của bước classification. Trong production, nếu classifier là T5-large và inference trên CPU, overhead này không tầm thường.

**Distribution shift**: Khi corpus thay đổi (domain mới, documents mới), difficulty thực sự của query thay đổi, nhưng classifier không có cách biết — không có feedback loop từ retrieval outcome trở lại classifier.

---

## 6. Vị trí trong Landscape

### So sánh với các phương pháp adaptive/iterative khác

| Method | Trigger cơ chế | Granularity | Proactive/Reactive | Overhead |
|---|---|---|---|---|
| **Adaptive-RAG** | Learned classifier trên query | Query-level (3 class) | Proactive | Classifier inference |
| **Self-RAG** | Reflection tokens (special tokens) | Token-level | Reactive | Extra tokens per step |
| **FLARE** | Generation confidence | Sentence-level | Reactive | Confidence check |
| **IRCoT** | Fixed multi-step | Query-level (always multi) | N/A (always on) | O(k) calls |
| **ReAct** | Tool-use reasoning | Step-level | Reactive | Reasoning overhead |

### Khi nào nên dùng từng phương pháp?

- **Adaptive-RAG**: Dataset query có distribution phức tạp đa dạng (mix single/multi-hop), muốn optimize cả accuracy lẫn compute cost, có thể build training data từ existing datasets.
- **Self-RAG**: Muốn fine-tune generator để tự quyết định retrieve hay không, không muốn thêm separate classifier component.
- **FLARE**: Không có training data, muốn threshold đơn giản trên confidence.
- **IRCoT**: Toàn bộ dataset là multi-hop phức tạp, không quan tâm overhead.

---

## 7. Takeaway

1. **Routing là optimization, không phải magic**: Adaptive-RAG thắng không phải vì có strategy mới, mà vì tránh sử dụng strategy sai. Phần lớn gain đến từ không over-engineer các câu hỏi đơn giản.

2. **Tạo nhãn từ model outcome là double-edged sword**: Phương pháp tạo nhãn tự động thông minh và scalable, nhưng tạo ra strong coupling giữa classifier và model/strategy cụ thể. Khi stack thay đổi, classifier có thể trở nên stale mà không có cảnh báo rõ ràng.

3. **Ba category là pragmatic, không phải principled**: Lựa chọn {no-retrieval, single-step, multi-step} hợp lý cho Wikipedia QA benchmark, nhưng trong production với RAG phức tạp hơn (agentic search, tool-use, hybrid graph-dense retrieval), boundary giữa các category mờ đi và classifier có thể cần redesign.

---

## Nguồn

- Jeong, S., Baek, J., Cho, S., Hwang, S. J., & Park, J. C. (2024). *Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity*. NAACL 2024. https://arxiv.org/abs/2403.14403
- Trivedi et al. (2022). *Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions* (IRCoT). https://arxiv.org/abs/2212.10509
- Asai et al. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*. https://arxiv.org/abs/2310.11511
- Jiang et al. (2023). *Active Retrieval Augmented Generation* (FLARE). https://arxiv.org/abs/2305.06983
