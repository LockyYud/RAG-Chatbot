# `ircot_2022`: Xen Kẽ Retrieval Với Chain-of-Thought Cho Multi-Hop QA

> **Paper**: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions
> **Tác giả**: Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, Ashish Sabharwal
> **Venue**: ACL 2023 — arXiv:2212.10509
> **Loại**: `paper_inspired`

IRCoT giải quyết multi-hop question answering bằng cách dùng từng câu reasoning trung gian làm retrieval query, thay vì retrieve một lần từ câu hỏi gốc. Cơ chế này tạo ra vòng phản hồi hai chiều: reasoning dẫn dắt retrieval, và evidence mới tiếp tục dẫn dắt reasoning, giúp hệ thống tự khám phá các "bridge document" mà one-step retrieval không thể tìm thấy.

---

## 1. Bối cảnh và Động lực

**Multi-hop QA là gì và tại sao khó?** Các câu hỏi multi-hop yêu cầu kết nối nhiều đoạn evidence phân tán, ví dụ:

```text
"Đạo diễn của bộ phim X sinh ra ở thành phố nào?"
```

Để trả lời, hệ thống phải: (1) tìm tên đạo diễn của phim X, (2) dùng tên đó để tìm quê hương. Hai bước retrieval phụ thuộc tuần tự vào nhau — bước sau không thể thực hiện nếu không có kết quả bước trước.

**One-step retrieval thất bại như thế nào?** Khi retriever chỉ nhận câu hỏi gốc làm query, nó không thể biết "bridge entity" (entity trung gian) cần tìm kiếm là gì. Câu hỏi gốc không chứa đủ tín hiệu để retrieve document ở hop thứ hai. Kết quả là top-k documents trả về thường chỉ cover một phần của chuỗi reasoning.

**CoT-only (không có retrieval) thất bại như thế nào?** Các mô hình ngôn ngữ lớn (LLM) như GPT-3 có thể sinh Chain-of-Thought reasoning mạch lạc, nhưng với các câu hỏi đòi hỏi kiến thức cụ thể (tên người, ngày tháng, sự kiện lịch sử), LLM thường hallucinate — tạo ra các bridge entity không tồn tại hoặc sai thực tế, dẫn đến chuỗi suy luận nghe hợp lý nhưng hoàn toàn sai.

**Câu hỏi nghiên cứu:** Liệu có thể xen kẽ retrieval và reasoning để giải quyết cả hai vấn đề trên không? Tức là, reasoning cung cấp tín hiệu cho retrieval, và retrieved evidence giảm thiểu hallucination trong reasoning?

---

## 2. Đóng góp Chính

1. **Kiến trúc interleaved mới**: IRCoT đề xuất vòng lặp xen kẽ giữa CoT generation và retrieval, phá vỡ giả định phổ biến rằng retrieval chỉ xảy ra một lần trước khi reasoning bắt đầu.

2. **Cơ chế retrieval signal từ CoT**: Đóng góp kỹ thuật cốt lõi — câu CoT cuối cùng được sinh ra trong bước hiện tại trở thành query cho retrieval bước tiếp theo. Đây là cơ chế đơn giản nhưng hiệu quả, không yêu cầu huấn luyện thêm hay module riêng biệt.

3. **Bằng chứng thực nghiệm trên nhiều dataset**: Paper chứng minh IRCoT cải thiện cả retrieval recall và downstream QA F1 trên bốn dataset multi-hop — HotpotQA, 2WikiMultihopQA, MuSiQue, và IIRC — so với các baseline mạnh bao gồm CoT-only, one-step retrieval, và ReAct.

---

## 3. Phương pháp Chi tiết

### Thuật toán IRCoT

```
Input:  question Q, corpus D, max_steps T
Output: final answer A

C ← []                          # context = tập evidence đã thu thập
cot ← ""                        # chuỗi reasoning tích lũy

for step = 1 to T:
    # Bước 1: Sinh câu CoT tiếp theo
    sentence_t ← LLM(Q, C, cot)  # few-shot prompt với demonstration

    # Bước 2: Kiểm tra điều kiện dừng
    if sentence_t contains "answer is":
        A ← extract_answer(sentence_t)
        return A

    # Bước 3: Dùng câu CoT mới nhất làm retrieval query
    query_t ← sentence_t         # ← đây là cơ chế cốt lõi

    # Bước 4: Retrieve documents mới
    docs_t ← retrieve(query_t, D, top_k)

    # Bước 5: Cập nhật context và chuỗi reasoning
    C ← C ∪ docs_t
    cot ← cot + sentence_t

# Nếu chưa có answer sau T steps
A ← LLM(Q, C, cot, "answer:")
return A
```

### Cơ chế retrieval signal

Điểm mấu chốt của IRCoT là **câu CoT cuối cùng làm query**, không phải câu hỏi gốc. Ví dụ:

```text
Q: "Ai là tác giả của cuốn sách được chuyển thể thành bộ phim do Spielberg đạo diễn năm 1993?"

Step 1:
  CoT sentence: "Bộ phim Spielberg đạo diễn năm 1993 là Schindler's List."
  Query → "Schindler's List"
  Retrieved: [document về Schindler's List, nguồn gốc tiểu thuyết]

Step 2:
  CoT sentence: "Schindler's List dựa trên tiểu thuyết Schindler's Ark của Thomas Keneally."
  Query → "Thomas Keneally Schindler's Ark"
  Retrieved: [document về Thomas Keneally]

Step 3:
  CoT sentence: "Vậy tác giả là Thomas Keneally."  ← "answer is" detected → dừng
```

### Tại sao cơ chế này hoạt động

Sau mỗi bước reasoning, câu CoT vừa sinh ra chứa đựng đúng cái mà hệ thống "biết mình chưa biết" — bridge entity vừa được xác định. Câu đó mang nhiều thông tin hơn câu hỏi gốc vì nó đã được ground bởi evidence ở bước trước.

### Điều kiện dừng

- Câu CoT chứa cụm "So the answer is" hoặc tương đương.
- Đã đạt `max_steps` (thường T = 4–8 trong paper).
- Không có document mới nào được retrieve (no new evidence).

### Khác biệt với ReAct

| Khía cạnh | IRCoT | ReAct |
|-----------|-------|-------|
| Format | Free-form CoT sentences | Explicit Thought/Action/Observation |
| Scope | Tập trung vào retrieval cho QA | General tool use |
| Query source | Câu CoT cuối cùng | Explicit "Search[query]" action |
| Stopping | Dựa trên "answer is" trong CoT | Explicit "Finish[answer]" action |

### Few-shot prompting

IRCoT dùng few-shot demonstrations được xây dựng thủ công cho mỗi dataset. Mỗi demonstration gồm: câu hỏi, các retrieved paragraphs xen kẽ với CoT sentences tương ứng, và final answer. Không có fine-tuning; toàn bộ là prompting với frozen LLM (GPT-3 `text-davinci-003` trong paper gốc).

---

## 4. Thực nghiệm và Kết quả

### Dataset và setup

- **HotpotQA** (fullwiki): 2-hop, Wikipedia corpus
- **2WikiMultihopQA**: 2-hop với bridge và comparison questions
- **MuSiQue**: 2-4 hop, khó hơn và ít data leakage hơn HotpotQA
- **IIRC**: câu hỏi cần đọc nhiều article liên quan
- **Retriever**: BM25 (TF-IDF-based), top-k = 2 mỗi bước
- **LLM**: GPT-3 (`text-davinci-003`) với few-shot prompting

### Kết quả chính: Answer F1 (%)

| Method | HotpotQA | 2WikiMultihopQA | MuSiQue | IIRC |
|--------|----------|-----------------|---------|------|
| CoT-only (no retrieval) | 28.9 | 19.4 | 8.4 | 21.2 |
| One-step Retrieve + Read | 38.5 | 29.4 | 12.5 | 27.3 |
| ReAct | 35.1 | 27.4 | 11.1 | — |
| **IRCoT (BM25)** | **49.3** | **61.1** | **26.0** | **46.5** |

*Nguồn: Bảng 2, Trivedi et al. 2023. Các giá trị là Answer F1 trên test set.*

IRCoT cải thiện so với one-step retrieval: **+10.8** trên HotpotQA, **+31.7** trên 2WikiMultihopQA, **+13.5** trên MuSiQue, **+19.2** trên IIRC.

### Kết quả Retrieval Recall (Paragraph F1)

IRCoT cũng cải thiện đáng kể retrieval recall — không chỉ answer quality:

| Method | HotpotQA | 2WikiMultihopQA | MuSiQue |
|--------|----------|-----------------|---------|
| One-step BM25 | 46.8 | 31.6 | 19.4 |
| **IRCoT BM25** | **60.7** | **75.3** | **39.0** |

### Ablation study

**Ablation 1: Dùng câu hỏi gốc thay vì câu CoT làm query**

Thay `query_t ← sentence_t` bằng `query_t ← Q` (tức là retrieve nhiều lần nhưng luôn dùng câu hỏi gốc):

| Dataset | IRCoT (CoT sentence query) | Ablation (Q as query) | Delta |
|---------|---------------------------|----------------------|-------|
| HotpotQA | 49.3 | 41.2 | −8.1 |
| 2WikiMultihopQA | 61.1 | 38.7 | −22.4 |
| MuSiQue | 26.0 | 17.3 | −8.7 |

Kết quả cho thấy chính cơ chế dùng CoT sentence làm query — không chỉ đơn thuần là retrieve nhiều lần — tạo ra phần lớn cải thiện.

**Ablation 2: Số lượng retrieval steps**

IRCoT thường sử dụng 2–4 retrieval steps trên HotpotQA (2-hop questions). Trên MuSiQue (3-4 hop), cần trung bình 3–5 steps. Paper cho thấy performance tăng dần theo số steps và plateau sau khoảng T = 4–6 đối với đa số câu hỏi.

**Số retrieval steps trung bình thực tế:**
- HotpotQA: ~2.1 steps
- 2WikiMultihopQA: ~2.3 steps
- MuSiQue: ~3.2 steps
- IIRC: ~2.8 steps

---

## 5. Phân tích Phê bình

**Giả định ngầm:** IRCoT giả định rằng BM25 hoặc dense retriever có thể tìm thấy bridge documents khi được cung cấp câu CoT vừa sinh ra. Điều này đúng khi câu CoT chứa entity name rõ ràng, nhưng thất bại khi reasoning sinh ra câu mô tả trừu tượng hoặc entity bị viết sai chính tả.

**Chi phí cao:** Mỗi câu hỏi cần T lần gọi LLM thay vì 1 lần. Với T = 4 và latency ~1–2s mỗi lần, tổng latency có thể lên đến 4–8 giây mỗi câu hỏi — không phù hợp cho low-latency applications.

**Khi nào thất bại:** Nếu bước CoT đầu tiên đi sai hướng (hallucinate bridge entity), các bước tiếp theo sẽ retrieve documents không liên quan, và toàn bộ chain bị lệch. IRCoT không có cơ chế backtrack.

**Giới hạn về độ dài chain:** Paper chủ yếu đánh giá trên 2–3 hop questions. Hiệu quả trên các câu hỏi yêu cầu 5+ hop chưa được khảo sát rõ ràng. Với chain dài, context window của LLM có thể bị overflow do evidence tích lũy.

**Phụ thuộc vào chất lượng few-shot demonstrations:** Demonstrations được xây dựng thủ công cho từng dataset. Chất lượng demonstrations ảnh hưởng lớn đến khả năng model biết khi nào nên dừng và câu CoT nào hữu ích làm query.

---

## 6. Vị trí trong Landscape

### So sánh các phương pháp multi-hop retrieval

| Method | Retrieval signal | Multi-hop? | Tool-agnostic? | Training needed? |
|--------|-----------------|-----------|----------------|-----------------|
| Standard RAG | Original question | Không | Có | Không |
| Self-Ask | Sub-question explicit | Có | Không | Không |
| **IRCoT** | **Last CoT sentence** | **Có** | **Có** | **Không** |
| ReAct | Explicit Search action | Có | Không | Không |
| FLARE | Low-confidence spans | Có (implicit) | Có | Không |
| Iter-RetGen | Generated answer | Có | Có | Không |

**IRCoT vs ReAct:** ReAct yêu cầu model học format Thought/Action/Observation một cách tường minh; IRCoT chỉ cần CoT tự nhiên. IRCoT đơn giản hơn để implement nhưng kém linh hoạt hơn với diverse tools.

**IRCoT vs FLARE:** FLARE trigger retrieval khi model không chắc chắn (low token probability), còn IRCoT trigger sau mỗi câu CoT. IRCoT dễ kiểm soát hơn nhưng có thể retrieve không cần thiết.

**Khi nào nên dùng IRCoT:**
- Câu hỏi rõ ràng là multi-hop (có bridge entity).
- Corpus đủ lớn để one-step retrieval không đủ.
- Latency không phải constraint cứng.
- Không cần diverse tool use.

**Khi nào IRCoT là overkill:**
- Single-hop QA hoặc factoid lookup đơn giản.
- Câu hỏi đã được answer bởi một document duy nhất.
- Real-time applications với latency < 500ms.

---

## 7. Takeaway

1. **CoT sentence là retrieval signal tốt hơn original question cho multi-hop QA.** Câu reasoning trung gian "biết" mình cần thêm thông tin gì — đây là insight cốt lõi và đơn giản nhất của paper. Ablation xác nhận rằng chính cơ chế này (không chỉ multi-step retrieval nói chung) tạo ra phần lớn cải thiện.

2. **Retrieval và reasoning không phải là hai giai đoạn tách biệt.** IRCoT chứng minh rằng interleaving hai quá trình này — để chúng "thông báo" cho nhau liên tục — tốt hơn hẳn pipeline tuyến tính retrieve-then-read hoặc reasoning thuần túy không có grounding. Kiến trúc này là mô hình tư duy tốt cho bất kỳ task nào cần thu thập thông tin theo từng bước.

3. **Few-shot prompting với frozen LLM đủ để implement IRCoT.** Không cần fine-tuning hay training thêm. Điều này có nghĩa là IRCoT có thể được kết hợp với bất kỳ LLM nào đủ mạnh, và phần engineering chủ yếu nằm ở orchestration loop và retrieval backend — không phải model training.

---

## Nguồn

- Trivedi, H., Balasubramanian, N., Khot, T., & Sabharwal, A. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In *Proceedings of ACL 2023*. arXiv:2212.10509. https://arxiv.org/abs/2212.10509
- Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.
- Ho, X., et al. (2020). Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. COLING 2020. (2WikiMultihopQA)
- Trivedi, H., et al. (2022). MuSiQue: Multihop Questions via Single-hop Question Composition. TACL 2022.
- Yao, S., et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.
- Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE). arXiv:2305.06983.
