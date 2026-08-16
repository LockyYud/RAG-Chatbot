# `self_rag_2023`: Reflection Tokens — Khi LLM Tự Quyết Định Cần Retrieve Gì

> **Paper**: Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
> **Tác giả**: Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
> **Venue**: ICLR 2024 — arXiv:2310.11511
> **Loại**: `paper_inspired`

Self-RAG không thêm external evaluator hay rule-based controller vào RAG pipeline — thay vào đó, LLM được fine-tune để sinh ra các *reflection tokens* ngay trong generation stream: quyết định khi nào cần retrieve, đánh giá relevance của passages vừa retrieved, kiểm tra grounding của output vừa sinh ra. Cách tích hợp retrieval decision-making trực tiếp vào generation này loại bỏ cần thiết phải có separate judge model như CRAG, nhưng yêu cầu fine-tuning LLM specifically cho task này.

---

## 1. Bối cảnh và Động lực

### RAG truyền thống: không có self-awareness

Standard RAG pipeline có một điểm mù: nó luôn retrieve, bất kể câu hỏi có cần retrieval không. Hỏi "What is 2+2?" → vẫn retrieve. Câu hỏi thực ra đơn giản và LLM biết câu trả lời từ parametric knowledge → retrieve documents gây noise hơn là giúp ích.

Ngược lại, Adaptive-RAG (Jeong et al., 2024) dùng một separate classifier để route queries. CRAG (Yan et al., 2024) dùng separate T5 evaluator để judge retrieval quality. Cả hai cách đều thêm một external module — overhead thêm và independent của generation process.

**Câu hỏi của Self-RAG:** Tại sao phải tách rời? LLM đã có đủ "kiến thức" về khi nào câu hỏi cần bằng chứng ngoài và khi nào không. Chỉ cần dạy nó thể hiện điều đó explicit trong output.

### Khái niệm reflection tokens

Reflection tokens là special tokens được insert vào generation stream, hoạt động như metadata:
- Trước segment: "Tôi có cần retrieve không?"
- Sau khi retrieve: "Passage này có relevant không?"
- Sau khi generate: "Output của tôi có được support bởi passage không?"
- Sau khi hoàn thành: "Response này có useful không?"

Bằng cách fine-tune LLM để sinh ra những tokens này, Self-RAG biến retrieval decision-making thành một phần của generation thay vì một module riêng biệt.

---

## 2. Đóng góp Chính

**Contribution 1 — Bốn loại reflection tokens**: Thiết kế tập token đặc biệt bao phủ toàn bộ lifecycle của RAG: retrieve decision, relevance judgment, support judgment, và utility scoring.

**Contribution 2 — GPT-4-assisted training data generation**: Pipeline offline dùng GPT-4 để annotate reflection tokens trên corpus lớn, tạo training data cho fine-tuning Llama.

**Contribution 3 — Segment-level beam search tại inference**: Thay vì generate toàn bộ answer rồi mới evaluate, Self-RAG generate từng segment, score bằng reflection tokens, rồi mới tiếp tục — cho phép beam search across generation paths.

---

## 3. Phương pháp Chi tiết

### 3.1 Bốn Loại Reflection Tokens

| Token | Khi nào sinh | Giá trị | Ý nghĩa |
|-------|-------------|---------|---------|
| `[Retrieve]` | Trước mỗi segment | `yes` / `no` / `continue` | Có cần retrieve thêm không? |
| `[IsREL]` | Sau retrieve, mỗi passage | `relevant` / `irrelevant` | Passage có liên quan đến query không? |
| `[IsSUP]` | Sau generate segment, mỗi passage | `fully supported` / `partially supported` / `not supported` | Segment có được support bởi passage không? |
| `[IsUSE]` | Cuối response | 1–5 | Response tổng thể có useful không? |

**Quan trọng:** Những tokens này không được inject từ external system — chính LLM sinh ra chúng như một phần của generation. Tương tự cách T5 có thể generate `<pad>`, LLM fine-tuned với Self-RAG vocabulary mới sẽ tự nhiên insert `[Retrieve]` tại đúng vị trí.

### 3.2 Inference Process

```
Input: instruction x (câu hỏi/task)
Initialize: context = ""

REPEAT for each generation segment:
  
  Step 1 — Retrieve decision:
  LLM generates [Retrieve] token
  
  IF [Retrieve] = no or continue:
    Generate next segment y_t WITHOUT retrieval
    context += y_t
    
  ELIF [Retrieve] = yes:
    
    Step 2 — Retrieve K passages:
    D = Retriever.retrieve(x + context, k=K)
    
    Step 3 — Generate K candidate segments:
    FOR each passage d_i in D:
      LLM generates:
        [IsREL](x, d_i)          # Relevance judgment
        y_t^i                    # Candidate segment given passage
        [IsSUP](y_t^i, d_i)      # Support judgment
      
    Step 4 — Select best segment:
    score(d_i) = w_rel × IsREL_score
               + w_sup × IsSUP_score
               + w_use × IsUSE_score (nếu final)
    y_t = y_t^{argmax score}
    context += y_t

UNTIL generation complete

Step 5 — Generate [IsUSE] for final response
```

**Điểm khác biệt với CRAG:** CRAG chạy T5 evaluator *trước* generation để quyết định action. Self-RAG *interleave* judgment với generation — quyết định được đưa ra trong-line với output.

### 3.3 Training Data Generation

Pipeline offline để tạo training data:

**Bước 1 — Annotation với GPT-4:**
GPT-4 được prompt để annotate reflection tokens cho các (instruction, passage, response) triples:
- "Với instruction này và passage này, segment này có fully supported không?" → generate `[IsSUP]` label
- "Với instruction này, có cần retrieval không?" → generate `[Retrieve]` label

**Bước 2 — Curate corpus:**
Thu thập diverse instruction-output pairs từ public datasets, thêm retrieval augmentation, annotate bằng GPT-4.

**Bước 3 — Fine-tune Llama:**
Train Llama-2 (7B và 13B) trên augmented corpus với reflection tokens như special vocabulary. Objective: standard language modeling loss, nhưng trên text kết hợp generation + reflection tokens.

Kết quả: model tự nhiên learns khi nào insert `[Retrieve] = yes` và cách assign đúng values cho `[IsREL]`, `[IsSUP]`, `[IsUSE]`.

### 3.4 Segment-Level Beam Search

Tại inference, Self-RAG không chỉ greedy decode. Với mỗi retrieved passage, có một *candidate path* (segment + judgments). Beam search chọn paths có highest cumulative score.

$$\text{score}(y, d) = \alpha \cdot \text{IsREL}(d) + \beta \cdot \text{IsSUP}(y, d) + \gamma \cdot \text{IsUSE}(y)$$

Weights $\alpha, \beta, \gamma$ có thể điều chỉnh tùy use case: ưu tiên factual grounding (tăng $\beta$) hay ưu tiên utility (tăng $\gamma$).

---

## 4. Thực nghiệm và Kết quả

### 4.1 Datasets

- **PopQA**: Open-domain factual QA. Metric: accuracy
- **PubHealth**: Health claim verification từ tin tức. Metric: accuracy
- **ASQA**: Long-form QA, nhiều facets. Metric: MAUVE, EM Recall
- **ARC-Challenge**: Multiple-choice science QA. Metric: accuracy
- **TriviaQA**: Open-domain factual QA. Metric: accuracy

### 4.2 Kết quả Chính

| Method | PopQA | PubHealth | ARC-Challenge | TriviaQA |
|--------|-------|-----------|---------------|----------|
| Llama2-7B (no retrieval) | 34.9 | 69.0 | 72.0 | 66.2 |
| Llama2-13B (no retrieval) | 46.1 | 67.4 | 78.2 | 72.4 |
| Naive RAG (Llama2-13B) | 50.5 | 70.6 | 72.1 | — |
| CRAG (Llama2-13B) | 56.4 | 73.0 | — | — |
| **Self-RAG-7B** | **54.9** | **72.4** | **73.7** | **68.2** |
| **Self-RAG-13B** | **56.4** | **72.4** | **76.7** | **73.5** |
| ChatGPT (no retrieval) | 50.4 | 72.4 | — | 84.7 |

Self-RAG-13B cạnh tranh với ChatGPT trên PopQA và PubHealth mặc dù model nhỏ hơn nhiều (13B vs ~175B).

### 4.3 Long-Form: ASQA

| Method | MAUVE ↑ | EM Recall ↑ | Citation Precision ↑ |
|--------|---------|------------|---------------------|
| Llama2-7B (no retrieval) | 21.6 | 26.8 | — |
| Naive RAG | 29.4 | 28.1 | 51.3 |
| **Self-RAG-7B** | **61.8** | **34.2** | **73.4** |
| GPT-3.5 turbo + RAG | 55.2 | 33.7 | 64.4 |

Self-RAG-7B significantly outperforms GPT-3.5 + RAG trên ASQA về cả MAUVE (fluency/coverage) và citation precision — chứng tỏ reflection tokens thực sự cải thiện grounding.

### 4.4 Ablation

| Variant | PopQA | Ghi chú |
|---------|-------|---------|
| Self-RAG đầy đủ | 56.4 | |
| Không có [Retrieve] (luôn retrieve) | 52.1 | −4.3 |
| Không có [IsSUP] | 54.2 | −2.2 |
| Không có [IsREL] | 55.8 | −0.6 |
| Không có adaptive retrieval (fixed K) | 53.7 | −2.7 |

`[Retrieve]` token quan trọng nhất — selective retrieval giúp ích nhiều. `[IsSUP]` là second most important — grounding check giảm hallucination.

---

## 5. Phân tích Phê bình

**Model phải được fine-tune specifically:** Đây là limitation lớn nhất so với CRAG hay Adaptive-RAG. Không thể apply Self-RAG mechanism vào GPT-4 hay bất kỳ black-box LLM nào. Mỗi base model cần được fine-tune lại với Self-RAG training data. Với tốc độ thay đổi models hiện tại (Llama 3, Gemma, Qwen...), maintenance overhead là đáng kể.

**Training data từ GPT-4 annotation có thể inherit GPT-4 biases:** GPT-4 labels cho reflection tokens không phải ground truth — chúng là GPT-4's judgment. Nếu GPT-4 có systematic biases về khi nào cần retrieve hay khi nào claim is supported, Self-RAG sẽ learn những biases đó.

**Self-assessment không phải external verification:** Khi `[IsSUP]` token đánh giá support, chính model đang tự đánh giá output của chính nó. Không có independent verification. Model có thể consistently overestimate support ("fully supported" cho claims mà thực ra only partially supported) vì confirmation bias trong generation.

**Segment boundaries là heuristic:** Khi nào kết thúc một "segment" và bắt đầu segment mới? Paper dùng sentence boundaries, nhưng đây là heuristic. Với output yêu cầu coherent multi-sentence reasoning, segment-level beam search có thể break logical flow.

**Inference chậm hơn đáng kể với K passages:** Với mỗi `[Retrieve] = yes`, cần K parallel generation paths. Ở beam search với beam size b và K passages, có thể có b × K paths active cùng lúc — memory và compute intensive.

---

## 6. Vị trí trong Landscape

| Method | Adaptive Retrieval | Retrieval Judge | LLM Fine-tune? | Inference Overhead |
|--------|------------------|-----------------|-----------------|--------------------|
| Naive RAG | Không | Không | Không | 1x |
| Adaptive-RAG | Có (classifier) | Separate model | Không | 1.2x |
| CRAG | Có (T5 evaluator) | Separate T5 | Không | 1.5x |
| **Self-RAG** | **Có (inline token)** | **Inline token** | **Có** | **K×** |
| FLARE | Có (probability threshold) | Probability-based | Không | Variable |

**Khi nào dùng Self-RAG:**
- Có thể fine-tune Llama-class model
- Cần citation accuracy cao (ASQA use case)
- Corpus yêu cầu selective retrieval (mix của parametric và retrieved knowledge)
- Long-form generation cần grounding

**Không phù hợp khi:**
- LLM là black-box hoặc không thể fine-tune
- Latency critical (inference K× đắt hơn)
- One-off deployment không justify fine-tuning cost

---

## 7. Takeaway

**Reflection tokens biến RAG từ pipeline thành capability:** Khi model có thể tự quyết định "tôi có cần retrieve không" và tự đánh giá "output của tôi có được support không", đây không còn là RAG như một external augmentation — đây là một loại knowledge access khác fundamentally. Model tích hợp retrieval như một tool nó chủ động lựa chọn dùng.

**Self-assessment là upper bound bởi model quality:** Khi `[IsSUP]` = "fully supported" nhưng thực ra không, lỗi không thể được detected. Self-RAG chỉ đáng tin cậy khi base LLM đủ mạnh để accurately assess grounding — một circular dependency. ARES-style external judge (Saad-Falcon et al.) là complementary: đánh giá independently.

**Fine-tuning cost là giá của inline intelligence:** CRAG và Adaptive-RAG không cần fine-tune LLM — họ add external modules. Self-RAG trade-off đó: tốn fine-tune cost để gain simpler inference graph và better integration. Không có lựa chọn nào "miễn phí" — chỉ là allocate cost ở đâu.

---

## Nguồn

- Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*. https://arxiv.org/abs/2310.11511
- Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). Corrective Retrieval Augmented Generation. *ICLR 2024*. https://arxiv.org/abs/2401.15884
- Jeong, S., et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. *NAACL 2024*. https://arxiv.org/abs/2403.14403
- Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE). *EMNLP 2023*. https://arxiv.org/abs/2305.06983
