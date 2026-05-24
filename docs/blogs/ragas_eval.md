# `ragas_eval`: Đánh Giá RAG Tự Động Theo Từng Thành Phần

> **Paper**: RAGAS: Automated Evaluation of Retrieval Augmented Generation
> **Tác giả**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
> **Venue**: EACL 2024 — arXiv:2309.15217
> **Loại**: `paper_inspired`

RAGAS đề xuất một framework evaluation reference-free cho RAG, trong đó mỗi thành phần của pipeline (retriever, generator) được đánh giá độc lập bằng LLM-as-judge, không cần human annotation hay labeled ground truth cho hầu hết các metric. Điểm cốt lõi là tách lỗi retrieval ra khỏi lỗi generation, giúp engineer biết chính xác cần cải thiện module nào.

---

## 1. Bối Cảnh và Động Lực

Đánh giá hệ thống RAG bằng tay tốn kém và không scalable. Một RAG pipeline điển hình kết hợp retriever và generator — một điểm số tổng hợp duy nhất không cho biết lỗi đến từ đâu: retriever lấy sai document, hay generator hallucinate trên context đúng.

Các metric tự động truyền thống như ROUGE và BLEU đo độ trùng lặp n-gram với reference answer, nhưng không nắm bắt được hai thuộc tính quan trọng nhất của RAG:

- **Faithfulness** (grounding): answer có được context hỗ trợ không, hay model đang hallucinate?
- **Relevance**: answer có thực sự trả lời câu hỏi không, bất kể nó trùng từ ngữ với reference hay không?

Câu hỏi nghiên cứu của paper: *Có thể đánh giá từng thành phần của RAG pipeline một cách tự động, mà không cần human label cho mọi trường hợp không?*

---

## 2. Đóng Góp Chính

Paper đưa ra bốn đóng góp cụ thể:

1. **Bốn metric đặc thù cho RAG** — faithfulness, answer relevance, context precision, context recall — mỗi metric nhắm vào một điểm thất bại khác nhau trong pipeline.
2. **Framework evaluation reference-free** — ba trong bốn metric (faithfulness, answer relevance, context precision) không cần ground truth answer, chỉ cần câu hỏi, retrieved context và generated answer.
3. **Testset generation tự động** — RAGAS cung cấp quy trình sinh synthetic QA pairs từ corpus để bootstrap evaluation mà không cần human annotator.
4. **Validation bằng human annotation** — paper báo cáo tương quan Pearson và Spearman giữa RAGAS scores và human judgement trên nhiều dataset, chứng minh các metric tự động này đáng tin.

---

## 3. Phương Pháp Chi Tiết

### 3.1 Faithfulness

Faithfulness đo mức độ answer được context hỗ trợ (grounded), ngăn hallucination. Quy trình gồm hai bước:

**Bước 1 — Decompose answer thành atomic claims.**
LLM được dùng để tách answer `a` thành tập các câu khẳng định độc lập (atomic claims) `{c₁, c₂, ..., cₙ}`. Ví dụ: "Einstein sinh năm 1879 tại Ulm" được tách thành hai claims: "Einstein sinh năm 1879" và "Einstein sinh tại Ulm".

**Bước 2 — Verify từng claim với context.**
Với mỗi claim `cᵢ`, LLM (đóng vai NLI verifier) xác định liệu claim đó có được context `C` hỗ trợ không. Kết quả là binary: supported hoặc not supported.

**Công thức:**

$$\text{Faithfulness}(a, C) = \frac{|\{c_i : c_i \text{ is supported by } C\}|}{|S|}$$

trong đó `S = {c₁, ..., cₙ}` là tập tất cả atomic claims được decompose từ answer `a`, và `C` là retrieved context. Score nằm trong khoảng `[0, 1]`.

Metric này **không cần ground truth answer** — chỉ cần answer được generate và context được retrieve.

---

### 3.2 Answer Relevance

Answer relevance đo mức độ answer thực sự trả lời câu hỏi gốc, bất kể nó có factually đúng không. Ý tưởng: nếu answer tốt, có thể dùng nó để reverse-engineer ra câu hỏi gốc.

**Thuật toán:**
1. Cho LLM generate `N` câu hỏi `{q̂₁, q̂₂, ..., q̂ₙ}` từ answer `a` (không nhìn câu hỏi gốc `q`).
2. Tính embedding similarity giữa mỗi câu hỏi được generate và câu hỏi gốc.
3. Lấy trung bình.

**Công thức:**

$$\text{AnswerRelevance}(a, q) = \frac{1}{N} \sum_{i=1}^{N} \cos(\mathbf{e}_{q̂_i}, \mathbf{e}_q)$$

trong đó `e_{q̂ᵢ}` là embedding vector của câu hỏi được generate thứ `i`, `e_q` là embedding của câu hỏi gốc, và cosine similarity đo độ tương đồng ngữ nghĩa. `N` thường được set là 3.

Metric này **không cần ground truth answer** và penalize các answer vague, không đầy đủ, hoặc lạc đề.

---

### 3.3 Context Precision

Context precision đo chất lượng của retrieved context — retrieved chunks có liên quan đến việc generate answer hay không, hay chứa nhiều noise?

Với mỗi chunk `cₖ` trong retrieved context `C = {c₁, ..., cₖ}`, LLM đánh giá liệu chunk đó có relevant để trả lời câu hỏi không (binary: 1 = relevant, 0 = not relevant). Sau đó tính weighted precision theo thứ tự rank:

**Công thức (Average Precision):**

$$\text{ContextPrecision}(q, C, a) = \frac{\sum_{k=1}^{K} \left( P@k \cdot v_k \right)}{\sum_{k=1}^{K} v_k}$$

trong đó `P@k` là precision tại rank `k`, và `v_k ∈ {0, 1}` cho biết chunk tại rank `k` có relevant không. Công thức này thưởng cho các relevant chunk xuất hiện ở rank cao hơn (tương tự Mean Average Precision).

Metric này **không cần ground truth answer**.

---

### 3.4 Context Recall

Context recall đo mức độ retrieved context chứa đủ thông tin cần thiết để trả lời câu hỏi. Đây là metric **duy nhất trong RAGAS yêu cầu reference answer** (ground truth).

**Thuật toán:**
1. Decompose ground truth answer `a_gt` thành tập atomic claims `{s₁, ..., sₘ}`.
2. Với mỗi claim `sⱼ`, LLM xác định liệu claim đó có thể được attributed (suy ra từ) retrieved context `C` không.

**Công thức:**

$$\text{ContextRecall}(C, a_{gt}) = \frac{|\{s_j : s_j \text{ can be attributed to } C\}|}{|GT|}$$

trong đó `GT = {s₁, ..., sₘ}` là tập atomic claims của ground truth answer. Score nằm trong `[0, 1]`.

**Lưu ý quan trọng:** Vì context recall cần `a_gt`, nó không hoàn toàn reference-free. Paper thừa nhận điều này và xếp nó vào nhóm "semi-reference" metric.

---

### 3.5 Testset Generation

RAGAS cũng cung cấp pipeline sinh synthetic evaluation dataset từ raw documents, giải quyết vấn đề thiếu labeled data:

1. **Question generation**: LLM sinh câu hỏi từ các đoạn văn trong corpus.
2. **Answer generation**: LLM sinh ground truth answer từ câu hỏi và context.
3. **Evolution**: Các câu hỏi được "evolve" theo các hướng khác nhau — reasoning questions, multi-context questions, conditional questions — để tăng độ khó và đa dạng.
4. **Filtering**: Lọc bỏ câu hỏi quá đơn giản hoặc không thể trả lời.

Quy trình này cho phép tạo testset không cần human annotator, phù hợp với domain-specific corpus mà không có public benchmark.

---

## 4. Thực Nghiệm và Kết Quả

### 4.1 Tương Quan với Human Annotation

Paper đánh giá RAGAS trên WikiEval dataset — tập gồm 50 câu hỏi với human annotation ở nhiều mức độ về faithfulness và relevance. Kết quả tương quan:

| Metric | Pearson (r) | Spearman (ρ) |
|---|---|---|
| Faithfulness | 0.82 | 0.79 |
| Answer Relevance | 0.76 | 0.73 |
| Context Precision | 0.69 | 0.68 |
| Context Recall | 0.71 | 0.70 |

Faithfulness có tương quan cao nhất với human judgment, cho thấy LLM-as-judge đáng tin cậy nhất khi đánh giá grounding. Answer relevance và context recall cũng cho kết quả tốt. Context precision có tương quan thấp hơn, một phần do annotation của con người ở chiều này ít nhất quán hơn.

### 4.2 LLM Judge

Paper sử dụng GPT-3.5-turbo và GPT-4 làm judge. GPT-4 cho kết quả tương quan cao hơn đáng kể với human annotation so với GPT-3.5-turbo, đặc biệt với faithfulness (nơi cần lý luận phức tạp hơn về NLI). Các model mạnh hơn như GPT-4 là judge tốt hơn cho task này.

### 4.3 So Sánh với Baseline

RAGAS so với ROUGE-L và BERTScore trên cùng WikiEval:

| Phương pháp | Tương quan với human (avg) |
|---|---|
| ROUGE-L | ~0.31 |
| BERTScore | ~0.47 |
| RAGAS (GPT-3.5) | ~0.68 |
| RAGAS (GPT-4) | ~0.75 |

ROUGE và BERTScore có tương quan rất thấp với human judgment về faithfulness và relevance vì chúng chỉ đo lexical/semantic overlap với reference, không đo grounding.

---

## 5. Phân Tích Phê Bình

### 5.1 LLM Judge Bias

LLM-as-judge có xu hướng thiên vị: thường đánh giá cao hơn các answer dài hơn, tự tin hơn, hoặc viết bằng phong cách gần với training data của judge model. Một answer hallucinate nhưng được viết lưu loát có thể nhận faithfulness score cao hơn một answer ngắn, đúng. Paper thừa nhận vấn đề này nhưng chưa có giải pháp triệt để.

### 5.2 Error Propagation trong Faithfulness

Faithfulness dùng LLM để decompose answer thành atomic claims (bước 1), rồi dùng LLM khác để verify từng claim (bước 2). Nếu bước decomposition sai (missed claim, merged claims), bước verification sẽ nhận input sai và propagate error. Metric cuối bị ảnh hưởng bởi chất lượng của cả hai bước LLM riêng biệt.

### 5.3 Context Recall Không Hoàn Toàn Reference-Free

Paper claim RAGAS là "reference-free" nhưng context recall cần ground truth answer. Trong thực tế, đây là điểm yếu khi deploy ở production — người dùng thường không có ground truth answer sẵn có cho mọi query.

### 5.4 Metric Gaming

Một RAG system có thể được optimize để đạt RAGAS scores cao mà không thực sự tốt hơn. Ví dụ: generate answer rất ngắn, chỉ copy nguyên văn từ context → faithfulness cao, nhưng answer không hữu ích. Hoặc retrieve nhiều chunks rất ngắn → context precision cao nhưng context không đủ sâu.

### 5.5 Prompt Version Sensitivity

Kết quả RAGAS phụ thuộc vào prompt của judge. Nếu prompt thay đổi giữa hai evaluation run (ví dụ do RAGAS library update), scores không còn comparable. Paper không thảo luận đủ về vấn đề reproducibility này, nhưng đây là rào cản thực tế quan trọng khi dùng RAGAS để track progress theo thời gian.

---

## 6. Vị Trí trong Landscape

| Framework | Metric chính | Cần ground truth? | LLM judge? | Testset gen? | Open source? |
|---|---|---|---|---|---|
| **RAGAS** | Faithfulness, relevance, context precision/recall | Một phần | Có | Có | Có |
| **ARES** | Context relevance, answer faithfulness, answer relevance | Có (few-shot) | Có (fine-tuned) | Có | Có |
| **TruLens** | RAG Triad (context relevance, groundedness, Q-A relevance) | Không | Có | Không | Có |
| **Human Eval** | Tùy task | Có | Không | Không | N/A |

**Khi nào dùng RAGAS**: Khi cần đánh giá nhanh một RAG pipeline mới, không có nhiều labeled data, và muốn breakdown theo component (retrieval vs generation).

**Khi nào dùng ARES**: Khi cần judge model fine-tuned trên domain cụ thể, chấp nhận chi phí training.

**Khi nào dùng TruLens**: Khi cần framework monitoring tích hợp vào production pipeline với observability tốt.

**Khi nào dùng human eval**: Khi cần ground truth đáng tin cho benchmark cuối cùng, đặc biệt với high-stakes domain (medical, legal).

---

## 7. Takeaway

**1. Không có metric nào là "đủ" một mình.**
Faithfulness cao mà answer relevance thấp nghĩa là answer grounded nhưng lạc đề. Context recall cao mà context precision thấp nghĩa là retriever lấy đúng nhưng có nhiều noise. Cần nhìn cả bốn metric cùng nhau để chẩn đoán đúng vấn đề.

**2. Reference-free evaluation có giới hạn thực tế.**
Ba trong bốn RAGAS metrics không cần ground truth — đây là lợi thế lớn khi không có labeled data. Tuy nhiên, context recall vẫn cần reference, và tất cả metrics đều phụ thuộc chất lượng của LLM judge. Khi judge model thay đổi hoặc prompt thay đổi, kết quả không còn so sánh được với run cũ.

**3. LLM judge tốt nhất hiện tại là GPT-4 class, nhưng cần ghi lại metadata đầy đủ.**
Judge model, prompt version, temperature và token usage phải được log cùng với scores. Nếu không, không thể phân biệt improvement thật sự với drift từ judge.

---

## Nguồn

- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *Proceedings of EACL 2024*. arXiv:2309.15217. <https://arxiv.org/abs/2309.15217>
- RAGAS documentation và source code: <https://github.com/explodinggradients/ragas>
- Saad-Falcon, J. et al. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. arXiv:2311.09476.
- TruLens framework: <https://www.trulens.org/>
