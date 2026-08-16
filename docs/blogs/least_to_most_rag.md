# `least_to_most_rag`: Decompose Trước, Giải Sau — Từ Đơn Giản Đến Phức Tạp

> **Paper**: Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
> **Tác giả**: Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi (Google Research)
> **Venue**: ICLR 2023 — arXiv:2205.10625
> **Loại**: `paper_inspired`

Least-to-Most (LtM) Prompting giải quyết điểm yếu của Chain-of-Thought: CoT giỏi trong sequential reasoning nhưng kém với bài toán yêu cầu systematic decomposition — khi câu trả lời của sub-problem A là prerequisite cho sub-problem B. LtM tách rời hai giai đoạn: (1) explicit decomposition thành sub-problems từ dễ đến khó, (2) sequential solving với sub-answer làm context tích lũy. Trong RAG context, pattern này map tự nhiên thành: decompose query → retrieve-and-answer cho từng sub-query → synthesize final answer.

> **Lưu ý:** Paper gốc không phải về RAG. Đây là prompting strategy paper. Phần 3.3 của blog này phân tích cách apply LtM vào RAG — đây là phân tích của người viết blog, không phải claim của paper.

---

## 1. Bối cảnh và Động lực

### CoT và giới hạn của nó

Chain-of-Thought prompting (Wei et al., 2022) đã chứng minh hiệu quả với nhiều reasoning tasks. Nhưng CoT có một điểm mù cụ thể: nó giả định LLM có thể implicit reason through all necessary steps. Với tasks yêu cầu **systematic generalization** — apply rules đã học sang cases phức tạp hơn — CoT thường fail.

**Ví dụ điển hình — SCAN task:**
SCAN yêu cầu map natural language commands sang action sequences. Training set chỉ chứa short commands; test set chứa long compositional commands.

- Training: "jump" → JUMP; "jump twice" → JUMP JUMP; "jump left" → LTURN JUMP
- Test: "jump around left twice" → LTURN JUMP LTURN JUMP LTURN JUMP LTURN JUMP

CoT: "Okay, 'jump around left twice' means jump around left, done twice. So..." → thường fail vì không explicitly decompose.

LtM: "To solve 'jump around left twice', what do I need to solve first?
1. What does 'jump left' mean? → LTURN JUMP
2. What does 'jump around left' mean? → LTURN JUMP LTURN JUMP LTURN JUMP LTURN JUMP
3. What does 'jump around left twice' mean? → [above] × 2" → đúng.

### Insight cốt lõi

LtM tách rời hai bài toán con mà CoT cố làm đồng thời:
1. **Decomposition**: Xác định sub-problems theo dependency order
2. **Sequential solving**: Solve từng sub-problem, dùng kết quả của sub-problem trước làm input cho sub-problem sau

Bài toán compositional generalization trở nên dễ hơn khi được tách rõ ràng như vậy.

---

## 2. Đóng góp Chính

**Contribution 1 — Two-stage prompting framework**: Giai đoạn 1 — prompt LLM để decompose; Giai đoạn 2 — prompt LLM để solve từng sub-problem sequentially với accumulated context.

**Contribution 2 — Empirical demonstration on compositional tasks**: Chứng minh CoT fail (16% accuracy) trong khi LtM near-perfect (99.7%) trên SCAN — gap 83 percentage points cho thấy tầm quan trọng của explicit decomposition.

**Contribution 3 — Generalization sang arithmetic và NLP**: LtM không chỉ apply cho SCAN (symbolic) mà còn improve trên DROP (numerical reasoning) và last-letter concatenation — cho thấy principle có thể generalize.

---

## 3. Phương pháp Chi tiết

### 3.1 Stage 1: Decomposition Prompting

LLM được prompt để answer câu hỏi: *"Để giải bài toán X, cần giải những bài toán nào trước?"*

**Few-shot examples cho decomposition stage:**
```
Q: "jump around left twice"
A: Before solving "jump around left twice", I need to:
   1. Solve: "jump left"
   2. Solve: "jump around left"  
   3. Then solve: "jump around left twice"

Q: "What is the last letter of the word 'thinking' concatenated 
    with the last letter of 'machine'?"
A: Before solving this, I need to:
   1. Solve: "What is the last letter of 'thinking'?"
   2. Solve: "What is the last letter of 'machine'?"
   3. Then concatenate the two results.
```

Output của stage 1: ordered list of sub-problems.

### 3.2 Stage 2: Sequential Solving

Với sub-problem list từ stage 1, LLM giải từng sub-problem theo thứ tự. Quan trọng: **câu trả lời của sub-problem trước được include trong context** của sub-problem tiếp theo.

```
Sub-problem 1: "What is the last letter of 'thinking'?"
Context: (empty)
Answer: "g"

Sub-problem 2: "What is the last letter of 'machine'?"
Context: "The last letter of 'thinking' is 'g'."
Answer: "e"

Sub-problem 3: "Concatenate 'g' and 'e'."
Context: "The last letter of 'thinking' is 'g'. The last letter of 'machine' is 'e'."
Answer: "ge"
```

**Key design choice:** Mỗi sub-problem được giải in sequence, không in parallel. Nếu sub-problem B phụ thuộc vào kết quả của A, B phải chờ A xong. Đây là trade-off với speed nhưng đảm bảo dependency được handled đúng.

### 3.3 Áp Dụng LtM vào RAG (Phân tích của tác giả blog)

*Phần này là phân tích về cách extend LtM cho RAG, không phải claim của paper gốc.*

Pattern của LtM map rất tự nhiên vào RAG multi-hop scenario:

**LtM-RAG Pipeline:**
```
Stage 1 — Decompose query:
  Input: complex question Q
  LLM → ["sub-question 1", "sub-question 2", ..., "sub-question n"]
  (theo dependency order: sq1 trả lời trước thì mới trả lời được sq2)

Stage 2 — Sequential retrieve-and-answer:
  For each sub-question sq_i:
    docs_i = Retriever.retrieve(sq_i + accumulated_context)
    answer_i = LLM(sq_i, docs_i, previous_answers)
    accumulated_context += f"Q: {sq_i}\nA: {answer_i}\n"

Stage 3 — Final synthesis:
  Final answer = LLM(original_Q, accumulated_context)
```

**Ví dụ với multi-hop factual query:**
```
Original query: "Người thành lập công ty đã mua lại Instagram thuộc quốc tịch gì?"

Stage 1 — Decompose:
  1. "Công ty nào đã mua lại Instagram?"
  2. "Ai là người thành lập công ty đó?"
  3. "Người đó có quốc tịch gì?"

Stage 2 — Sequential retrieve-and-answer:
  sq1: Retrieve về "Instagram acquisition" → "Facebook mua Instagram năm 2012"
  sq2: Retrieve về "Facebook founder" + "Facebook acquired Instagram" → "Mark Zuckerberg"
  sq3: Retrieve về "Mark Zuckerberg nationality" → "Người Mỹ (American)"
  
Final answer: "Người Mỹ"
```

**So sánh với IRCoT:**
- IRCoT: CoT reasoning trong-line với retrieval, không explicit decomposition stage
- LtM-RAG: explicit decomposition trước, từng sub-question retrieve independently
- LtM tốt hơn khi sub-questions truly independent (riêng biệt); IRCoT tốt hơn khi reasoning cần flow liên tục

---

## 4. Thực nghiệm và Kết quả

### 4.1 SCAN — Symbolic Compositional Generalization

SCAN (Simplified version of the CommAI Navigation) test khả năng generalize từ short commands sang long compositional commands.

| Method | Accuracy |
|--------|---------|
| Standard prompting | 6.9% |
| Chain-of-Thought (CoT) | 16.0% |
| **Least-to-Most** | **99.7%** |

Gap 83 điểm percentage giữa CoT và LtM là kết quả dramatic nhất trong paper — chứng tỏ explicit decomposition là critical cho systematic compositional generalization.

### 4.2 Last Letter Concatenation

Task: concatenate last letters của N từ. LtM benefit rõ ràng khi N tăng:

| N (số từ) | CoT | Least-to-Most |
|-----------|-----|--------------|
| 2 | 97.2% | 98.0% |
| 4 | 89.3% | 99.1% |
| 6 | 79.2% | **98.1%** |
| 8 | 65.1% | 97.3% |

CoT degrades rapidly với N lớn; LtM maintains near-perfect performance vì explicit sub-problem structure.

### 4.3 DROP — Discrete Reasoning over Paragraphs

DROP yêu cầu numeric reasoning (addition, comparison, sorting) trên text passages.

| Method | F1 |
|--------|-----|
| Standard few-shot | 48.9 |
| CoT | 52.7 |
| **Least-to-Most** | **62.4** |

### 4.4 GSM8K — Math Word Problems

| Method | Accuracy |
|--------|---------|
| Few-shot | 51.8% |
| CoT | 56.5% |
| **Least-to-Most** | **59.7%** |

Improvement nhỏ hơn nhiều trên GSM8K (3.2% vs 83% trên SCAN) vì GSM8K questions ít compositional hơn — sequential arithmetic không benefit nhiều từ explicit decomposition.

---

## 5. Phân tích Phê bình

**Decomposition quality phụ thuộc hoàn toàn vào LLM:** Nếu LLM decompose sai — tạo sub-problems không đúng order, bỏ sót sub-problem, hay tạo sub-problems không cần thiết — toàn bộ pipeline fail. Stage 1 không có verification mechanism. Với complex real-world questions, wrong decomposition có thể harder to detect than wrong final answer.

**Error propagation không có backtracking:** Nếu sub-answer A sai, sub-answer B sẽ được tính toán dựa trên A sai. Không có mechanism để go back và retry A nếu B suggests A was wrong. RAG với LtM là một-chiều — không recover được từ early mistakes.

**SCAN/SCAN-like tasks không representative của real-world RAG:** Kết quả dramatic trên SCAN (99.7% vs 16%) xảy ra vì SCAN là synthetic, rule-based task với cấu trúc decomposition rất rõ ràng. Real-world questions thường messier — sub-questions ambiguous, overlapping, hay không clearly ordered. LtM benefit có thể much smaller trên real QA tasks.

**Hai LLM calls minimum cho every query:** Stage 1 (decompose) + Stage 2 (solve) = ít nhất 2 LLM calls. Với N sub-problems, total = 1 + N calls. Với GPT-4, đây là significant cost increase so với single-call approaches. Paper không analyze cost efficiency.

**Không evaluate trên multi-document RAG directly:** Paper gốc không test LtM trong retrieval setting. Extension sang RAG là plausible nhưng unvalidated trong paper. Practical questions về integration (how to retrieve for each sub-query, how to handle conflicting sub-answers) chưa được addressed.

---

## 6. Vị trí trong Landscape

| Method | Decomposition | Sequential Solving | RAG Integration | Multi-hop Support |
|--------|--------------|-------------------|-----------------|------------------|
| Standard RAG | Không | N/A | Direct | Kém |
| CoT + RAG | Implicit | Trong-line | Trước retrieval | Khá |
| **LtM-RAG** | **Explicit** | **Sequential** | **Per sub-query** | **Tốt** |
| IRCoT | Implicit (CoT) | Interleaved | Per CoT step | Tốt |
| Plan-and-Solve | Explicit (plan) | Per plan step | Per plan step | Tốt |
| Least-to-Most + IRCoT | Explicit + implicit | Hybrid | Hybrid | Tốt nhất (complex) |

**Khi nào dùng LtM-RAG:**
- Câu hỏi có rõ ràng các sub-questions có dependency (multi-hop factual)
- Sub-questions có thể retrieve independently
- Có đủ LLM API budget cho multiple calls

**Không phù hợp khi:**
- Câu hỏi simple (single-hop) — overhead không đáng
- Câu hỏi không decomposable rõ ràng (analytical, opinion)
- Latency hay cost là constraint chính

---

## 7. Takeaway

**Explicit decomposition thắng implicit reasoning trên compositional tasks:** Gap 83% trên SCAN là evidence mạnh nhất. LtM không chỉ là prompt engineering — nó reflect một truth về reasoning: khi sub-problems có explicit dependencies, making those dependencies visible cho model improve performance dramatically.

**Sub-answer accumulation là context management pattern:** Giải sub-problem A, include answer trong context cho B, include cả A và B trong context cho C — đây là pattern context management hữu ích có thể apply rộng rãi, không chỉ trong LtM. Bất kỳ RAG system nào xử lý sequential dependent questions đều có thể benefit từ explicit accumulated context.

**Benefits diminish on less compositional tasks:** Trên GSM8K (arithmetic word problems), LtM chỉ cải thiện 3.2% so với CoT. Trên SCAN (symbolic composition), cải thiện 83%. Lesson: LtM worth the overhead khi tasks have clear compositional structure; otherwise simpler approaches đủ.

---

## Nguồn

- Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O., Le, Q., & Chi, E. (2023). Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. *ICLR 2023*. https://arxiv.org/abs/2205.10625
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. https://arxiv.org/abs/2201.11903
- Trivedi, H., et al. (2022). IRCoT: Interleaving Retrieval with Chain-of-Thought Reasoning. *ACL 2023*. https://arxiv.org/abs/2212.10509
- Wang, L., et al. (2023). Plan-and-Solve Prompting. *ACL 2023*. https://arxiv.org/abs/2305.04091
