# `plan_and_solve_rag`: Lập Kế Hoạch Trước Khi Giải — Zero-Shot Planning cho RAG

> **Paper**: Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models
> **Tác giả**: Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, Ee-Ping Lim
> **Venue**: ACL 2023 — arXiv:2305.04091
> **Loại**: `paper_inspired`

Plan-and-Solve (PS) Prompting cải thiện zero-shot CoT bằng cách thêm explicit planning instruction: thay vì chỉ "let's think step by step", PS yêu cầu LLM *trước tiên* devise a plan, *sau đó* carry out the plan. PS+ — phiên bản tăng cường — yêu cầu extract relevant variables và pay attention to calculation accuracy. Điều này giảm hai loại lỗi phổ biến của zero-shot CoT: calculation errors và missing-step errors, đạt gains đáng kể trên math reasoning benchmarks mà không cần few-shot examples.

> **Lưu ý:** Paper gốc tập trung vào mathematical reasoning, không phải RAG. Phần 3.3 của blog phân tích extension sang RAG — đây là nhận định của tác giả blog.

---

## 1. Bối cảnh và Động lực

### Zero-shot CoT và các lỗi thường gặp

Kojima et al. (2022) chứng minh rằng chỉ cần thêm "Let's think step by step" vào prompt đã cải thiện LLM reasoning đáng kể trên nhiều tasks — surprising vì hoàn toàn zero-shot (không cần examples). Nhưng phân tích lỗi cho thấy zero-shot CoT vẫn mắc các lỗi pattern:

**Calculation errors:** LLM compute sai trong intermediate steps
- "3 × 7 = 24" (should be 21) → wrong final answer
- Model tiếp tục với số sai, không double-check

**Missing-step errors:** LLM skip important steps
- "We need to find X" → "Therefore X = Y" (skip actual computation)
- Steps implicit rather than explicit

**Semantic misunderstanding errors:** Hiểu sai đề bài
- Misidentify what is asked
- Wrong variable assignment

### Zero-shot CoT vs Few-shot CoT

Few-shot CoT (Wei et al., 2022) cho examples trong prompt → expensive (token cost) và cần careful example selection. Zero-shot CoT không cần examples nhưng less accurate.

**Research question:** Có thể cải thiện zero-shot CoT để approach few-shot CoT performance mà không cần examples, bằng cách thay đổi instruction chứ không phải thêm examples?

---

## 2. Đóng góp Chính

**Contribution 1 — Plan-and-Solve (PS) prompt**: "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan and solve the problem step by step." Tách explicitly planning từ execution.

**Contribution 2 — Plan-and-Solve+ (PS+) prompt**: Extended prompt yêu cầu extract relevant variables, pay attention to calculation, và solve step by step. Tăng calculation accuracy.

**Contribution 3 — Comprehensive evaluation trên 8 benchmarks**: Arithmetic (GSM8K, MultiArith, AQuA-RAT, SVAMP), commonsense (CommonSenseQA, StrategyQA), symbolic (Last Letter, Coin Flip) reasoning.

---

## 3. Phương pháp Chi tiết

### 3.1 Prompt Variants

**Zero-shot CoT (baseline):**
```
Q: {problem}
A: Let's think step by step.
```

**Plan-and-Solve (PS):**
```
Q: {problem}
A: Let's first understand the problem and devise a plan to solve 
   the problem. Then, let's carry out the plan and solve the 
   problem step by step.
```

**Plan-and-Solve+ (PS+):**
```
Q: {problem}
A: Let's first understand the problem, extract relevant variables 
   and their corresponding numerals, and make a complete plan. 
   Then, let's carry out the plan, calculate intermediate results 
   (pay attention to calculation and common sense), solve the 
   problem step by step, and show the answer.
```

**Tại sao explicit planning giúp ích?**

Zero-shot CoT: "Think step by step" → model có thể jump directly to guessed answer rồi rationalize backward.

PS: "Devise a plan first" → model phải articulate *approach* trước khi compute. Planning step force model để identify: what variables are needed, what operations to apply, what order.

PS+: "Extract relevant variables explicitly" → giảm errors từ wrong variable identification. "Pay attention to calculation" → meta-instruction that signals this is important, reduces careless computation.

### 3.2 Ví dụ Cụ thể

**Problem:** "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes muffins for her friends every day with 4933. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much does she make every day at the farmers' market?"

**Zero-shot CoT output (fail example):**
"Janet's ducks lay 16 eggs per day. She eats 3 and uses 4933... [incorrectly assumes 4 for muffins] ...she has 16 - 3 - 4 = 9 eggs. 9 × $2 = $18. Answer: 18."
(Note: 4933 là typo/red herring trong problem — zero-shot thường fails on these)

**PS+ output:**
"Let me first understand the problem and extract variables:
- Eggs per day: 16
- Eggs eaten for breakfast: 3
- Eggs for muffins: 4 (from 'bakes muffins' — note: 4933 appears to be a typo, likely 4)
- Price per egg: $2

Plan:
1. Total eggs: 16
2. Eggs used: 3 (breakfast) + 4 (muffins) = 7
3. Eggs remaining: 16 - 7 = 9
4. Revenue: 9 × $2 = $18

Calculation:
Step 1: 16 - 3 = 13
Step 2: 13 - 4 = 9
Step 3: 9 × 2 = 18

Answer: $18"

### 3.3 Áp Dụng vào RAG (Phân tích của tác giả blog)

*Phần này là phân tích extension sang RAG, không phải claim của paper gốc.*

Plan-and-Solve trong RAG context thêm một **pre-retrieval planning stage**:

**PS-RAG Pipeline:**
```
Stage 1 — Plan (không cần retrieval):
  Input: complex question Q
  LLM → Plan: ["Step 1: Find X", "Step 2: Using X, calculate Y", 
                "Step 3: Combine X and Y for final answer"]

Stage 2 — Retrieve-and-Execute per plan step:
  For each plan step:
    retrieval_query = formulate_query(plan_step + previous_answers)
    docs = Retriever.retrieve(retrieval_query)
    partial_answer = LLM(plan_step, docs, accumulated_context)
    accumulated_context += partial_answer

Stage 3 — Final synthesis:
  Final answer = LLM(Q, full plan, accumulated_context)
```

**So sánh với Least-to-Most:**

| Aspect | Least-to-Most | Plan-and-Solve |
|--------|--------------|----------------|
| Stage 1 output | Sub-questions (specific) | Steps/plan (abstract) |
| Stage 2 | Answer sub-questions | Execute plan steps |
| Best for | Factual multi-hop | Procedural, analytical |
| Example | "Who founded X?" → "Who is CEO of X?" | "Understand → plan → calculate" |

LtM tốt hơn cho chains of factual lookups. PS tốt hơn cho problems requiring a strategy/approach (financial analysis, scientific reasoning, multi-step calculation).

---

## 4. Thực nghiệm và Kết quả

### 4.1 Arithmetic Benchmarks (GPT-3.5-turbo)

| Method | GSM8K | MultiArith | AQuA-RAT | SVAMP |
|--------|-------|-----------|---------|-------|
| Few-shot CoT | 68.9 | 95.8 | 59.8 | 86.4 |
| Zero-shot CoT | 56.4 | 87.4 | 51.6 | 80.6 |
| Plan-and-Solve (PS) | 63.4 | 95.2 | 54.3 | 82.4 |
| **Plan-and-Solve+ (PS+)** | **71.5** | **97.6** | **56.7** | **83.2** |

**Kết quả đáng chú ý:** PS+ *outperforms* few-shot CoT trên GSM8K (71.5% vs 68.9%) và MultiArith (97.6% vs 95.8%) mà không cần bất kỳ few-shot examples nào. Đây là kết quả surprising — zero-shot PS+ tốt hơn few-shot CoT baseline.

### 4.2 Commonsense và Symbolic Reasoning

| Method | CommonSenseQA | StrategyQA | Last Letter | Coin Flip |
|--------|--------------|-----------|------------|----------|
| Zero-shot CoT | 64.5 | 63.7 | 57.6 | 91.4 |
| **PS+** | **70.2** | **65.4** | **60.7** | **94.1** |

Improvements smaller on commonsense tasks — PS helps more on math/calculation-heavy tasks.

### 4.3 Error Analysis

Paper phân loại lỗi theo 3 types và measure reduction với PS+:

| Error Type | Zero-shot CoT | PS+ | Reduction |
|-----------|--------------|-----|-----------|
| Calculation errors | 28.4% | 12.1% | −57% |
| Missing-step errors | 19.7% | 8.3% | −58% |
| Semantic understanding errors | 14.2% | 11.8% | −17% |

PS+ đặc biệt effective với calculation và missing-step errors — phù hợp với design intent. Semantic understanding errors giảm ít hơn — harder to address với prompt wording alone.

---

## 5. Phân tích Phê bình

**Gains mainly on arithmetic, less on commonsense/knowledge tasks:** PS+ cải thiện ~15% trên GSM8K (math) nhưng ~6% trên CommonSenseQA. Pattern này suggests PS benefits come primarily from better structured arithmetic execution, không phải từ general reasoning improvement. Cho RAG use cases (knowledge retrieval + reasoning), gains có thể much smaller.

**Zero-shot advantage may not generalize to all models:** Paper evaluate trên GPT-3.5-turbo. Smaller models (7B, 13B) có thể không follow PS+ instruction equally well — the "pay attention to calculation" instruction relies on model understanding meta-instructions. Không có evaluation trên smaller open-source models.

**Planning phase có thể introduce wrong plan:** Nếu LLM generates a wrong plan (wrong approach, missing necessary steps), execution phase follows the wrong plan. A bad plan + perfect execution = wrong answer. Paper không report rate of planning failures vs execution failures separately.

**Evaluation phiến diện — chủ yếu là math:** 5 trong 8 benchmarks là mathematical reasoning. Generalization to open-domain QA, factual retrieval, hay long-form generation chưa được validated. Math problems có clean, verifiable answers — harder to verify plan quality for open-ended tasks.

**"Pay attention to X" là meta-instruction không robust:** Instruction "pay attention to calculation" works in zero-shot setting, nhưng is fragile. Nếu calculation error là common, re-emphasis của instruction sẽ không solve underlying capability gap. Long-term fix cần better numerical reasoning, không phải better prompts.

---

## 6. Vị trí trong Landscape

| Method | Planning Type | Zero-shot? | RAG Integration | Best For |
|--------|--------------|-----------|----------------|---------|
| Standard CoT | Implicit | Có | Single retrieve | Simple reasoning |
| Few-shot CoT | Implicit | Không | Single retrieve | Complex, in-domain |
| **Plan-and-Solve** | **Explicit (abstract plan)** | **Có** | **Per plan step** | **Procedural/math** |
| Least-to-Most | Explicit (sub-questions) | Partial | Per sub-question | Compositional/factual |
| ReAct | Explicit (action plan) | Partial | Per action | Agent tasks |
| IRCoT | Implicit | Partial | Per CoT step | Multi-hop reasoning |

**PS vs LtM trong RAG context:**
- PS: plan là abstract ("find X, then calculate Y using X") → retrieval queries cần formulated from plan steps
- LtM: plan là concrete sub-questions ("what is X?", "what is Y given X?") → sub-questions có thể retrieve trực tiếp
- LtM thường easier to integrate với retrieval vì sub-questions là natural retrieval queries
- PS có advantage khi problem structure là procedural hay analytical, không factual lookup chains

---

## 7. Takeaway

**Explicit planning instruction mitigates two common CoT failure modes:** Calculation errors và missing-step errors giảm >50% với PS+. Planning forces model to organize thoughts before computing — giảm likelihood của jumping to conclusions và skipping steps.

**Zero-shot PS+ có thể match few-shot CoT:** Trên một số benchmarks (GSM8K, MultiArith), PS+ outperforms few-shot CoT. Điều này có practical implications: có thể save few-shot token budget mà vẫn achieve comparable performance. Cost reduction matter ở scale.

**Meta-instructions ("pay attention to X") work, but fragile:** PS+'s "pay attention to calculation and common sense" works because it triggers deliberate computation mode. Nhưng cần thận trọng về fragility — với different domains hay harder problems, meta-instructions có thể không sufficient. Combine với self-consistency sampling (multiple solutions, majority vote) để robust hơn.

---

## Nguồn

- Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-W., & Lim, E.-P. (2023). Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models. *ACL 2023*. https://arxiv.org/abs/2305.04091
- Kojima, T., et al. (2022). Large Language Models are Zero-Shot Reasoners. *NeurIPS 2022*. https://arxiv.org/abs/2205.01068
- Wei, J., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*. https://arxiv.org/abs/2201.11903
- Zhou, D., et al. (2023). Least-to-Most Prompting. *ICLR 2023*. https://arxiv.org/abs/2205.10625
