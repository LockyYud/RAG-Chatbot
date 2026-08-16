# `flare_2023`: Forward-Looking Active Retrieval — Retrieve Khi Sắp Nói Sai

> **Paper**: Active Retrieval Augmented Generation
> **Tác giả**: Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, Graham Neubig
> **Venue**: EMNLP 2023 — arXiv:2305.06983
> **Loại**: `paper_inspired`

FLARE (Forward-Looking Active REtrieval) xuất phát từ một quan sát đơn giản: LLM biết *khi nào nó không chắc* — token probability thấp là signal tự nhiên. Thay vì retrieve một lần ở đầu (passive RAG) hay để model tự quyết định như Self-RAG, FLARE dùng heuristic hệ thống: generate câu tiếp theo như một bản nháp, nếu có low-probability tokens thì retrieve documents liên quan, sau đó generate lại câu đó với context mới. Cách này active, forward-looking, và không cần fine-tune LLM.

---

## 1. Bối cảnh và Động lực

### Passive RAG: một lần retrieve, không bao giờ nhìn lại

RAG truyền thống có kiến trúc tuyến tính:

```
Query → Retrieve once → Concat context → Generate full answer
```

Vấn đề cốt lõi: tất cả retrieval xảy ra *trước* khi biết generation sẽ đi theo hướng nào. Với câu hỏi multi-hop — "Người sáng lập công ty mà đã acquire Slack năm 2021 sinh năm nào?" — retrieval về Slack không đủ; phải retrieve tiếp về người sáng lập công ty acquire Slack (Salesforce). Nhưng ban đầu không biết Salesforce là answer cho hop 1.

**Hai pattern thất bại của passive RAG với multi-hop queries:**
1. Retrieve về Slack → không tìm được info về Salesforce founder → hallucinate
2. Retrieve cả Slack lẫn Salesforce founder từ đầu → query quá broad → noise

### Active retrieval là gì?

Active retrieval = triggered by generation process itself. Retrieval xảy ra khi và chỉ khi cần — không phải fixed schedule hay external trigger, mà dựa vào signal từ model's own uncertainty.

**FLARE:** dùng token probability như "uncertainty signal". Khi model sắp generate điều gì đó mà nó không chắc (low probability), đó là lúc cần retrieve.

**So sánh với Self-RAG:** Self-RAG dùng `[Retrieve]` token (requires fine-tuning). FLARE dùng probability threshold (zero-shot hoặc few-shot, không cần fine-tune).

---

## 2. Đóng góp Chính

**Contribution 1 — FLARE direct**: Generate câu nháp, detect low-confidence spans, dùng chúng làm retrieval query, regenerate với documents. Inference-time mechanism, không cần fine-tune.

**Contribution 2 — FLARE instruct**: Dùng few-shot prompting với `[Search(query)]` format để model explicitly generate search queries khi cần. Không cần access to token probabilities — applicable cho API-based black-box LLMs.

**Contribution 3 — Forward-looking query generation**: Thay vì dùng input query cho retrieval (như standard RAG), FLARE dùng *generated text so far* + *tentative next sentence* làm query. Queries tự nhiên tiến hóa theo generation.

---

## 3. Phương pháp Chi tiết

### 3.1 FLARE Direct: Probability-Based Active Retrieval

**Core mechanism:**
```
FLARE Direct Inference:
Input: instruction x, LLM M, Retriever R, threshold θ
context = ""

LOOP until generation complete:
  
  # Step 1: Generate tentative next sentence
  s = M.generate_sentence(x, context)    # Greedy/beam decode one sentence
  
  # Step 2: Check confidence
  probs = M.token_probabilities(s)       # Get per-token probability
  low_conf_tokens = [t for t in s if probs[t] < θ]
  
  IF len(low_conf_tokens) == 0:
    # High confidence: keep sentence, continue
    context += s
    
  ELSE:
    # Low confidence: retrieve and regenerate
    
    # Step 3: Form retrieval query
    # Option A: Use full tentative sentence as query
    query = s
    # Option B: Mask out low-confidence tokens, use remaining as query
    # query = " ".join([t if probs[t] >= θ else "[MASK]" for t in s])
    
    # Step 4: Retrieve
    docs = R.retrieve(query)
    
    # Step 5: Regenerate with new context
    new_context = context + relevant_docs_from(docs)
    s = M.generate_sentence(x, new_context)
    context += s

RETURN context (full generated text)
```

**Key design choices:**
- **θ (threshold):** Thường 0.5. Nếu ANY token trong câu < θ → trigger retrieval. Paper test θ ∈ {0.3, 0.5, 0.7} — 0.5 là sweet spot.
- **Query formation:** Dùng full tentative sentence (not masked version) thường tốt hơn vì query đầy đủ hơn và retriever có nhiều signal hơn.
- **Regeneration:** Sau khi retrieve, không giữ tentative sentence mà generate lại từ đầu với context mới.

**Tại sao "forward-looking"?** Query cho retrieval là *tentative future sentence* — model đang retrieve dựa trên *những gì nó nghĩ sẽ nói tiếp theo*, không phải những gì đã nói. Đây là khác biệt với standard RAG (retrieve dựa trên input question) và iterative RAG (retrieve dựa trên output đã generate).

### 3.2 FLARE Instruct: Few-Shot Black-Box Variant

FLARE direct yêu cầu access đến token probabilities — không phải lúc nào cũng available với API-based LLMs. FLARE instruct giải quyết điều này:

**Few-shot prompt format:**
```
Đây là ví dụ về cách trả lời câu hỏi với search:

Q: [Example question requiring multi-hop]
A: [Search("query about first hop")] [Retrieved result] Based on this, ... 
   [Search("query about second hop")] [Retrieved result] Therefore, ...

Q: {actual question}
A: 
```

Model được prompted với ví dụ showing: khi nào nên search, format `[Search(query)]`, và cách tiếp tục sau khi nhận kết quả. Model học pattern này in-context và apply cho câu hỏi mới.

**Execution loop:**
```
FLARE Instruct:
Prompt LLM với few-shot examples có [Search(...)] format
LLM generates text, có thể chứa [Search("query")]

PARSE generated text:
  IF text contains [Search("q")]:
    docs = Retriever.retrieve(q)
    Append retrieved text to context
    Continue generation from context
  ELSE:
    Continue generation normally
```

FLARE instruct không cần token probabilities nhưng yêu cầu model capable của following few-shot instruction format. Performance slightly lower than FLARE direct do implicit signal thay vì explicit probability.

### 3.3 Passive vs Active Retrieval: So sánh trực quan

```
PASSIVE RAG:
 Query ──► [Retrieve] ──► [Generate full answer]
           (once)         (no more retrieval)

ITERATIVE RAG (simple):
 Query ──► [Retrieve] ──► [Generate hop 1] ──► [Retrieve] ──► [Generate hop 2]
           (fixed)                              (fixed)

FLARE DIRECT:
 Query ──► [Generate s1] → high confidence → keep s1
                        → low confidence → [Retrieve with s1] → [Regen s1'] → keep s1'
       ──► [Generate s2] → high confidence → keep s2
                        → low confidence → [Retrieve with s2] → [Regen s2'] → ...
       (adaptive, sentence-by-sentence)
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 Datasets

- **ASQA**: Long-form QA yêu cầu tổng hợp nhiều facts. Metric: EM Recall (exact match over sub-answers), ROUGE-L
- **2WikiMultiHopQA**: Multi-hop QA trên Wikipedia, cần 2 hop reasoning. Metric: F1
- **HotpotQA (open-domain)**: Reasoning qua 2 documents. Metric: F1
- **StrategyQA**: Yes/no questions yêu cầu implicit multi-hop. Metric: accuracy

### 4.2 Kết quả Chính

| Method | ASQA EM Recall | 2WikiMHQA F1 | HotpotQA F1 | StrategyQA |
|--------|---------------|-------------|-------------|-----------|
| Codex (no retrieval) | 28.1 | 17.2 | 20.3 | 67.5% |
| Single-step RAG | 33.5 | 20.3 | 20.8 | 70.1% |
| IRCoT | 31.7 | 31.0 | 32.8 | 70.0% |
| **FLARE direct** | **35.0** | **27.6** | **22.5** | **71.4%** |
| **FLARE instruct** | **32.3** | **26.8** | **21.9** | **70.8%** |

**Kết quả quan trọng:**
- FLARE significantly outperforms single-step RAG trên 2WikiMHQA (+7.3 F1) — nơi multi-hop retrieval rõ ràng cần thiết
- IRCoT outperforms FLARE trên HotpotQA và 2WikiMHQA — IRCoT interleaves CoT reasoning với retrieval, phù hợp hơn cho structured multi-hop
- FLARE direct > FLARE instruct nhất quán, nhưng gap không quá lớn

### 4.3 Ablation: Threshold θ

| θ | 2WikiMHQA F1 | Avg retrievals/query |
|---|------------|---------------------|
| 0.3 | 25.1 | 4.2 |
| 0.5 | 27.6 | 2.8 |
| 0.7 | 24.3 | 1.1 |

θ = 0.5 là best — quá thấp trigger quá nhiều retrievals (noise), quá cao bỏ lỡ nhiều uncertain spans.

### 4.4 Số Retrieval Hops

Với 2WikiMHQA, FLARE average 2.8 retrievals/query — phù hợp với nature của dataset (2-hop questions). Điều này cho thấy FLARE tự nhiên calibrate số retrieval hops theo câu hỏi thay vì fixed số.

---

## 5. Phân tích Phê bình

**Threshold θ là hyperparameter nhạy cảm:** Paper report best results với θ = 0.5, nhưng optimal θ có thể thay đổi đáng kể theo domain, model, và task. Medical text (domain-specific vocabulary) có thể cần θ khác news text. Không có principled way để choose θ mà không tune trên validation set.

**Token probability là noisy uncertainty signal:** Low probability không nhất thiết có nghĩa là wrong — model có thể low-confidence về cách phrase một fact nhưng still know the fact. Ngược lại, model có thể high-confidence nhưng hallucinate. Probability calibration của LLMs là known problem; FLARE inherit tất cả miscalibration issues.

**Tentative sentence làm query có thể đã sai:** Nếu model generate tentative "Einstein was born in 1880" (wrong year), sẽ retrieve docs về Einstein sinh năm 1880 — không tồn tại — thay vì docs đúng. Query từ wrong tentative sentence dẫn đến wrong retrieval, không phải correct retrieval. Paper không phân tích tần suất của failure mode này.

**Regeneration loop không có backtracking:** Sau khi retrieve và regenerate, nếu regeneration vẫn sai, không có mechanism để try again. FLARE là one-shot per sentence — không có self-correction loop.

**IRCoT outperforms FLARE trên structured multi-hop:** Kết quả benchmark cho thấy FLARE không phải best approach cho multi-hop reasoning — IRCoT (interleaved CoT + retrieval) thường tốt hơn trên datasets với structured reasoning chains. FLARE có lợi thế trên ASQA (long-form, less structured) hơn là 2WikiMHQA (structured 2-hop).

---

## 6. Vị trí trong Landscape

| Method | Trigger | Fine-tune? | Query Source | Multi-hop Support |
|--------|---------|------------|-------------|-------------------|
| Passive RAG | Fixed (once) | Không | Original query | Kém |
| Self-RAG | [Retrieve] token | Có | Generated context | Có |
| **FLARE direct** | **Low probability** | **Không** | **Tentative sentence** | **Tốt** |
| **FLARE instruct** | **[Search(...)] format** | **Không** | **Model-generated query** | **Tốt** |
| IRCoT | Each CoT step | Không | CoT reasoning step | Tốt nhất |
| Adaptive-RAG | Query complexity | Không | Original query | Hạn chế |

**FLARE trong bức tranh tổng thể:**
FLARE là middle ground thú vị — không đơn giản như passive RAG, không phức tạp như Self-RAG (fine-tune) hay IRCoT (structured CoT). Phù hợp nhất cho:
- Long-form generation cần multiple retrievals
- Setting không thể fine-tune LLM
- Tasks không có structure cụ thể (khác structured multi-hop)

---

## 7. Takeaway

**Token probability là cheap uncertainty signal có thể exploit:** Bất kỳ LLM nào expose log probabilities đều có thể dùng với FLARE direct. Đây là zero-overhead signal (đã tính toán trong forward pass) mà thông thường bị bỏ qua. FLARE biến side information này thành actionable retrieval trigger.

**Tentative sentence là retrieval query tốt hơn input query:** Original query "Who is the CEO of the company that acquired Slack?" ít specific hơn câu nháp "Salesforce CEO Marc Benioff..." — câu nháp mang thông tin về intermediate state của reasoning, phù hợp hơn cho retrieve next-hop evidence. Forward-looking query formulation này là insight có thể apply rộng rãi.

**Active retrieval không thay thế structured reasoning:** FLARE outperforms passive RAG nhưng không outperforms IRCoT trên structured multi-hop tasks. Lesson: token probability là good trigger cho biết *khi nào* cần retrieve, nhưng structured reasoning (CoT, decomposition) vẫn cần thiết để biết *retrieve gì* cho complex chains of reasoning.

---

## Nguồn

- Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., & Neubig, G. (2023). Active Retrieval Augmented Generation. *EMNLP 2023*. https://arxiv.org/abs/2305.06983
- Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. *ICLR 2024*. https://arxiv.org/abs/2310.11511
- Trivedi, H., et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. *ACL 2023*. https://arxiv.org/abs/2212.10509
- Jeong, S., et al. (2024). Adaptive-RAG. *NAACL 2024*. https://arxiv.org/abs/2403.14403
