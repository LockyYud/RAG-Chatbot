# `ares_eval`: Đánh Giá RAG Với Độ Tin Cậy Thống Kê Thay Vì LLM-as-Judge Thuần Túy

> **Paper**: ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems
> **Tác giả**: Jon Saad-Falcon, Omar Khattab, Christopher Potts, Matei Zaharia
> **Venue**: NAACL 2024 — arXiv:2311.09476
> **Loại**: `paper_inspired`

ARES đi theo hướng khác với RAGAS: thay vì dùng GPT-4 làm judge cho mọi evaluation call, ARES train lightweight domain-specific LM judges từ synthetic data, và dùng Prediction-Powered Inference (PPI) để kết hợp LM predictions với một lượng nhỏ human annotations nhằm tạo ra confidence intervals có statistical guarantees. Kết quả: evaluations với statistical bounds, rẻ hơn GPT-4 judge, và adaptable đến domain-specific data — phù hợp cho systematic benchmarking hơn là quick ad-hoc evaluation.

---

## 1. Bối cảnh và Động lực

### RAGAS và LLM-as-Judge: nhanh nhưng thiếu độ tin cậy thống kê

[RAGAS](./ragas_eval.md) (Es et al., 2023) đề xuất 4 metrics (faithfulness, answer relevance, context precision, context recall) đánh giá bằng LLM-as-judge. Approach này: nhanh, flexible, không cần human annotation. Nhưng có hai vấn đề lớn:

**Vấn đề 1 — GPT-4 judge expensive và không reproducible:** Mỗi evaluation query gọi GPT-4, tốn kém và kết quả có thể thay đổi khi GPT-4 model version update. Evaluation trên 1000 queries = 1000 GPT-4 calls.

**Vấn đề 2 — Không có statistical guarantees:** RAGAS trả về point estimates (e.g., faithfulness = 0.73) mà không có confidence interval. Liệu 0.73 ± 0.01 hay 0.73 ± 0.15? Khi so sánh hai RAG systems, cải thiện 0.05 điểm có significant không? Không có statistical framework để trả lời.

### Bài toán ARES giải quyết

ARES đặt câu hỏi: có thể đánh giá RAG systems với (a) cost thấp hơn GPT-4 judge, (b) statistical confidence intervals, và (c) adaptability đến domain-specific datasets?

---

## 2. Đóng góp Chính

**Contribution 1 — Lightweight domain-specific LM judges**: Fine-tune small models (DeBERTa, BERT-large) trên synthetic training data làm judges cho 3 dimensions: context relevance, answer faithfulness, answer relevance.

**Contribution 2 — Prediction-Powered Inference (PPI)**: Statistical framework cho phép combine nhiều cheap LM predictions với ít expensive human annotations để create confidence intervals với statistical guarantees.

**Contribution 3 — Synthetic training data generation pipeline**: Generate balanced positive/negative training examples từ corpus để bootstrap judge training mà không cần large human-labeled datasets.

---

## 3. Phương pháp Chi tiết

### 3.1 Ba Dimensions và LM Judges

ARES evaluate 3 dimensions (tương tự RAGAS nhưng khác implementation):

| Dimension | Đánh giá gì | Judge input |
|-----------|------------|------------|
| Context Relevance | Context có relevant với query không? | (query, context) |
| Answer Faithfulness | Answer có được support bởi context không? | (query, context, answer) |
| Answer Relevance | Answer có trả lời được query không? | (query, answer) |

Mỗi dimension có **một judge model riêng** (binary classifier) — không phải generative LLM.

**Tại sao classifier thay vì generative judge?**
- Classifier (DeBERTa-large ~400M params) rẻ hơn 100x so với GPT-4
- Deterministic và reproducible
- Can be fine-tuned cho specific domain → better calibration
- Faster inference (no autoregressive decoding)

### 3.2 Synthetic Training Data Generation

Để train judge models mà không cần large human-labeled dataset:

**Positive examples** (relevant/faithful/relevant):
1. Sample passages từ corpus
2. LLM (GPT-3.5 hoặc local model) generates question từ passage
3. LLM generates answer conditioned on question + passage
4. Triple (question, passage, answer) = positive training example

**Negative examples** (irrelevant/unfaithful/irrelevant):
1. **Irrelevant context:** Swap passage với random passage từ corpus
2. **Unfaithful answer:** Perturb answer bằng cách replace entities với wrong ones
3. **Irrelevant answer:** Use answer từ different question

```
Training Data Pipeline:
Corpus documents
    │
    ▼ LLM generation
Positive triples: (q, p, a) where p is source of a
    │
    ├──► Irrelevant context negatives: (q, p_random, a)
    ├──► Unfaithful answer negatives: (q, p, a_perturbed)
    └──► Irrelevant answer negatives: (q, p, a_other_question)
    │
    ▼
Fine-tune DeBERTa-large as binary judge
```

### 3.3 Prediction-Powered Inference (PPI)

Đây là contribution quan trọng nhất và technically novel nhất của ARES.

**Setup:**
- Có N queries trong evaluation set (N thường lớn: 500-2000)
- Có n human annotations (n << N: thường 50-200)
- Có N LM judge predictions (cheap)
- Goal: estimate true metric (e.g., faithfulness) với confidence interval

**PPI framework (Angelopoulos et al., 2023):**

LM predictions là *approximately* correct nhưng có bias $b$:
$$\hat{\mu}_{\text{LM}} = \mu_{\text{true}} + b$$

Human annotations cho phép estimate bias từ held-out set. PPI correction:

$$\hat{\mu}_{\text{PPI}} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} \hat{f}(x_i)}_{\text{LM mean}} - \underbrace{\frac{1}{n}\sum_{j=1}^{n} (\hat{f}(x_j) - y_j)}_{\text{bias estimate}}$$

Trong đó:
- $\hat{f}(x_i)$: LM judge prediction cho example $i$
- $y_j$: human label cho example $j$ (chỉ có cho subset $n$)
- $\hat{\mu}_{\text{PPI}}$: debiased estimate của true metric

**Confidence interval:**

PPI cho confidence interval width ~ $\pm C/\sqrt{n}$ với constant $C$ phụ thuộc vào variance của LM predictions và correlation với human labels. Với n = 150 và good LM judge, typical width là ±0.03 ở 95% confidence.

**Ví dụ cụ thể:**
```
Evaluation scenario:
- N = 1000 queries
- n = 150 human annotations (từ 1000)
- LM judge faithfulness scores: mean = 0.76
- Human labels cho 150 queries: LM mean = 0.74, Human mean = 0.71
- Bias estimate: 0.74 - 0.71 = 0.03

PPI corrected estimate: 0.76 - 0.03 = 0.73
95% CI: 0.73 ± 0.025

→ System A faithfulness = 0.73 ± 0.025
→ System B faithfulness = 0.65 ± 0.024
→ Difference: 0.08, CI for difference: [0.04, 0.12]
→ Statistically significant (interval doesn't include 0)
```

### 3.4 Full ARES Pipeline

```
SETUP (one-time per corpus/domain):
1. Generate synthetic training data từ corpus
2. Fine-tune judge models (DeBERTa-large × 3 dimensions)
3. Collect n human annotations từ validation set

EVALUATION (per RAG system):
1. Run RAG system trên test queries → (query, context, answer) triples
2. Run judge models → N predictions per dimension
3. Apply PPI correction với n human annotations
4. Output: point estimate + confidence interval per dimension

COMPARISON:
- Compare two systems: compute PPI estimates và test if CIs overlap
- If CIs don't overlap → statistically significant difference
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 Pearson Correlation với Human Judgments

ARES được evaluated bằng cách measure correlation giữa ARES scores và human annotations trên 3 RAG datasets (NQ, HotpotQA, WoW — Wizard of Wikipedia).

| Metric | ARES (DeBERTa judge) | RAGAS (GPT-4 judge) |
|--------|---------------------|---------------------|
| Context Relevance | 0.832 | 0.738 |
| Answer Faithfulness | 0.817 | 0.768 |
| Answer Relevance | 0.801 | 0.712 |

ARES domain-specific judges correlate better với human judgments than GPT-4 general-purpose judge. Điều này counter-intuitive — smaller fine-tuned model > larger general model vì fine-tuning với domain data captures domain-specific relevance patterns.

### 4.2 PPI Statistical Power

| Human annotations (n) | CI width (±) | Statistical power |
|----------------------|-------------|------------------|
| 50 | ±0.052 | 60% |
| 100 | ±0.037 | 78% |
| **150** | **±0.030** | **85%** |
| 300 | ±0.021 | 95% |

Với n = 150 human annotations, 85% power để detect 0.05 difference giữa hai systems. Practical: 150 annotations là feasible (few hours of human annotation) cho corpus lớn.

### 4.3 Cost Comparison

| Evaluator | Cost per 1000 queries | Reproducible |
|-----------|----------------------|-------------|
| Human only | $300-500 | Khó |
| GPT-4 judge (RAGAS-style) | $20-30 | Có (but model can change) |
| **ARES (DeBERTa judge)** | **$0.50-2** | **Có** |
| ARES + 150 human labels | $0.50-2 + $150 (one-time) | Có |

ARES inference cost là ~10-15x rẻ hơn GPT-4 judge per query, với better correlation và added statistical guarantees.

---

## 5. Phân tích Phê bình

**Synthetic training data quality bounds judge quality:** Judge models chỉ tốt bằng training data của chúng. Nếu LLM-generated questions hay perturbed negatives không representative của real failure modes, judges sẽ miss important error types. Ví dụ: nếu negatives chỉ include random-passage irrelevant contexts (easy to detect), judge sẽ struggle với subtly irrelevant contexts (similar topic nhưng không answerable).

**PPI requires n human annotations — "automated" là partial:** ARES không fully automated. PPI component yêu cầu n ground truth human labels để debias và compute CI. Với 150 annotations, đây là manageable nhưng không free. Paper frame này như feature (statistical rigor), nhưng là real cost khi deploying cho new domain/corpus.

**Domain-specific fine-tuning = maintenance overhead:** Mỗi new domain cần: new synthetic data generation, new judge fine-tuning. Với rapidly evolving RAG system hay multiple domains, maintenance cost accumulates. GPT-4 judge (RAGAS-style) được dùng zero-shot — no domain-specific training needed.

**PPI assumptions:** PPI assumes LM judge predictions are "informative" (correlated với true labels). Nếu LM judge is poorly calibrated (e.g., always predicts high scores), PPI corrections sẽ have wide CIs and limited power. Paper reports good correlation on their test sets nhưng không guarantee cho arbitrary new domains.

**Evaluation pipeline complexity:** ARES pipeline requires: corpus access, LLM generation for synthetic data, fine-tuning infrastructure, human annotation collection, PPI implementation. Much more complex than RAGAS (just run prompts). For teams without MLOps infrastructure, barrier to adoption is high.

---

## 6. Vị trí trong Landscape

| Framework | Judge Type | Statistical Guarantees | Domain-specific | Cost/1k queries |
|-----------|-----------|----------------------|----------------|----------------|
| Human evaluation | Human | Có (sample stats) | Có | $300-500 |
| RAGAS | GPT-4 (generative) | Không | Không | $20-30 |
| **ARES** | **Fine-tuned DeBERTa** | **Có (PPI)** | **Có** | **$0.50-2** |
| G-Eval | GPT-4 (generative) | Không | Không | $20-30 |
| Prometheus | Fine-tuned LLaMA | Không | Có (general) | $2-5 |

**Khi nào dùng ARES:**
- Systematic benchmarking (compare multiple RAG systems, need statistical significance)
- Large-scale evaluation (nhiều queries, GPT-4 cost prohibitive)
- Domain-specific corpus (benefit từ domain-adapted judges)
- Need reproducible, versioned evaluation (không dependent on GPT-4 version)

**Khi nào dùng RAGAS:**
- Quick prototyping và ad-hoc evaluation
- No infrastructure for fine-tuning
- Domain-agnostic evaluation
- Team không cần statistical CI

---

## 7. Takeaway

**Statistical confidence intervals là missing piece của RAG evaluation:** Khi engineering team report "our RAG achieves faithfulness 0.73", không biết là 0.73 ± 0.03 (tight) hay 0.73 ± 0.15 (wide). ARES chứng minh rằng với chỉ 150 human labels, có thể make statistical claims about RAG quality — standard practice trong A/B testing nhưng absent trong RAG evaluation.

**Domain-specific small judges > general large judges cho calibration:** Counter-intuitive nhưng consistent: fine-tuned DeBERTa-large correlates better với human judgments than GPT-4 on specific domains. Lesson: when data distribution is known and specific, smaller specialized models can outperform larger generalists. Applicable beyond evaluation — cùng principle cho retrieval và generation judges.

**"Automated" evaluation là spectrum, không binary:** RAGAS là more automated (zero human labels needed) nhưng less rigorous. ARES requires 150 human labels but provides statistical guarantees. Full human evaluation là most rigorous but most expensive. Chọn điểm trên spectrum dựa vào: stakes (production deployment cần rigorous), scale (nhiều queries = prefer cheap per-query), và team capabilities (có fine-tuning infrastructure không?).

---

## Nguồn

- Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *NAACL 2024*. https://arxiv.org/abs/2311.09476
- Es, S., et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. *EACL 2024*. https://arxiv.org/abs/2309.15217
- Angelopoulos, A. N., et al. (2023). Prediction-Powered Inference. *Science 2023*. https://arxiv.org/abs/2301.09633
- He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. *ICLR 2021*. https://arxiv.org/abs/2006.03654
