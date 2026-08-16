# `replug_2023`: RAG Cho Black-Box LLM — Khi Không Có Gradients

> **Paper**: REPLUG: Retrieval-Augmented Language Models
> **Tác giả**: Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, Wen-tau Yih
> **Venue**: NAACL 2023 — arXiv:2301.12652
> **Loại**: `paper_inspired`

REPLUG giải quyết bài toán mà hầu hết RAG research bỏ qua: làm sao augment một LLM mà bạn không thể modify, fine-tune, hay lấy internal probabilities từ? Với GPT-3 và các proprietary models, chỉ có input và output. REPLUG chứng minh rằng ensemble over multiple retrieved passages — mỗi passage được prepend vào query độc lập, output được weighted bởi LLM perplexity của passage — cải thiện cả language modeling lẫn downstream tasks. REPLUG LSR đi xa hơn bằng cách train retriever với signal từ chính LLM.

---

## 1. Bối cảnh và Động lực

### Vấn đề với RAG classic

RAG gốc (Lewis et al., 2020) và FiD (Izacard & Grave, 2021) assume access to model internals:
- **FiD**: T5 encoder + decoder được fine-tune end-to-end với retrieved passages
- **RAG**: BART được fine-tune với marginalization over retrieved passages
- Cả hai yêu cầu gradient access và khả năng modify model weights

Với GPT-3, Codex, và sau này GPT-4, điều này không thể. Models là black-box: bạn gửi text, nhận text. Đây là thực tế của hầu hết enterprise RAG deployment năm 2023.

**Vấn đề quan trọng hơn:** Ngay cả với open-source models, fine-tuning LLM lớn (175B parameters như GPT-3) là expensive và không phải lúc nào cũng cần thiết. Có cách nào augment LLM *tại inference time* mà không cần fine-tune không?

### REPLUG framework

REPLUG đặt câu hỏi khác: thay vì "làm thế nào để LLM học dùng retrieved context", hỏi "làm thế nào compose retrieved passages vào input để LLM có thể dùng mà không cần training?"

Research question: **Có thể cải thiện LLM performance bằng cách ensemble predictions over multiple retrieved passages, sử dụng chỉ input/output API?**

---

## 2. Đóng góp Chính

**Contribution 1 — REPLUG framework cho black-box LLM**: Retrieve K passages, cho mỗi passage tính LLM probability của query given passage, weight predictions theo scores, ensemble final probabilities. Không cần gradient hay fine-tuning LLM.

**Contribution 2 — REPLUG LSR (LM-Supervised Retrieval)**: Dùng LLM probability scores làm training signal cho retriever. Retriever được train để assign high scores cho passages mà LLM finds helpful (giảm perplexity). Vẫn black-box với LLM nhưng train retriever dùng LLM feedback.

**Contribution 3 — Evaluation trên language modeling và MMLU**: Chứng minh REPLUG giúp ích không chỉ cho factual QA mà còn cho language modeling (perplexity reduction) và knowledge-intensive tasks (MMLU).

---

## 3. Phương pháp Chi tiết

### 3.1 REPLUG: Ensemble over Retrieved Passages

**Bước 1 — Retrieval:**
Với input context $x$ (câu hỏi hoặc text cần continue), retrieve K passages $\{d_1, ..., d_K\}$ từ corpus $\mathcal{D}$ sử dụng retriever $R$.

**Bước 2 — Passage Scoring bằng LLM:**
Với mỗi passage $d_i$, tính probability của LLM generating $x$ given $d_i$ là context:

$$s(d_i, x) = P_{\text{LLM}}(x | d_i) = \exp\left(\frac{1}{|x|}\sum_{t=1}^{|x|} \log P_{\text{LLM}}(x_t | d_i, x_{<t})\right)$$

Đây thực chất là **perplexity-based score**: passage nào giúp LLM predict $x$ tốt hơn (lower perplexity) thì score cao hơn.

**Bước 3 — Ensemble:**
Tính softmax normalized weights từ LLM scores:

$$P(y | x) = \sum_{i=1}^{K} \underbrace{\frac{e^{\lambda \cdot s(d_i, x)}}{\sum_j e^{\lambda \cdot s(d_j, x)}}}_{\text{passage weight}} \cdot P_{\text{LLM}}(y | d_i, x)$$

Trong đó:
- $y$ là generated output (next tokens hoặc full answer)
- $\lambda$ là temperature scaling parameter
- Passage weight tỷ lệ với "mức độ passage này giúp LLM hiểu input"

**Quan trọng:** Đây là *distribution ensemble*, không phải output ensemble. Mỗi passage cho ra một distribution over next tokens; final distribution là weighted average của K distributions. LLM API phải cho phép lấy log probabilities (available trong GPT-3 API, Llama API...).

```
REPLUG Inference:
Input: query x, corpus D, retriever R, LLM M

1. docs = R.retrieve(x, k=K)

2. scores = []
   for d in docs:
     s = M.log_prob(x | context=d)  # log P(x|d)
     scores.append(s)

3. weights = softmax(scores, temperature=λ)

4. final_prob = Σ weights[i] × M.prob_dist(y | docs[i], x)

5. answer = argmax(final_prob)
```

### 3.2 REPLUG LSR: Training Retriever với LLM Feedback

REPLUG base dùng off-the-shelf retriever (Contriever, DPR). REPLUG LSR *trains* retriever để align với LLM preferences.

**Training signal:**
Với query $x$ và retrieved passages $\{d_1, ..., d_K\}$:
- Compute LLM scores: $s_{\text{LM}}(d_i) = P_{\text{LLM}}(x | d_i)$ (passage perplexity giúp ích bao nhiêu)
- Compute retriever scores: $s_{\text{R}}(d_i) = R(d_i, x)$ (retriever rank passage này như thế nào)
- Loss: KL divergence để align retriever distribution với LLM feedback distribution:

$$\mathcal{L}_{\text{LSR}} = \text{KL}(\tilde{s}_{\text{LM}} \| \tilde{s}_{\text{R}})$$

Trong đó $\tilde{s}$ là softmax-normalized versions của raw scores.

**Ý nghĩa:** Retriever học "passages nào LLM thấy useful cho query này" — không phải traditional relevance mà là LLM-centric relevance.

```
REPLUG LSR Training:
For each batch (x, D_retrieved):
  # LM scores (frozen LLM, black-box)
  lm_scores = [LLM.log_prob(x | d) for d in D_retrieved]
  lm_dist = softmax(lm_scores)
  
  # Retriever scores (trainable)
  r_scores = [Retriever.score(d, x) for d in D_retrieved]
  r_dist = softmax(r_scores)
  
  # KL divergence loss
  loss = KL(lm_dist || r_dist)
  loss.backward()
  optimizer.step(Retriever)  # Only update retriever, not LLM
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 Language Modeling (Perplexity)

Đánh giá trên Pile test set (diverse text corpus) dùng GPT-3 as backbone:

| Method | GPT-3 Curie Perplexity ↓ | GPT-3 Davinci Perplexity ↓ |
|--------|------------------------|--------------------------|
| No retrieval (baseline) | 10.3 | 9.1 |
| REPLUG (Contriever) | 9.5 | 8.7 |
| **REPLUG LSR** | **8.9** | **8.2** |

REPLUG LSR giảm perplexity khoảng 10% so với no-retrieval baseline — significant improvement cho language modeling.

### 4.2 MMLU (Knowledge-Intensive Tasks)

Đánh giá trên MMLU benchmark (57 academic subjects, multiple choice):

| Method | MMLU Accuracy |
|--------|--------------|
| GPT-3 davinci (no retrieval) | 53.9% |
| GPT-3 davinci + REPLUG (Contriever) | 56.1% |
| **GPT-3 davinci + REPLUG LSR** | **58.2%** |
| Codex + REPLUG LSR | ~57% |

REPLUG LSR cải thiện MMLU accuracy khoảng 4.3% tuyệt đối so với no-retrieval — consistent improvement across subject areas.

### 4.3 REPLUG vs REPLUG LSR

| Retriever | Perplexity (Pile) | MMLU |
|-----------|-----------------|------|
| Contriever (off-the-shelf) | 9.5 | 56.1% |
| DPR (off-the-shelf) | 9.7 | 55.8% |
| **Contriever + LSR training** | **8.9** | **58.2%** |

LSR training consistently cải thiện so với off-the-shelf retriever — chứng tỏ LLM feedback là valuable signal cho retriever training.

---

## 5. Phân tích Phê bình

**LLM perplexity là proxy không hoàn hảo cho usefulness:** REPLUG sử dụng "passage giúp LLM predict input better" làm proxy cho "passage useful cho answering query". Nhưng lower perplexity ≠ correct answer. Một passage có thể lower perplexity vì nó semantically similar đến input (topic overlap) chứ không phải vì chứa evidence cần thiết. Đây là fundamental assumption gap.

**K lần LLM inference:** Mỗi query cần K LLM calls để score K passages. Với K = 10 và GPT-3, đây là 10x cost tăng so với standard RAG. Paper không report latency hay cost breakdown. Với GPT-4 pricing, REPLUG at scale là expensive.

**REPLUG LSR cần labeled data ngầm:** Training REPLUG LSR yêu cầu queries và retrieved passages, rồi score passages bằng LLM. Đây không phải unsupervised — cần query set representative của distribution. Với domain-specific corpora, cần domain-specific queries để train retriever effectively.

**Black-box assumption ngày càng lỗi thời:** Khi Llama, Mistral, và nhiều open-source models đã available, "black-box LLM" là ít phổ biến hơn. Với open-source models, có thể fine-tune trực tiếp — FiD hoặc joint training approaches có thể outperform REPLUG với cùng compute budget.

**Evaluation phiến diện:** Paper focus vào language modeling (perplexity) và MMLU — cả hai là aggregate metrics. Long-form generation quality, citation accuracy, hallucination rate không được evaluated. Không rõ REPLUG giúp ích như thế nào cho RAG use cases thực tế như document QA hay summarization.

---

## 6. Vị trí trong Landscape

| Method | LLM Access Required | Training | K Inference Calls | Use Case |
|--------|--------------------|---------|--------------------|---------|
| Standard RAG (no retrieval) | Black-box OK | Không | 1 | Simple Q&A |
| RAG (Lewis 2020) | Fine-tune needed | Có (LLM) | 1 | Fine-tunable LLM |
| FiD | Fine-tune needed | Có (seq2seq) | K (encoder) | Fine-tunable encoder |
| **REPLUG** | **Black-box OK** | **Không (base)** | **K** | **Black-box LLM** |
| **REPLUG LSR** | **Black-box for LLM** | **Có (retriever only)** | **K** | **Black-box LLM + trainable retriever** |
| In-context RAG | Black-box OK | Không | 1 (long context) | Simple, single passage |

**Khi nào dùng REPLUG:**
- LLM là black-box (GPT-4 API, Claude API) và không thể fine-tune
- Câu hỏi cần evidence từ nhiều nguồn (benefit từ ensemble)
- Có LLM log-probabilities API (requirement không phải lúc nào cũng satisfied)
- Budget cho K × LLM calls mỗi query

**Không phù hợp khi:**
- Latency/cost critical (K LLM calls tốn kém)
- LLM không expose log probabilities (nhiều production APIs không có)
- Câu hỏi simple, single-document lookup

---

## 7. Takeaway

**LLM perplexity là retrieval supervision signal:** REPLUG LSR chứng minh rằng ngay cả black-box LLM có thể cung cấp training signal cho retriever thông qua perplexity scores. Đây là một hướng khác với traditional relevance labeling — không cần human annotators, chỉ cần LLM API calls.

**Ensemble over retrieved passages giải quyết uncertainty của retriever:** Thay vì chọn một passage tốt nhất và dùng, REPLUG weighted-averages K passages. Điều này robust hơn khi retriever không hoàn hảo — nếu passage tốt nhất ở rank 2 thay vì rank 1, ensemble vẫn capture được nó với weight thấp hơn thay vì bỏ qua hoàn toàn.

**Black-box constraint thúc đẩy sáng tạo:** REPLUG là ví dụ hay về cách constraint (không access LLM internals) dẫn đến giải pháp clean và generalizable hơn. Pattern "retrieve, score with LLM, ensemble" có thể apply với bất kỳ LLM nào expose log probabilities — không bị tied với một architecture cụ thể.

---

## Nguồn

- Shi, W., Min, S., Yasunaga, M., Seo, M., James, R., Lewis, M., Zettlemoyer, L., & Yih, W.-T. (2023). REPLUG: Retrieval-Augmented Language Models. *NAACL 2023*. https://arxiv.org/abs/2301.12652
- Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*. https://arxiv.org/abs/2005.11401
- Izacard, G., & Grave, E. (2021). Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering. *EACL 2021*. https://arxiv.org/abs/2007.01282
- Gao, T., et al. (2022). Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever). *TMLR 2022*. https://arxiv.org/abs/2112.09118
