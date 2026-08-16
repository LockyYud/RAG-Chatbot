# `hipporag_2024`: Hippocampal Memory cho RAG — Graph + Personalized PageRank

> **Paper**: HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models
> **Tác giả**: Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su
> **Venue**: NeurIPS 2024 — arXiv:2405.14831
> **Loại**: `paper_inspired`

HippoRAG lấy cảm hứng từ hệ thống bộ nhớ hippocampal-neocortical của não người để giải quyết điểm yếu cốt lõi của flat vector retrieval: không capture được associative multi-hop connections giữa các facts phân tán trong corpus. Thay vì retrieve K nearest neighbors theo cosine similarity, HippoRAG xây dựng knowledge graph từ triplets, dùng dense retrieval để tìm seed entities liên quan đến query, rồi chạy Personalized PageRank (PPR) trên graph để propagate relevance theo associative pathways — một cơ chế gần với cách hippocampus index và recall memories.

---

## 1. Bối cảnh và Động lực

### Flat retrieval fail ở đâu với multi-hop queries

Xét câu hỏi: *"Tổng thống Mỹ nào đã ký thỏa thuận khiến quốc gia X gia nhập NATO?"*

**Với flat vector retrieval:**
- Embed query → tìm chunks có highest cosine similarity
- Chunks về NATO membership của country X có thể không mention tên tổng thống explicitly
- Chunks về việc tổng thống ký thỏa thuận NATO có thể không mention country X explicitly
- Hai facts ở *hai passages khác nhau* cần được **liên kết** mới trả lời được

Flat retrieval treat mỗi chunk độc lập — không có mechanism nào để follow links giữa entities.

### Metaphor hippocampal

Paper của Gutiérrez et al. đề xuất: RAG system nên hoạt động giống hippocampus:

- **Neocortex** = LLM (parametric knowledge, pattern completion)
- **Hippocampus** = indexing mechanism (binding associations giữa entities)
- **Entorhinal cortex / Parahippocampal region** = perception layer (text encoder, dense retrieval)

Hippocampus không lưu nội dung (đó là neocortex) mà lưu *associations* — "A relates to B relates to C". Khi recall một memory, hippocampus activate seed nodes rồi propagate activation qua associations. PPR là mathematical analog của mechanism này.

**Research question:** Nếu build explicit association graph và dùng propagation-based retrieval thay vì nearest-neighbor, liệu có capture được multi-hop evidence tốt hơn không?

---

## 2. Đóng góp Chính

**Contribution 1 — OpenIE-style triplet extraction**: Dùng LLM để extract (subject, predicate, object) triplets từ passages, tạo explicit association structure.

**Contribution 2 — Personalized PageRank cho retrieval**: Từ seed entities (found bằng dense retrieval), propagate relevance scores qua KG bằng PPR để find associatively related entities — không chỉ semantically similar.

**Contribution 3 — Passage retrieval via entity scores**: Aggregated entity PPR scores → passage relevance, không trực tiếp retrieve passages theo query embedding.

---

## 3. Phương pháp Chi tiết

### 3.1 Offline Indexing: Building the Hippocampal Graph

**Bước 1 — Extract triplets từ passages:**
Với mỗi passage, LLM được prompted để extract OpenIE-style triplets:

```
Passage: "Marie Curie discovered polonium, named after her native Poland. 
          She was the first woman to win a Nobel Prize."

Extracted triplets:
  (Marie Curie, discovered, polonium)
  (polonium, named after, Poland)
  (Marie Curie, nationality, Polish)
  (Marie Curie, achievement, first woman Nobel Prize winner)
```

**Bước 2 — Build knowledge graph:**
- Nodes: entities (subjects và objects của triplets)
- Edges: predicates, weighted bởi frequency và confidence
- Mỗi entity được linked đến source passages chứa nó

**Bước 3 — Embed entities:**
Mỗi entity name (hoặc entity + description) được embedded bằng dense encoder (SBERT, E5, hoặc similar).

```
KG Structure:
  (Marie Curie) ──[discovered]──► (polonium)
       │                              │
       │──[nationality]──► (Polish) ──[named after]──► (Poland)
       │
       └──[won]──► (Nobel Prize) ──[first woman]──► (achievement)
  
  Passage store: each entity → list of source passage IDs
  Vector store: each entity → embedding vector
```

### 3.2 Online Retrieval: PPR-Based Association Propagation

**Bước 1 — Query processing và Seed Finding:**
```
Input: query q

# Dense retrieval để tìm seed entities
q_embedding = embed(q)
seed_entities = ANN_search(q_embedding, entity_embeddings, k=K_seed)
# K_seed thường = 5-10 entities
```

**Bước 2 — Personalized PageRank:**
PPR là variant của PageRank trong đó restart probability được *personalized* về tập seed nodes.

Công thức PPR:

$$\mathbf{r} = (1 - \alpha) \cdot A \cdot \mathbf{r} + \alpha \cdot \mathbf{p}$$

Trong đó:
- $\mathbf{r} \in \mathbb{R}^{|V|}$: PPR score vector (relevance của mỗi entity)
- $A$: normalized adjacency matrix của KG
- $\alpha$: damping/teleportation factor (thường 0.15 hoặc 0.85 tùy convention)
- $\mathbf{p}$: personalization vector (seed distribution)

**Personalization vector $\mathbf{p}$:**
$$p_v = \begin{cases} s_v / \sum_{u \in V_{\text{seed}}} s_u & \text{nếu } v \in V_{\text{seed}} \\ 0 & \text{otherwise} \end{cases}$$

Trong đó $s_v$ là similarity score của entity $v$ với query từ bước dense retrieval.

**Intuition của PPR:**
- Khởi đầu: các seed entities được gán weight tỷ lệ với similarity score với query
- Mỗi bước: một random walker ở entity $v$ có thể:
  - Với probability $\alpha$: teleport về một seed entity (giữ personalization)
  - Với probability $(1-\alpha)$: random walk sang neighbor entity trong KG
- Sau nhiều bước, entities được "reach" thường xuyên hơn từ seeds sẽ có high PPR score
- Entities kết nối multi-hop với seed entities sẽ accumulate score dần

```
PPR Example:
Query: "Who signed the treaty that made country X join NATO?"

Seed entities (from dense retrieval):
  - "country X" (score: 0.9)
  - "NATO" (score: 0.85)
  - "NATO membership" (score: 0.7)

PPR propagation:
  Iteration 1: 
    "country X" → neighbors: {"NATO", "President Y", "1999", "treaty Z"}
    "NATO" → neighbors: {"Brussels", "country X", "collective defense", "treaties"}
  
  After convergence:
    High PPR: "President Y" (reached from country X + treaty context)
    High PPR: "treaty Z" (reached from both NATO and country X)
    Low PPR: "Brussels" (reached only from NATO, not from country X path)
```

**Bước 3 — Passage Retrieval:**
```
# Aggregate entity PPR scores thành passage scores
for each passage p:
  passage_score[p] = max(ppr_score[e] for e in entities_in_passage[p])
  # hoặc sum hoặc mean — paper test nhiều aggregation strategies

Top-K passages = sorted by passage_score, take top K
Context = retrieved passages
```

### 3.3 End-to-End Pipeline

```
OFFLINE:
Text corpus → [LLM extraction] → Triplets → KG (V, E) + entity embeddings

ONLINE:
Query q
  │
  ▼ embed(q)
  Seed entities E_seed = ANN(q, entity_embeddings, k=5)
  │
  ▼ Personalized PageRank on KG
  r = PPR(KG, personalization=E_seed)  # entity relevance scores
  │
  ▼ Aggregate to passages
  passage_scores = aggregate(r, entity_passage_mapping)
  │
  ▼ Top-K passages
  Context = top-K passages by passage_scores
  │
  ▼ LLM generation
  Answer = LLM(query, context)
```

---

## 4. Thực nghiệm và Kết quả

### 4.1 Datasets — Focus trên Multi-Hop

HippoRAG được đánh giá chủ yếu trên datasets yêu cầu multi-hop reasoning:

- **MuSiQue**: 2-hop questions, requires explicit multi-hop reasoning
- **HotpotQA**: 2-hop questions trên Wikipedia
- **2WikiMultiHopQA**: Multi-hop QA với 2-4 hops

Metric: F1 score (answer coverage)

### 4.2 Kết quả Chính

| Method | MuSiQue F1 ↑ | HotpotQA F1 ↑ | 2WikiMHQA F1 ↑ |
|--------|-------------|-------------|--------------|
| BM25 | 26.7 | 73.6 | 47.9 |
| DPR | 30.8 | 72.3 | 49.2 |
| Contriever | 32.1 | 74.5 | 52.3 |
| IRCoT | 37.1 | 78.6 | 55.4 |
| **HippoRAG** | **42.5** | **78.3** | **52.3** |

**Điểm nổi bật:**
- MuSiQue: HippoRAG đạt 42.5 vs IRCoT 37.1 — significant gain (+5.4 F1) trên dataset yêu cầu multi-hop nhất
- HotpotQA: HippoRAG ≈ IRCoT (78.3 vs 78.6) — cạnh tranh trực tiếp mà không cần chain-of-thought reasoning
- 2WikiMHQA: IRCoT > HippoRAG — IRCoT có structured CoT reasoning advantage trên dataset này

### 4.3 Ablation: Components của HippoRAG

| Variant | MuSiQue F1 |
|---------|-----------|
| HippoRAG đầy đủ | 42.5 |
| Không có PPR (chỉ seed entities → passages) | 33.2 |
| Không có KG (chỉ dense retrieval) | 32.1 |
| PPR α = 0.1 (ít teleportation) | 40.1 |
| PPR α = 0.85 (nhiều teleportation) | 38.7 |
| PPR α = 0.5 | **42.5** |

PPR là component quan trọng nhất — bỏ PPR (chỉ dùng seed entities) loses 9.3 F1 points. Optimal α = 0.5 là balance giữa exploration (follow KG paths) và exploitation (stay near seeds).

---

## 5. Phân tích Phê bình

**PPR là global computation và không scale tốt:** PPR phải converge trên toàn bộ KG — với corpus lớn (100K+ entities), mỗi query cần power iteration trên large sparse matrix. Paper evaluate trên Wikipedia subsets (~100K passages), nhưng production corpora thường lớn hơn nhiều. Approximate PPR (push-based PPR cho local subgraph) là possible solution nhưng chưa được explored trong paper.

**Triplet extraction quality là critical bottleneck:** OpenIE-style extraction không hoàn hảo — LLM có thể miss key relations, hallucinate entities, hay misparse complex sentences. Với domain-specific text (legal contracts, medical literature), triplet extraction quality có thể giảm đáng kể. Paper dùng GPT-3.5/4 nhưng không systematic evaluation of extraction quality vs downstream retrieval quality.

**Biological metaphor là appealing nhưng loosely justified:** Paper claim HippoRAG mirrors hippocampal memory indexing, nhưng actual hippocampal mechanisms (place cells, sharp-wave ripples, sleep consolidation) không có direct analog trong PPR. Metaphor là useful framing nhưng không mechanistically grounded.

**KG không capture all types of information:** OpenIE triplets tốt cho factual claims (X did Y to Z) nhưng kém cho: causal reasoning ("because"), temporal order, negations, và conditional facts. Passages về processes, procedures, hay nuanced arguments khó được captured đầy đủ bằng (subject, predicate, object) triplets.

**Evaluation domain hẹp:** HippoRAG được tested chủ yếu trên Wikipedia-based QA. Performance trên domain-specific corpora (legal, medical, scientific papers) chưa được reported. Real-world corpora thường có richer structure hơn Wikipedia passages đơn giản.

---

## 6. Vị trí trong Landscape

| Method | Multi-hop Support | Index Type | Query Mechanism | Cost |
|--------|-----------------|-----------|----------------|------|
| Flat RAG (DPR) | Kém | Vector | ANN | Thấp |
| IRCoT | Tốt | Vector | Iterative retrieval + CoT | Thấp (inference) |
| GraphRAG | Khá | KG + communities | Map-Reduce | Rất cao |
| LightRAG | Tốt | KG + embeddings | Dual-level ANN | Cao |
| **HippoRAG** | **Tốt nhất (MuSiQue)** | **KG + PPR** | **Seed → PPR propagation** | **Cao** |
| RAPTOR | Khá | Tree + summaries | Collapsed tree | Trung bình |

**HippoRAG vs GraphRAG:**
- HippoRAG: tốt cho factual multi-hop lookup (who, what, when)
- GraphRAG: tốt hơn cho thematic global questions (what are the main themes)
- HippoRAG rẻ hơn (no community summarization), nhưng không support global queries tốt bằng

**HippoRAG vs IRCoT:**
- HippoRAG retrieves multi-hop evidence trong một step (PPR)
- IRCoT explicit reasoning trước mỗi retrieval step → better for structured chains
- HippoRAG có advantage khi associations are implicit and graph-based, không explicit

---

## 7. Takeaway

**Graph structure captures associative links mà vector similarity bỏ qua:** Hai passages về cùng entity nhưng khác aspects (Marie Curie's nationality vs Marie Curie's discoveries) có thể không similar về embedding nhưng linked qua entity "Marie Curie" trong KG. PPR propagates relevance qua các links này — đây là fundamental advantage của graph-based over similarity-based retrieval.

**PPR teleportation parameter α điều chỉnh local vs global scope:** Α cao → ở gần seeds (more local). Α thấp → explore sâu qua graph (more global, có thể noisy). Đây là tunable knob cho retrieval scope mà flat retrieval không có — valuable trong settings khi biết trước loại queries cần balance khác nhau.

**Khi multi-hop association là bottleneck, graph > vector:** Kết quả ablation rõ ràng: bỏ PPR (giữ KG nhưng chỉ seed-to-passage) mất 9+ F1 points. Graph structure không đủ — phải có propagation mechanism. Lesson chung: nếu evaluation cho thấy retrieval recall cao nhưng multi-hop accuracy thấp, đây là indicator để consider graph-based approach.

---

## Nguồn

- Gutiérrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., & Su, Y. (2024). HippoRAG: Neurologically Inspired Long-Term Memory for Large Language Models. *NeurIPS 2024*. https://arxiv.org/abs/2405.14831
- Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). The PageRank Citation Ranking: Bringing Order to the Web. *Stanford Technical Report*.
- Trivedi, H., et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning (IRCoT). *ACL 2023*. https://arxiv.org/abs/2212.10509
- Edge, D., et al. (2024). GraphRAG. arXiv:2404.16130. https://arxiv.org/abs/2404.16130
- Guo, Z., et al. (2024). LightRAG. arXiv:2410.05779. https://arxiv.org/abs/2410.05779
