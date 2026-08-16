# `agentic_rag_arag`: Khi RAG Chuyển Từ Pipeline Tĩnh Sang Agent Tự Quyết

> **Nguồn chính**: A-RAG — *Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces* (arXiv:2602.03442, 2026)
> **Bối cảnh**: *Reasoning RAG via System 1 or System 2: A Survey* (arXiv:2506.10408, 2025); Search-R1 (arXiv:2503.09516, 2025); AutoSearch (arXiv:2604.17337, 2026)
> **Loại**: `paper_inspired`

Năm 2026, câu hỏi "RAG đã chết chưa?" được trả lời rõ: chưa, nhưng *hình dạng* của nó đổi. Pipeline tĩnh `retrieve → generate` đang nhường chỗ cho một agent: LLM tự quyết định khi nào truy hồi, dùng công cụ nào, truy vấn gì, và khi nào đủ bằng chứng để trả lời. Bài này phân tích cú chuyển dịch đó và cách đưa nó vào lab mà **không cần train RL**.

---

## 1. Bối cảnh và Động lực

RAG cổ điển một-lượt thất bại ở ba chỗ:
- **Multi-hop**: câu hỏi cần bắc cầu nhiều mẩu bằng chứng. Một lần retrieve không đủ vì query ban đầu chưa chứa thực thể chỉ xuất hiện sau bước suy luận đầu tiên.
- **Chọn sai kênh retrieval**: có câu cần exact keyword (BM25), có câu cần ngữ nghĩa (dense). Pipeline tĩnh cố định một kênh.
- **Over/under-retrieval**: lấy quá ít thì thiếu, quá nhiều thì nhiễu và tốn token.

Đồng thời, cuộc tranh luận **long-context vs RAG** (benchmark LaRA, ICML 2025) kết luận: không cái nào là thuốc tiên — RAG vẫn thắng khi corpus > 2M token, cần freshness, cần attribution, và rẻ hơn 8–82×. Nên giải pháp không phải "bỏ retrieval" mà là "điều phối retrieval thông minh hơn". Đó chính là agentic RAG.

Câu hỏi nghiên cứu: *làm sao để model tự điều phối truy hồi nhiều bước, đa công cụ, mà vẫn hiệu quả về số token?*

---

## 2. Đóng góp Chính

A-RAG (2026) và dòng reasoning-RAG đóng góp:

- **Hierarchical retrieval interfaces**: thay vì một retriever, expose nhiều *tool* cho model — keyword search, semantic search, và chunk-read (đọc ở granularity khác). Model tự chọn.
- **Agent tự quyết, không workflow cứng**: bỏ các luồng định trước (như "luôn rewrite rồi retrieve rồi rerank"); để chính LLM lập kế hoạch truy hồi.
- **Hiệu quả token**: A-RAG báo cáo thắng baseline với số token retrieve *ngang hoặc thấp hơn*, và scale theo năng lực model nền.

Dòng song song dùng RL (Search-R1, AutoSearch) bổ sung: huấn luyện policy để biết *khi nào dừng* (AutoSearch học "độ sâu tìm kiếm tối thiểu đủ", chống over-search).

---

## 3. Phương pháp Chi tiết

### Vòng lặp agent (training-free)

```
state: question, evidence=[], trace=[]
loop step = 0..max_steps:
    action = POLICY(question, evidence, trace)      # LLM emit JSON
    if action == ANSWER: return answer
    results = TOOLS[action.tool](action.query, k)   # keyword|semantic|hybrid|chunk_read
    evidence += dedup(results)
    trace.append(step, tool, query, #new_evidence)
# hết bước -> ép trả lời từ evidence đã có
```

Mỗi bước, policy nhận trạng thái hiện tại (câu hỏi + bằng chứng đã gom + lịch sử) và phát ra **một** JSON action: hoặc `search` (chọn tool + query), hoặc `answer`.

### Vì sao training-free vẫn hợp lý

Search-R1/AutoSearch huấn luyện policy bằng PPO để tối ưu *khi nào* và *bao nhiêu* bước. Nhưng RL cần hạ tầng train + reward + rollout — ngoài tầm một lab evaluation-first. A-RAG cho thấy **chỉ cần prompt** một model đủ mạnh là đã có hành vi agentic hữu ích. Ta mượn *ý tưởng dừng thích nghi* của AutoSearch (dừng khi đủ tự tin trả lời) nhưng thực thi bằng prompt + guard `max_steps`, không phải reward học được.

### Guard bắt buộc

Agent loop không có cap dễ "spin" vô hạn hoặc đốt cost. Hai guard tối thiểu: `max_steps` (số bước) và cap `max_evidence`. Đồng thời emit **structured trace** để phân tích thất bại — đúng tiêu chí Wave 4/đa bước của roadmap repo.

---

## 4. Thực nghiệm và Kết quả (từ các paper)

**Datasets**: NQ, TriviaQA, PopQA (single-hop); HotpotQA, 2WikiMultiHopQA, Bamboogle (multi-hop).

| Hướng | Kết quả tiêu biểu |
|---|---|
| Search-R1 (RL) | +41% (Qwen2.5-7B) so với RAG baseline |
| AutoSearch (RL) | EM cao hơn Search-R1/StepSearch với *ít bước hơn* (giảm over-search) |
| A-RAG (inference-time) | Thắng baseline với token retrieve ngang/thấp hơn, scale theo model size |

Điểm chung: lợi ích lớn nhất ở **multi-hop**, nơi truy hồi một-lượt yếu nhất.

---

## 5. Phân tích Phê bình

**Assumption ẩn**: agentic RAG giả định model nền đủ giỏi tool-use và biết tự đánh giá "đủ bằng chứng chưa". Model yếu sẽ chọn tool kém và dừng sai lúc — lợi ích biến mất.

**Limitation thừa nhận**: chi phí/độ trễ tăng theo số bước (nhiều lần gọi LLM). AutoSearch ra đời chính để giảm over-search này.

**Limitation không nêu rõ**: với câu hỏi single-fact đơn giản, agent loop *chậm hơn và đắt hơn* mà không tốt hơn một lượt hybrid — cần một router quyết định "câu này có cần agentic không" (đây là cầu nối tới Adaptive-RAG).

**Reproducibility**: A-RAG public code; các bản RL (Search-R1) cần GPU + pipeline train, khó tái lập trong môi trường nhẹ.

---

## 6. Vị trí trong Landscape

| Method | Điều phối retrieval | Cần train? | Khi nên dùng |
|---|---|---|---|
| Hybrid + rerank (một lượt) | Cố định | Không | Câu hỏi đơn giản, latency thấp |
| Adaptive-RAG | Router theo độ phức tạp | Nhẹ/không | Mix đơn giản + phức tạp |
| CRAG | Sửa khi evidence kém | Không (eval-based) | Khi retrieval hay trượt |
| **A-RAG (agentic, inference)** | LLM tự quyết đa bước, đa tool | **Không** | Multi-hop, cần trace |
| Search-R1 / AutoSearch | Policy học bằng RL | **Có (PPO)** | Có hạ tầng train, cần tối ưu sâu |

Agentic RAG inference-time là *điểm ngọt* cho lab không train: lấy phần lớn lợi ích của reasoning-retrieval mà không cần RL. Nó cũng là **lớp orchestrator chung** mà CRAG/Adaptive-RAG/IRCoT/ReAct có thể tái dùng bằng cách thay policy.

---

## 7. Takeaway

1. **RAG không chết — control flow của nó được "agent hóa"**: từ pipeline tĩnh sang vòng lặp reasoning ↔ retrieval do LLM điều khiển.
2. **Phần lớn giá trị đạt được không cần RL**: A-RAG cho thấy prompt + tool + guard đã đủ; RL (Search-R1/AutoSearch) là lớp tối ưu thêm, không phải điều kiện cần.
3. **Hạ tầng retrieval cũ trở thành toolset của agent**: BM25, dense, RRF hybrid, contextual index không bị thay thế — chúng là các *tool* mà agent gọi. Đầu tư vào retriever tốt vẫn sinh lời trong kỷ nguyên agentic.

---

## Nguồn

- A-RAG (2026). *Scaling Agentic RAG via Hierarchical Retrieval Interfaces*. <https://arxiv.org/abs/2602.03442>
- Reasoning RAG via System 1 or System 2: A Survey (2025). <https://arxiv.org/abs/2506.10408>
- Search-R1 (2025). <https://arxiv.org/abs/2503.09516>
- AutoSearch (2026). <https://arxiv.org/abs/2604.17337>
- LaRA benchmark (RAG vs long-context, ICML 2025).
- Liên quan trong repo: [`bm25_hybrid_rerank`](./bm25_hybrid_rerank.md), [`crag_2024`](./crag_2024.md), [`adaptive_rag_2024`](./adaptive_rag_2024.md), [`ircot_2022`](./ircot_2022.md)
