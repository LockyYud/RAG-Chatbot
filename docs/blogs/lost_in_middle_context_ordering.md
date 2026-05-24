# `lost_in_middle_context_ordering`: Position Bias trong Long-Context LLM và Hậu Quả với RAG

> **Paper**: Lost in the Middle: How Language Models Use Long Contexts
> **Tác giả**: Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
> **Venue**: TACL 2024 — arXiv:2307.03172
> **Loại**: `paper_inspired`

Các language model có khả năng xử lý context dài không sử dụng đều toàn bộ thông tin trong đó — performance giảm rõ rệt khi thông tin quan trọng nằm ở giữa context so với khi nằm ở đầu hoặc cuối. Hiện tượng này tạo ra một đường cong U-shaped trong accuracy theo vị trí, và nó xuất hiện nhất quán trên nhiều model, kể cả các model được quảng cáo là có extended context window.

---

## 1. Bối cảnh và Động lực

Giả định phổ biến trong cộng đồng NLP và RAG là: context window dài hơn đồng nghĩa với khả năng reasoning tốt hơn. Khi các model như GPT-3.5 Turbo, Claude, và các biến thể mở rộng window lên 16k, 32k, thậm chí 100k token, kỳ vọng là retrieval-augmented generation sẽ được hưởng lợi trực tiếp từ việc đưa nhiều document hơn vào prompt.

Tuy nhiên, Liu et al. quan sát thấy một vấn đề tinh vi hơn: các chỉ số retrieval (Recall@k, MRR) không phản ánh đúng chất lượng generation cuối cùng. Một pipeline có thể truy xuất đúng passage, nhưng nếu passage đó bị đặt ở vị trí bất lợi trong context, model vẫn trả lời sai. Retrieval metrics trông ổn, generation metrics lại không.

Câu hỏi nghiên cứu cụ thể là: *vị trí của document liên quan trong input context ảnh hưởng như thế nào đến accuracy của câu trả lời?* Đây là câu hỏi về context utilization, không phải về retrieval quality.

---

## 2. Đóng góp Chính

**Phát hiện thực nghiệm về U-shaped performance**: Paper cung cấp bằng chứng định lượng rõ ràng rằng LLM không sử dụng context đồng đều. Performance cao ở đầu và cuối, thấp nhất ở giữa — một U-shape nhất quán qua nhiều model và nhiều cấu hình.

**Framework hai task để kiểm soát biến**: Tác giả thiết kế hai task độc lập — multi-document QA và key-value retrieval — để tách biệt hiệu ứng vị trí khỏi các yếu tố nhiễu như độ khó câu hỏi, chất lượng document, hay sự đa dạng của corpus.

**Hệ quả trực tiếp cho RAG system design**: Paper chỉ ra rằng thứ tự đặt document trong prompt là một trục tối ưu riêng, độc lập với retrieval quality. Điều này có nghĩa là bất kỳ RAG system nào cũng nên xem xét context ordering như một thành phần thiết kế chứ không phải implementation detail.

---

## 3. Phương pháp Chi tiết

### Task 1: Multi-Document Question Answering

Tác giả dùng dataset NaturalQuestions (open-domain QA). Với mỗi câu hỏi, họ tạo một context gồm nhiều document, trong đó có đúng một "gold document" chứa câu trả lời đúng. Các document còn lại là distractor — những document liên quan về chủ đề nhưng không chứa câu trả lời.

Biến thực nghiệm duy nhất bị thay đổi là *vị trí của gold document* trong context. Vị trí được index từ 0 (đầu tiên) đến k-1 (cuối cùng), với k là tổng số document. Tất cả yếu tố khác — câu hỏi, nội dung document, prompt template — được giữ nguyên.

Số document thử nghiệm: 10 và 20 document. Metric đo lường: exact match accuracy với gold answer.

### Task 2: Key-Value Retrieval

Đây là synthetic task được thiết kế để loại bỏ hoàn toàn yếu tố ngữ nghĩa. Context là một JSON-like object gồm nhiều cặp key-value ngẫu nhiên (UUID strings). Model được yêu cầu trả về value tương ứng với một key cho trước.

Task này quan trọng vì nó kiểm soát chặt hơn: không có ambiguity về nội dung, không có distractor "gần đúng", không có prior knowledge từ pretraining. Nếu U-shape vẫn xuất hiện ở đây, nguyên nhân phải nằm ở cơ chế xử lý vị trí của model, không phải ở sự khó khăn ngữ nghĩa.

Số cặp key-value: 75, 140 cặp (tương ứng với context dài hơn).

### Các Models Được Kiểm Tra

| Model | Context Window | Ghi chú |
|---|---|---|
| GPT-3.5-Turbo | 4k token | OpenAI |
| GPT-3.5-Turbo (16k) | 16k token | Extended window version |
| Claude 1.3 | 8k token | Anthropic |
| Claude 1.3 (100k) | 100k token | Extended context version |
| LongChat-13B | 16k token | Open-source, fine-tuned long context |
| MPT-30B-Instruct | 8k token | Open-source |
| Text-Davinci-003 | 4k token | OpenAI (instruction-tuned) |

Việc bao gồm cả Claude 100k và GPT-3.5-Turbo 16k cho phép tác giả kiểm tra liệu các model được marketing là "long-context capable" có thực sự thoát khỏi hiệu ứng vị trí không.

---

## 4. Thực nghiệm và Kết quả

### U-Shaped Curve: Multi-Document QA

Với GPT-3.5-Turbo trên NaturalQuestions, 10 documents trong context:

| Vị trí Gold Document | Accuracy |
|---|---|
| Vị trí 0 (đầu tiên) | ~71% |
| Vị trí 4–5 (giữa) | ~45–50% |
| Vị trí 9 (cuối cùng) | ~65–68% |

Hiệu ứng rõ ràng: accuracy ở giữa thấp hơn khoảng 20–25 percentage point so với đầu. Đây không phải biến động nhỏ — đây là degradation nghiêm trọng với một hệ thống production.

Khi tăng lên 20 documents, U-shape sâu hơn: accuracy ở giữa giảm xuống còn khoảng 35–40%, trong khi đầu vẫn giữ ở ~70%.

### U-Shaped Curve: Key-Value Retrieval

Task key-value retrieval cho thấy U-shape sắc nét hơn, vì task này đơn giản hơn về mặt ngữ nghĩa:

- **GPT-3.5-Turbo** với 75 cặp key-value: accuracy ~100% ở vị trí đầu và cuối, giảm xuống còn **~60%** ở giữa.
- Với 140 cặp, hiệu ứng tệ hơn: accuracy ở giữa chỉ còn khoảng **~40%** trên một số model.

Điều này đặc biệt đáng lo ngại vì key-value retrieval về lý thuyết là trivial — nếu model đọc toàn bộ context, nó không thể sai. Sự degradation chứng minh rằng model thực sự không chú ý đồng đều.

### Extended Context Models: Không Miễn Nhiễm

**Claude 1.3 (100k context window)**: U-shape vẫn rõ ràng. Model này thậm chí thể hiện recency bias mạnh hơn primacy bias ở một số configuration — accuracy cuối context cao bất thường, nhưng accuracy giữa vẫn thấp.

**GPT-3.5-Turbo (16k)**: Hiệu ứng vẫn có, mặc dù recency bias rõ hơn so với phiên bản 4k. Model có xu hướng dùng thông tin ở cuối tốt hơn đầu trong một số cấu hình.

**LongChat-13B**: Open-source model với long context fine-tuning cũng không thoát khỏi U-shape, thậm chí có performance thấp hơn ở middle positions so với các OpenAI model.

### Điểm Quan Trọng Nhất

Không có model nào trong số được kiểm tra thể hiện performance phẳng (flat) theo vị trí. U-shape hoặc một biến thể của nó (J-shape với recency bias mạnh hơn primacy bias, hoặc ngược lại) xuất hiện ở toàn bộ experimental setup. Extended context window không giải quyết vấn đề — nó đôi khi làm vấn đề tệ hơn vì context dài hơn đẩy middle positions xa hơn khỏi cả hai đầu.

---

## 5. Phân tích Phê bình

**Giới hạn về ecological validity**: Cả hai task đều dùng controlled setup, trong đó mỗi context chứa đúng một gold document. Trong RAG production thực tế, nhiều document có thể cùng chứa thông tin liên quan ở mức độ khác nhau, và "vị trí của evidence" không rõ ràng như trong thí nghiệm. Kết quả có thể khó transfer trực tiếp.

**Primacy và recency bias đã biết trong cognition**: U-shaped serial position effect là hiện tượng nổi tiếng trong tâm lý học nhận thức con người (Murdock 1962). Câu hỏi thực sự không phải là "tại sao LLM có bias này?" mà là "tại sao chúng ta kỳ vọng LLM không có?" Paper không trả lời câu hỏi cơ chế một cách sâu sắc.

**Cơ chế chưa được giải thích đầy đủ**: Liu et al. không phân tích attention patterns hay positional encoding để giải thích mechanistically tại sao middle positions bị bất lợi. Câu hỏi liệu vấn đề nằm ở sinusoidal/RoPE positional encoding, attention sink, hay training data distribution vẫn còn mở.

**Fix (reordering) rẻ nhưng không triệt để**: Đặt relevant document lên đầu là một heuristic hiệu quả, nhưng nó giả định ta biết document nào relevant nhất trước khi generation. Trong RAG, retrieval score là proxy imperfect cho relevance. Và việc fix vị trí không giải quyết câu hỏi tại sao model fail ở middle.

**Tính bền vững qua RLHF và instruction tuning**: Paper không kiểm tra liệu instruction tuning hay RLHF có giảm thiểu U-shape không. Các model hiện tại (GPT-4, Claude 2+) có thể đã được fine-tune theo cách giảm bớt bias này, nhưng không có bằng chứng trong paper.

---

## 6. Vị trí trong Landscape

**Ảnh hưởng toàn diện lên RAG**: Khác với các paper tối ưu một thành phần cụ thể (better embedding, better chunking), phát hiện của Liu et al. ảnh hưởng đến bất kỳ RAG pipeline nào sử dụng multi-document context. Không có retriever nào — dù BM25, dense retrieval, hay hybrid — có thể tự mình giải quyết vấn đề này.

**Liên hệ với RAG-Fusion và RRF**: RAG-Fusion (Rackauckas 2024) dùng Reciprocal Rank Fusion để merge kết quả từ nhiều query. Thứ tự document trong context sau khi fusion vẫn bị ảnh hưởng bởi U-shape. RRF scoring quyết định relevance ranking, nhưng position-aware ordering phải được áp dụng sau đó.

**Liên hệ với RAPTOR và hierarchical retrieval**: RAPTOR (Sarthi et al. 2024) xây dựng cây phân cấp để giảm số lượng document trong mỗi context. Một tác dụng phụ tích cực: ít document hơn trong context có thể giảm thiểu U-shape effect vì khoảng cách từ center đến edges ngắn hơn.

**Diagnostic vs. Solution**: Paper này là một *diagnostic* — nó đo lường và định lượng một vấn đề. Nó không đề xuất kiến trúc mới. Các paper theo hướng giải quyết vấn đề này bao gồm: Focused Transformer (tăng độ phân giải attention ở middle), context compression (giảm số token trước khi đưa vào LLM), và các phương pháp fine-tuning chuyên biệt cho long-context utilization.

---

## 7. Takeaway

**Context ordering là một design decision, không phải implementation detail**: Trong bất kỳ RAG system nào, thứ tự đặt retrieved document vào prompt ảnh hưởng trực tiếp đến generation quality. Đặt document có relevance score cao nhất ở đầu (và document có score cao thứ hai ở cuối) là một heuristic có cơ sở thực nghiệm từ paper này.

**Retrieval metrics và generation metrics đo hai thứ khác nhau**: Recall@k tốt không đảm bảo answer quality tốt. Nếu một RAG pipeline chỉ được đánh giá qua retrieval metrics, nó có thể miss toàn bộ lớp lỗi liên quan đến context utilization. Full-pipeline evaluation (answer correctness, faithfulness) là cần thiết để phát hiện loại lỗi này.

**Longer context window ≠ better context utilization**: Tăng context window từ 4k lên 100k không giải quyết vấn đề. Context dài hơn thậm chí có thể làm middle positions "bị lãng quên" nhiều hơn vì chúng ở xa cả hai cực hơn. Đây là lý do tại sao context compression và selective retrieval (lấy ít document hơn nhưng chính xác hơn) thường outperform naive "cho tất cả vào context".

---

## Nguồn

- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the Middle: How Language Models Use Long Contexts. *Transactions of the Association for Computational Linguistics (TACL)*, 12, 157–173. https://arxiv.org/abs/2307.03172
