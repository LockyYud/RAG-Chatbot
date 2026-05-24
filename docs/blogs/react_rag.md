# `react_rag`: Synergizing Reasoning và Acting trong Language Models

> **Paper**: ReAct: Synergizing Reasoning and Acting in Language Models
> **Tác giả**: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
> **Venue**: ICLR 2023 — arXiv:2210.03629
> **Loại**: `paper_inspired`

ReAct đề xuất một paradigm mới: thay vì tách biệt reasoning (Chain-of-Thought) và acting (gọi tool/API), model được nhắc tạo ra **reasoning traces và actions xen kẽ nhau** trong cùng một trajectory. Điều này cho phép model vừa lập luận rõ ràng, vừa căn cứ vào thông tin thực từ môi trường bên ngoài — tạo ra hành vi agent vừa có khả năng giải thích vừa được kiểm chứng thực tế.

---

## 1. Bối cảnh và Động lực

Trước ReAct, hai hướng nghiên cứu chính tồn tại song song nhưng độc lập:

**Reasoning thuần túy (Chain-of-Thought)**: Wei et al. (2022) chứng minh rằng few-shot prompting với chuỗi suy luận trung gian cải thiện đáng kể khả năng giải bài toán đa bước. Tuy nhiên, CoT hoạt động hoàn toàn trong không gian nội tại của model — không có cơ chế nào để truy cập thông tin bên ngoài hay kiểm tra giả thuyết với dữ liệu thực tế. Hậu quả là model dễ **hallucinate facts** khi phải trả lời câu hỏi đòi hỏi kiến thức cụ thể.

**Acting thuần túy (API/tool calls)**: Các hệ thống action-only (như WebGPT, SayCan) cho phép model tương tác với môi trường nhưng thiếu cơ chế lập luận trung gian tường minh. Không có reasoning trace, model khó xử lý các tình huống phức tạp đòi hỏi nhiều bước lập kế hoạch, và kết quả trở nên **brittle** trước các observation bất ngờ.

**Câu hỏi nghiên cứu trung tâm**: Liệu một LLM đơn lẻ có thể synergize cả hai khả năng — reasoning và acting — trong một unified framework, thu được lợi ích của cả hai mà không chịu hạn chế của từng loại riêng lẻ?

---

## 2. Đóng góp Chính

ReAct mang lại ba đóng góp cốt lõi:

1. **Thought-Action-Observation format**: Định nghĩa một prompt format thống nhất, trong đó model xen kẽ *thought* (suy luận nội tại), *action* (tương tác với môi trường), và *observation* (kết quả trả về) trong cùng một chuỗi token. Format này không yêu cầu fine-tuning — chỉ cần few-shot demonstration.

2. **Chứng minh synergy**: Thực nghiệm cho thấy việc kết hợp reasoning và acting không chỉ cộng thêm lợi ích mà còn tạo ra tương tác tích cực: reasoning giúp model diễn giải observation và lên kế hoạch action tiếp theo; action giúp reasoning được căn cứ vào thông tin thực tế, giảm hallucination.

3. **Đánh giá trên hai loại task**: ReAct được đánh giá trên cả **knowledge-intensive QA tasks** (HotpotQA, FEVER) — đòi hỏi truy xuất thông tin — và **decision-making tasks** (AlfWorld, WebShop) — đòi hỏi tương tác tuần tự với môi trường. Sự bao quát này chứng minh tính tổng quát của framework.

---

## 3. Phương pháp Chi tiết

### ReAct Format: Thought → Action → Observation

ReAct hoạt động bằng cách nhắc model tạo ra các bước xen kẽ trong một trajectory duy nhất. Mỗi bước gồm:

- **Thought**: Model diễn đạt tường minh suy luận của mình — phân tích observation vừa nhận, lên kế hoạch action tiếp theo, hoặc tổng hợp thông tin đã thu thập.
- **Action**: Model thực hiện một hành động có cấu trúc từ action space được định nghĩa sẵn (ví dụ: `search[entity]`, `lookup[keyword]`, `finish[answer]`).
- **Observation**: Môi trường hoặc tool trả về kết quả, được nối vào context để model tiếp tục.

### Ví dụ Trace Thực Tế (HotpotQA)

Câu hỏi: *"Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to work with?"*

```
Thought 1: I need to find what program Apple Remote was originally designed to work with,
           then find other devices that can control it.
Action 1: search[Apple Remote]
Observation 1: The Apple Remote is a remote control device designed by Apple Inc. It was
               originally designed to control Front Row media center software ...

Thought 2: Front Row is the program. Now I need to find what other devices can control
           Front Row besides the Apple Remote.
Action 2: search[Front Row software]
Observation 2: Front Row is a media center software application for Mac OS X ... it can be
               controlled with the Apple Remote or a keyboard ...

Thought 3: The keyboard can also control Front Row. So the answer is keyboard.
Action 3: finish[keyboard]
```

### Few-Shot Prompting

ReAct không yêu cầu fine-tuning. Các demonstration được xây dựng thủ công: mỗi example là một chuỗi Thought/Action/Observation hoàn chỉnh giải quyết một câu hỏi mẫu. Với HotpotQA, tác giả xây dựng 6 demonstration examples; với FEVER, 3 examples. Các demonstration này được prepend vào mỗi query lúc inference.

### Action Space

Với knowledge-intensive tasks (HotpotQA, FEVER), action space gồm:
- `search[entity]`: Tìm kiếm Wikipedia về entity.
- `lookup[keyword]`: Tra cứu keyword trong trang Wikipedia hiện tại.
- `finish[answer]`: Kết thúc và trả lời.

Với decision-making tasks (AlfWorld), action space là tập hợp các ngôn ngữ tự nhiên hành động như `go to desk 1`, `pick up book 2`, `put book 2 in/on shelf 1`.

### Tại Sao Thought Trước Action?

Thought buộc model phải **tường minh hóa kế hoạch** trước khi hành động. Điều này có hai tác dụng:
1. Action được căn cứ vào reasoning rõ ràng, giảm khả năng gọi action ngẫu nhiên.
2. Khi observation trả về kết quả bất ngờ, thought tiếp theo có thể điều chỉnh kế hoạch thay vì tiếp tục một cách mù quáng.

**Contrast với CoT thuần túy**: CoT không có cơ chế lấy thông tin từ bên ngoài — toàn bộ kiến thức phải có sẵn trong model, dẫn đến hallucination với câu hỏi thực tế.

**Contrast với Act-only**: Không có reasoning trace, model không thể lập kế hoạch đa bước hay diễn giải observation phức tạp — behavior trở nên brittle.

---

## 4. Thực nghiệm và Kết quả

### Knowledge-Intensive Tasks

**HotpotQA** (multi-hop question answering, metric: Exact Match):

| Phương pháp | EM (%) |
|---|---|
| Standard prompting | 29.7 |
| CoT (Chain-of-Thought) | 35.1 |
| Act-only | 28.0 |
| **ReAct** | **35.1** |
| ReAct → CoT (best of both) | **40.4** |

**FEVER** (fact verification, metric: Accuracy):

| Phương pháp | Accuracy (%) |
|---|---|
| Standard prompting | 56.3 |
| CoT | 56.3 |
| Act-only | 58.9 |
| **ReAct** | **63.1** |
| ReAct → CoT (best of both) | **66.6** |

Trên FEVER, ReAct vượt trội rõ rệt so với CoT vì task yêu cầu verification dựa trên evidence thực tế — không thể suy luận thuần túy.

### Decision-Making Tasks

**AlfWorld** (text-based household environment, metric: Success Rate):

| Phương pháp | Success Rate (%) |
|---|---|
| Act-only (BUTLER) | 45 |
| ReAct (1-shot) | 71 |
| **ReAct (2-shot)** | **71** |

ReAct đạt 71% success rate so với 45% của Act-only baseline BUTLER — cải thiện 26 điểm phần trăm tuyệt đối. Điều này cho thấy reasoning traces giúp model lập kế hoạch tốt hơn trong môi trường nhiều bước.

**WebShop** (e-commerce shopping environment, metric: Average Reward):

| Phương pháp | Avg. Reward | Success Rate (%) |
|---|---|---|
| IL + RL (imitation + reinforcement learning) | 62.4 | 28.7 |
| Act-only | 62.2 | 30.1 |
| **ReAct** | **66.6** | **40.0** |

ReAct đạt average reward 66.6 và success rate 40.0% — vượt cả Act-only lẫn hệ thống IL+RL được train chuyên biệt, chỉ bằng few-shot prompting.

### Key Findings

- Trên **knowledge tasks**: ReAct cải thiện interpretability và giảm hallucination; combination ReAct → CoT (fallback) đạt kết quả tốt nhất.
- Trên **decision-making tasks**: Reasoning traces quan trọng hơn — ReAct vượt Act-only trên cả AlfWorld và WebShop.
- **Human interpretability**: Evaluator người đánh giá ReAct traces dễ theo dõi và debug hơn hẳn Act-only traces.

---

## 5. Phân tích Phê bình

**Chi phí few-shot prompting**: Mỗi demonstration là một trajectory dài — với HotpotQA, 6 full examples được prepend vào mỗi query. Điều này tiêu tốn nhiều token và tăng chi phí inference đáng kể. Với các LLM có context window ngắn (thời điểm 2022), đây là hạn chế thực tế.

**Action space cố định và pre-defined**: ReAct giả định action space được thiết kế sẵn cho từng domain. Generalization sang môi trường mới đòi hỏi viết lại demonstration và định nghĩa lại action space — không tự động. Điều này hạn chế tính plug-and-play.

**Hallucination vẫn tồn tại trong Thought steps**: Mặc dù observation từ bên ngoài giảm hallucination tổng thể, các thought step vẫn có thể chứa reasoning sai dẫn đến action không phù hợp. Tác giả nhận diện đây là lỗi phổ biến nhất trong analysis định tính.

**Repetitive loops**: Một failure mode thường gặp là model lặp đi lặp lại cùng một action (ví dụ: `search` cùng query nhiều lần) khi không tìm được thông tin cần thiết. Không có cơ chế thoát vòng lặp nội tại — cần guard bên ngoài.

**Môi trường đánh giá synthetic**: AlfWorld và WebShop là môi trường mô phỏng với action space có kiểm soát. Transfer sang real-world settings (web thực, hệ thống thực) chưa được đánh giá, và có thể gặp vấn đề với ambiguity và noise thực tế.

**Context length**: Trajectory dài (nhiều Thought/Action/Observation steps) có thể vượt context limit, đặc biệt với các task phức tạp hoặc model có window ngắn. ReAct không đề xuất giải pháp cho vấn đề này.

---

## 6. Vị trí trong Landscape

### So sánh với các phương pháp liên quan

| Phương pháp | Reasoning | External Actions | Multi-hop | Tool Generality |
|---|---|---|---|---|
| Standard RAG | Không | Retrieval (1 lần) | Không | Thấp |
| Chain-of-Thought (CoT) | Tường minh | Không | Có | Không áp dụng |
| Self-Ask | Có (sub-questions) | Search | Có | Thấp |
| Toolformer | Không | Nhiều API | Không | Trung bình |
| IRCoT | Có (CoT + retrieval) | Retrieval | Có | Thấp |
| **ReAct** | **Tường minh** | **Nhiều tool** | **Có** | **Cao** |

### Khi nào nên dùng ReAct-style Agentic RAG?

**Phù hợp khi**:
- Câu hỏi đòi hỏi nhiều bước retrieval với reasoning trung gian (multi-hop).
- Cần interpretability — trace rõ ràng để debug và audit.
- Có nhiều loại tool/action cần phối hợp.
- Câu hỏi yêu cầu kiểm tra giả thuyết hoặc verification.

**Không cần thiết (overkill) khi**:
- Single-hop QA với top-k retrieval đơn giản là đủ.
- Latency và cost là ưu tiên hàng đầu.
- Action space chỉ có một loại (pure retrieval) — IRCoT đơn giản hơn và đủ dùng.
- Môi trường không cung cấp structured feedback (observation chất lượng thấp).

---

## 7. Takeaway

1. **Interleaving là chìa khóa, không phải kết hợp tuần tự**: ReAct không phải là "CoT rồi action" hay "action rồi CoT". Giá trị thực sự đến từ việc xen kẽ — mỗi thought được inform bởi observation vừa nhận, mỗi action được căn cứ bởi thought vừa viết. Tách hai bước ra sẽ mất đi synergy này.

2. **Thought step là interface để debug, không chỉ là internal state**: Khác với CoT thuần túy (có thể là private reasoning), thought steps trong ReAct phục vụ mục đích kép: giúp model lập kế hoạch tốt hơn VÀ cung cấp human-readable trace để kiểm tra hành vi agent. Đây là lợi thế quan trọng cho production systems.

3. **Failure modes cần guard bên ngoài**: ReAct không tự giải quyết repetitive loops, excessive tool calls hay context overflow. Bất kỳ triển khai production nào cần bổ sung: `max_steps` guard, loop detection (kiểm tra repeated action/query), và fallback strategy khi observation liên tiếp không hữu ích. Framework mạnh nhất là ReAct + external scaffolding, không phải ReAct standalone.

---

## Nguồn

- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). **ReAct: Synergizing Reasoning and Acting in Language Models**. *ICLR 2023*. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Wei, J. et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *NeurIPS 2022*.
- Trivedi, H. et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT). *ACL 2023*.
- Shridhar, M. et al. (2020). ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. *ICLR 2021*.
- Yao, S. et al. (2022). WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. *NeurIPS 2022*.
