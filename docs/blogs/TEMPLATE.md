# Academic Blog Template — RAG Paper Series

Template này dùng cho tất cả blog trong series. Mục tiêu: phân tích paper học thuật nghiêm túc,
không phải hướng dẫn implementation.

---

# `[technique_id]`: [Tiêu đề mô tả đóng góp chính của paper]

> **Paper**: [Tên đầy đủ của paper]
> **Tác giả**: [Tên tác giả chính] et al.
> **Venue**: [Conference/Journal] [Năm] — [arXiv ID hoặc link]
> **Loại**: `paper_reproduction` | `paper_inspired` | `production_pattern`

[1-2 câu tóm tắt cốt lõi của paper — không phải abstract copy, mà là insight chính
theo góc nhìn của người đọc đã quen với RAG.]

---

## 1. Bối cảnh và Động lực

[Phần này trả lời: prior work bị giới hạn ở đâu, và gap nào paper này lấp đầy.]

[Mô tả trạng thái trước paper: hệ thống RAG standard làm gì, và fail ở đâu cụ thể.
Nên có ví dụ cụ thể về loại câu hỏi hoặc corpus khiến prior work thất bại.]

[Kết phần này bằng câu research question mà paper đặt ra.]

---

## 2. Đóng góp Chính

[Liệt kê 2-4 contribution cụ thể — đây là những gì tác giả claim là novel.]

- **[Tên contribution 1]**: [Mô tả kỹ thuật ngắn gọn, 1-2 câu]
- **[Tên contribution 2]**: [...]
- **[Tên contribution 3]**: [...]

[Phân biệt rõ: đây là claim của tác giả, chưa phải đánh giá của người viết blog.]

---

## 3. Phương pháp Chi tiết

[Đây là phần quan trọng nhất. Phải đủ chi tiết để reader hiểu cơ chế, không chỉ ý tưởng.]

### [Tên component/module chính]

[Giải thích thuật toán với notation cụ thể. Nếu paper có công thức, trích nguyên hoặc
paraphrase chính xác. Nếu không có công thức, dùng pseudocode.]

```
[Pseudocode hoặc ASCII diagram mô tả flow của method]
```

[Với mỗi design choice quan trọng, giải thích tại sao tác giả chọn thế này thay vì
alternative hiển nhiên. Đây là nơi blog tạo ra giá trị — không chỉ là "cái gì" mà là "tại sao".]

### [Component tiếp theo nếu cần]

[...]

---

## 4. Thực nghiệm và Kết quả

[Phần này phải có số liệu cụ thể từ paper. Không có số = không phải academic writing.]

**Datasets**: [Tên dataset, kích thước, domain]

**Baselines**: [Tên baseline, brief description]

**Metrics**: [Tên metric và ý nghĩa]

### Kết quả Chính

| Method | [Dataset 1] | [Dataset 2] | [Dataset 3] |
|--------|------------|------------|------------|
| [Baseline 1] | [số] | [số] | [số] |
| [Baseline 2] | [số] | [số] | [số] |
| **[Method của paper]** | **[số]** | **[số]** | **[số]** |

[Giải thích kết quả: tăng bao nhiêu điểm, trên dataset nào là strongest/weakest, và tại sao.]

### Ablation Study

[Mô tả ablation quan trọng nhất của paper: component nào đóng góp bao nhiêu vào kết quả.
Nếu paper có bảng ablation, tóm tắt hoặc trích lại.]

---

## 5. Phân tích Phê bình

[Đây là điểm phân biệt blog học thuật vs. blog summary. Không viết "paper này tốt vì..."
mà viết theo góc nhìn phân tích.]

**Assumption ẩn**:
[Các assumption mà paper dựa vào nhưng không phát biểu tường minh. Ví dụ: "paper assume
retriever đã có recall đủ cao", hoặc "paper assume query luôn well-formed".]

**Limitation mà paper thừa nhận**:
[Những gì tác giả tự nêu trong paper — thường ở conclusion hoặc limitation section.]

**Limitation mà paper không thấy**:
[Những gì người đọc có thể nhận ra nhưng paper không đề cập. Đây là phân tích của người
viết blog, cần được đánh dấu rõ là nhận xét, không phải claim của paper.]

**Reproducibility**:
[Code có public không? Dataset có accessible không? Có điều gì trong setup khó replicate?]

---

## 6. Vị trí trong Landscape

[So sánh paper này với 2-3 paper liên quan gần nhất. Mục tiêu: giúp reader biết khi nào
dùng paper này thay vì alternative.]

| Method | Ý tưởng lõi | Khi nên dùng | Hạn chế chính |
|--------|------------|--------------|---------------|
| [Paper này] | [...] | [...] | [...] |
| [Alternative 1] | [...] | [...] | [...] |
| [Alternative 2] | [...] | [...] | [...] |

[Kết phần này bằng 1-2 câu về vị trí của paper trong trajectory của field: nó mở ra
hướng gì, hoặc đóng lại câu hỏi gì?]

---

## 7. Takeaway

[3 điều reader nên nhớ sau khi đọc. Không phải "kết luận" chung chung — phải là insight
cụ thể có thể apply vào suy nghĩ về RAG.]

1. **[Insight 1]**: [1-2 câu giải thích]
2. **[Insight 2]**: [1-2 câu giải thích]
3. **[Insight 3]**: [1-2 câu giải thích]

---

## Nguồn

- [Citation chính của paper]
- [Các paper liên quan được nhắc đến]
- [Code/dataset link nếu có]
