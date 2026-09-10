# Synthetic Benchmark v2 — Thiết Kế

Tài liệu này thiết kế lại đường "người dùng đưa tài liệu thô vào, repo tự sinh bộ đánh giá rồi chấm các
technique". Đường này hiện đã tồn tại (`raglab dataset generate` rồi `raglab bench --docs --qa`), nhưng đang có
một lỗi trả về số sai và ba điểm yếu phương pháp làm kết quả không đáng tin.

Nguyên tắc xuyên suốt: **không cố làm bộ sinh đề hoàn hảo, mà làm cho benchmark tự nói được nó đáng tin tới
đâu.** Mọi bước sinh đề đều trả về một con số, và report bị gắn nhãn không đáng tin khi các con số đó dưới ngưỡng,
cùng tinh thần với `claim_eligibility` ở [benchmark_protocol_v1.md](./benchmark_protocol_v1.md).

Phạm vi: đường dữ liệu người dùng (`datasets/user_data/`). Không đụng tới fixed dataset adapter, không đụng tới
suite `claim_eligible`. Một bộ synthetic **không bao giờ** claim-eligible; nó luôn là `protocol_split: dev`.

---

## 1. Hiện trạng và vì sao phải làm lại

### 1.1 Lỗi đang trả số sai

`SyntheticQAGenerator` tự chia đoạn và đặt id dạng `doc:synthetic:0001`
([synthetic.py](../ragbench/datasets/synthetic.py), hàm `_chunk`), rồi ghi id đó vào `expected_chunk_ids`. Pipeline
lại đặt id dạng `doc:c1:<sha1 10 ký tự>` ([chunkers/common.py](../ragbench/processing/chunkers/common.py), hàm
`chunk_id`). Hai họ id không bao giờ trùng.

Ở [metrics/__init__.py](../ragbench/evaluation/metrics/__init__.py):

```python
expected = expected_chunks or expected_docs
```

Câu hỏi trả lời được luôn có cả hai trường, nên `expected_chunks` giành quyền ưu tiên, giao với tập retrieved luôn
rỗng. Recall, hit rate, nDCG, MRR, MAP bằng 0 cho mọi technique, còn `retrieval_evaluated` vẫn là `True`. Report in
ra một bảng toàn số 0 trông y hệt kết quả thật.

### 1.2 Ba điểm yếu phương pháp

| # | Vấn đề | Cơ chế | Model mạnh hơn có sửa được không |
|---|---|---|---|
| A | **Nhãn thiếu** (false negative) | Mỗi câu chỉ gắn một đoạn nguồn. Tài liệu thật lặp lại nhiều; retriever trả về đoạn khác cũng đúng thì bị chấm 0. Retriever càng tốt càng dễ bị phạt | **Không.** Lỗi nằm ở quy trình gán một đoạn, không ở chất lượng câu chữ |
| B | **Lệch từ vựng** | Câu hỏi viết từ đoạn nên mượn lại cách diễn đạt của đoạn, ưu ái BM25 và lexical | Giảm mức độ, không đổi bản chất |
| C | **Câu không trả lời được là giả** | Model chỉ thấy một đoạn nên "không trả lời được từ đoạn này" bị ghi thành "không trả lời được". Corpus có thể trả lời ở chỗ khác | **Không.** Cần nhìn toàn corpus |

Điểm chung: sai lệch **tương quan với thứ đang đo**, nên tăng số câu hỏi không làm nó loãng đi. Đây là lý do
không thể coi nó như nhiễu.

### 1.3 Vòng lặp tự chứng

`dataset generate --model` mặc định `gpt-4.1-mini`; judge trong [judge.py](../ragbench/evaluation/judge.py) cũng mặc
định `gpt-4.1-mini`. Cùng một model ra đề và chấm bài. Điểm faithfulness và correctness bị thổi, không có cảnh báo.

---

## 2. Nguyên tắc thiết kế

1. **Mỗi bước lọc trả về một tỉ lệ loại**, ghi vào manifest. Tỉ lệ loại cao là tín hiệu model sinh đề yếu, không phải
   lỗi cần giấu.
2. **Tách ba vai model**: sinh đề, kiểm chứng, chấm bài. Không vai nào được trùng model với vai khác.
3. **Nhãn ở cấp chunk của pipeline**, không phải cấp chunk riêng của generator. Cùng một chunker, cùng một hàm
   `chunk_id`.
4. **Nhãn có bậc và có nhiều đoạn**, sinh bằng pooling. Ghi vào `metadata.relevance_by_doc_id` mà metrics đã đọc.
5. **Report phải mang khối tin cậy**, và nhãn `synthetic_untrusted` không thể tắt bằng cờ CLI.
6. **Tái lập được**: cùng corpus, cùng seed, cùng cấu hình model thì ra cùng bộ. Fingerprint corpus và toàn bộ cấu
   hình đi vào manifest.

---

## 3. Pipeline sinh đề

Bảy bước, chạy tuần tự trên từng corpus. Ký hiệu vai: **G** = generator, **V** = verifier, **P** = pooler (dùng V).

### Bước 1 — Chunk bằng chunker của pipeline

- **Vào:** thư mục tài liệu (`.md`, `.markdown`, `.txt`; định dạng khác nằm ngoài phạm vi v2, xem mục 8).
- **Làm:** gọi đúng chunker mặc định của technique baseline (`naive_rag`) qua `BasePipeline`, không dùng
  `_load_text_chunks` riêng. Chunk id sinh từ `chunkers/common.chunk_id`.
- **Ra:** danh sách `Chunk` có `chunk_id`, `doc_id`, `text`.
- **Ghi:** `chunker_id`, `chunk_config`, `chunk_count`.
- **Vì sao:** sửa lỗi 1.1 tại gốc. Sinh đề phải nhìn đúng đoạn văn mà retrieval sẽ trả về, nếu không câu hỏi có
  thể được viết từ đoạn chữ không chunk nào chứa. Lưu ý chunk ở đây dùng để **sinh đề và kiểm span**, không dùng
  để đặt nhãn — xem mục 10.

### Bước 2 — Sinh câu hỏi kèm đoạn đáp án chép nguyên văn (G)

- **Vào:** một chunk.
- **Làm:** G sinh `questions_per_chunk` câu. Mỗi câu gồm `question`, `answer_span` (chép nguyên văn từ chunk),
  `ground_truth_answer`, `question_type`, `difficulty`. Ràng buộc trong prompt: câu hỏi phải tự đứng được, không
  tham chiếu "đoạn trên", "tài liệu này".
- **Lọc cơ học, không tốn LLM:** `answer_span` phải là chuỗi con của `chunk.text` sau chuẩn hoá khoảng trắng.
  Trượt thì loại. Câu `unanswerable` không cần span.
- **Ghi:** `generated_count`, `span_reject_count`.
- **Vì sao:** span chép nguyên văn là bộ lọc rẻ nhất chống model yếu bịa đáp án.

### Bước 3 — Lọc tính trả lời được (V)

- **Vào:** câu hỏi và **chỉ** chunk nguồn. Không đưa `answer_span`.
- **Làm:** V trả lời câu hỏi từ chunk. So với `answer_span` bằng token overlap (ROUGE-L ≥ 0.5) hoặc V tự đánh
  `answerable: bool` kèm đáp án; dùng cả hai, cần cả hai đồng ý.
- **Lọc:** V không trả lời được, hoặc đáp án lệch, thì loại.
- **Ghi:** `answerability_reject_count`, `answerability_reject_rate`.
- **Vì sao:** đây là thước đo trực tiếp nhất của chất lượng G. Tỉ lệ loại là số đầu tiên người đọc nhìn.

### Bước 4 — Đo trùng từ vựng, viết lại khi vượt ngưỡng (G rồi V)

- **Làm:** tính `lexical_overlap` = Jaccard trên token (sau tách từ tiếng Việt bằng tokenizer sẵn có trong
  `core/text.py`) giữa câu hỏi và chunk. Vượt `overlap_threshold` (mặc định 0.6) thì G viết lại **một lần** với yêu
  cầu không dùng lại cụm từ của nguồn, rồi **chạy lại bước 3** trên câu đã viết lại. Trượt bước 3 sau viết lại thì
  giữ bản gốc và gắn cờ `high_overlap: true`.
- **Ghi:** phân bố `lexical_overlap` (p50, p90), `rewrite_count`, `rewrite_reject_count`.
- **Vì sao:** không xoá được lệch B nhưng làm nó đo được. Viết lại một lần thôi, vì paraphrase nhiều vòng làm câu
  hỏi mất nghĩa.

### Bước 5 — Gộp nhãn bằng pooling (P)

- **Vào:** câu hỏi, corpus đã index.
- **Làm:** chạy `BM25Retriever` và `DenseRetriever` (cùng embedding model với pipeline baseline), lấy top-`pool_k`
  (mặc định 20) mỗi bên, hợp lại, bỏ chunk nguồn. Đưa **cả gói** cho P trong một lần gọi: "đoạn nào trong danh
  sách trả lời được câu hỏi này, ở mức đầy đủ hay một phần". P trả về id kèm bậc.
- **Nhãn ra:**
  - `expected_doc_ids` = tài liệu nguồn + mọi tài liệu P đánh "đầy đủ".
  - `expected_chunk_ids` = **rỗng**, luôn luôn. Chunk id gắn với chunker, nên không làm nhãn được — mục 10.
  - `metadata.relevance_by_doc_id` = `{doc_id: 2}` cho đầy đủ, `1` cho một phần. Tài liệu nguồn luôn `2`, kể cả
    khi pool rỗng.
- **Xử lý câu `unanswerable`:** nếu P đánh bất kỳ đoạn nào trong pool là "đầy đủ", câu này bị **gắn lại** thành
  answerable với nhãn từ pool, và ghi `relabelled_from_unanswerable: true`. Nếu không có, giữ unanswerable với
  `expected_*` rỗng.
- **Ghi:** `pool_size_mean`, `extra_relevant_mean` (số đoạn thêm ngoài nguồn, trung bình), `relabelled_count`.
- **Vì sao:** đây là bước duy nhất chạm được vào lỗi A, và giải quyết luôn lỗi C. Chi phí một lần gọi mỗi câu.
  `extra_relevant_mean` cao là bằng chứng corpus lặp nhiều, tức benchmark một-nhãn sẽ sai nặng trên corpus đó.

### Bước 6 — Khử trùng lặp

- **Làm:** embed câu hỏi bằng embedding model của pipeline; cosine ≥ `dedup_threshold` (mặc định 0.92) thì giữ
  câu có `lexical_overlap` thấp hơn.
- **Ghi:** `dedup_removed_count`.

### Bước 7 — Ghi bộ dữ liệu và manifest

- Ghi `qa.jsonl` theo schema mục 4.1, `manifest.json` theo mục 4.2.
- `protocol_split` luôn `"dev"`. Không có cờ để đổi. Người dùng muốn claim thì phải qua đường golden set.

Sơ đồ:

```text
raw docs
  -> [1] pipeline chunker            -> chunks (id thật)
  -> [2] G: question + answer_span   -> lọc span cơ học
  -> [3] V: trả lời từ chunk         -> lọc answerability
  -> [4] overlap; G viết lại; V lại  -> cờ high_overlap
  -> [5] BM25 (+ dense tùy chọn) pool; P đánh bậc -> nhãn nhiều tài liệu, có bậc; gắn lại unanswerable
  -> [6] dedup theo embedding
  -> [7] qa.jsonl + manifest.json (protocol_split: dev)
```

---

## 4. Schema đầu ra

### 4.1 Một dòng `qa.jsonl`

Tương thích ngược với `EvalItem.from_dict`. Trường mới chỉ nằm trong `metadata`.

```json
{
  "question_id": "syn_0042",
  "question": "Nhân viên thử việc được nghỉ phép năm bao nhiêu ngày?",
  "ground_truth_answer": "12 ngày, tính theo tỉ lệ thời gian làm việc.",
  "expected_doc_ids": ["chinh_sach_nhan_su", "so_tay_nhan_vien"],
  "expected_chunk_ids": [],
  "expected_citations": ["chinh_sach_nhan_su"],
  "metadata": {
    "question_type": "factual",
    "difficulty": "medium",
    "is_answerable": true,
    "generated": true,
    "generator_version": "v2",
    "source_chunk_id": "chinh_sach_nhan_su:c7:3f9a1c0b2e",
    "answer_span": "được nghỉ phép năm 12 ngày, tính theo tỉ lệ thời gian làm việc",
    "lexical_overlap": 0.31,
    "high_overlap": false,
    "rewritten": true,
    "relevance_by_doc_id": {"chinh_sach_nhan_su": 2, "so_tay_nhan_vien": 2},
    "pool_candidates": 27,
    "relabelled_from_unanswerable": false
  }
}
```

Quy ước:

- `expected_citations` chỉ chứa **tài liệu nguồn**. Citation là "bạn có trích đúng chỗ tôi lấy đáp án không", khác
  với "bạn có tìm được một tài liệu trả lời được không".
- `source_chunk_id` là **xuất xứ**, không phải nhãn. Nó nói câu hỏi sinh ra từ đâu, và không tham gia chấm điểm.
- `relevance_by_doc_id` ở cấp doc vì metrics hiện đọc ở cấp đó. Khi metrics đối chiếu được theo `answer_span` thì
  thêm trường mới, không đổi trường cũ.

### 4.2 `manifest.json`

```json
{
  "dataset_id": "user_hr_policy_2026_09",
  "generator_version": "v2",
  "created_at": "2026-09-10T14:02:00+07:00",
  "corpus": {
    "path": "datasets/user_data/hr_policy",
    "fingerprint": "sha256:...",
    "documents": 14,
    "chunks": 213,
    "chunker_id": "recursive_token",
    "chunk_config": {"chunk_size": 400, "overlap": 50}
  },
  "models": {
    "generator": "gpt-4.1-mini",
    "verifier": "gpt-4.1",
    "pooler": "gpt-4.1",
    "embedding": "bge-m3"
  },
  "params": {
    "seed": 42,
    "questions_per_chunk": 2,
    "overlap_threshold": 0.6,
    "pool_k": 20,
    "dedup_threshold": 0.92
  },
  "stages": {
    "generated": 426,
    "span_reject": 18,
    "answerability_reject": 61,
    "answerability_reject_rate": 0.150,
    "rewrite": 97,
    "rewrite_reject": 12,
    "lexical_overlap_p50": 0.34,
    "lexical_overlap_p90": 0.58,
    "pool_size_mean": 26.4,
    "extra_relevant_mean": 0.8,
    "relabelled_from_unanswerable": 9,
    "dedup_removed": 14,
    "final": 333,
    "final_unanswerable": 31
  },
  "metadata": {
    "protocol_split": "dev",
    "generated": true
  }
}
```

`corpus.fingerprint` dùng `canonical_fingerprint` sẵn có trong `core/measure.py`, tính trên nội dung đã chuẩn hoá
của mọi file, để hai lần chạy trên cùng corpus ra cùng fingerprint.

---

## 5. Cổng tin cậy

Pipeline mục 3 giảm lỗi nhãn nhưng không chứng minh được nó đã giảm đủ. Phần này đo phần còn sót.

### 5.1 Bộ vàng

- 50 đến 100 câu do người gán, dùng schema [golden](../ragbench/datasets/golden.py) hiện có
  (`question_id`, `question`, `reference_answer`, `is_answerable`, `required_claims`, `evidence_spans`).
- Bắt buộc `evidence_spans` có `doc_id` và `text`. Từ span suy ra `expected_doc_ids`; khi map được span vào chunk
  thì suy `expected_chunk_ids`.
- Người gán không được nhìn bộ synthetic trước khi viết, để hai bộ độc lập.
- Tỉ lệ unanswerable trong bộ vàng ≥ 10%.

### 5.2 Lệnh `raglab dataset audit`

```bash
raglab dataset audit \
  --synthetic datasets/user_data/hr_policy/qa.jsonl \
  --golden    datasets/golden/hr_policy.jsonl \
  --docs      datasets/user_data/hr_policy \
  --techniques naive_rag bm25_hybrid_rerank parent_child hyde_2022 \
  --mode retrieval_only \
  --output benchmarks/results/hr_policy_audit
```

Làm ba việc:

1. **Tương quan thứ hạng.** Chạy mọi technique trên cả hai bộ, xếp hạng theo từng primary metric
   (`ndcg_at_10`, `recall_at_10`, `mrr`), tính **Kendall tau-b** giữa hai bảng cho từng metric, và tau trên
   thứ hạng tổng hợp. Kèm bootstrap CI95 trên tau bằng cách lấy lại mẫu câu hỏi của bộ synthetic, dùng lại
   `benchmarks/statistics.py`.
2. **Độ chính xác nhãn.** Lấy mẫu `--label-sample` câu (mặc định 40) từ bộ synthetic, xuất ra file để người kiểm
   từng cặp (câu hỏi, đoạn được gắn nhãn liên quan) và đánh đúng/sai. Đọc lại file đó để tính `label_precision`.
   Bước này có người trong vòng, không tự động; lệnh in hướng dẫn và chờ file trả về.
3. **Đọc manifest** để lấy các tỉ lệ loại từ mục 3.

### 5.3 Khối `trust` trong report

Mọi report chạy trên bộ có `metadata.generated: true` **phải** mang khối này. Nếu chưa audit, khối vẫn xuất hiện
với `audited: false` và verdict `synthetic_untrusted`.

```json
"trust": {
  "audited": true,
  "audit_path": "benchmarks/results/hr_policy_audit/audit.json",
  "golden_queries": 80,
  "rank_agreement": {
    "ndcg_at_10": {"kendall_tau": 0.73, "ci95": [0.41, 0.93]},
    "recall_at_10": {"kendall_tau": 0.60, "ci95": [0.20, 0.87]},
    "mrr": {"kendall_tau": 0.67, "ci95": [0.33, 0.90]}
  },
  "label_precision": {"sampled": 40, "correct": 36, "value": 0.90},
  "generation": {
    "answerability_reject_rate": 0.150,
    "lexical_overlap_p90": 0.58,
    "extra_relevant_mean": 0.8
  },
  "verdict": "synthetic_trusted_for_ranking",
  "reasons": []
}
```

### 5.4 Ngưỡng và verdict

| Điều kiện | Verdict |
|---|---|
| Chưa audit | `synthetic_untrusted` |
| Kendall tau tổng hợp < 0.5, hoặc CI95 dưới chứa 0 | `synthetic_untrusted`, lý do `rank_disagreement` |
| `label_precision` < 0.8 | `synthetic_untrusted`, lý do `label_noise` |
| `answerability_reject_rate` > 0.4 | `synthetic_untrusted`, lý do `weak_generator` |
| `lexical_overlap_p90` > 0.75 | thêm cảnh báo `lexical_bias` nhưng không hạ verdict |
| Qua hết | `synthetic_trusted_for_ranking` |

Không có verdict nào cho phép dùng **con số tuyệt đối**. Tên verdict nói rõ: chỉ tin để xếp hạng. Ngưỡng ban đầu
đặt theo kinh nghiệm; sau ba corpus thật thì hiệu chỉnh lại và ghi lịch sử vào tài liệu này.

Report Markdown in khối trust **trước** bảng kết quả, không phải ở phụ lục.

---

## 6. Tách vai model

Ba vai: `generator`, `verifier` (dùng cho bước 3, 4 và pooler bước 5), `judge` (đường `--judge` của `eval` và
`bench`).

- Cấu hình qua env: `RAGBENCH_GENERATOR_MODEL`, `RAGBENCH_VERIFIER_MODEL`, `RAGBENCH_JUDGE_MODEL`. Cờ CLI
  `--model` của `dataset generate` chỉ đặt generator.
- **Kiểm ở thời điểm chạy:** `dataset generate` từ chối chạy nếu generator trùng verifier. `eval`/`bench --judge`
  từ chối chạy nếu judge trùng generator ghi trong manifest của bộ đang chấm. Cờ `--allow-same-model` bỏ chặn
  nhưng ghi `same_model_roles: [...]` vào report và hạ verdict về `synthetic_untrusted` với lý do
  `circular_judge`.
- Khuyến nghị: generator được phép rẻ; verifier mạnh hơn generator; judge khác họ với cả hai nếu có thể.

---

## 7. Thay đổi CLI

| Lệnh | Thay đổi |
|---|---|
| `dataset generate` | Chạy pipeline v2. Thêm `--verifier-model`, `--pool-k`, `--overlap-threshold`, `--dedup-threshold`, `--seed`, `--dataset-id`. Ghi `manifest.json` bên cạnh `qa.jsonl`. Bỏ `--limit` theo số câu, thay bằng `--max-chunks` để giới hạn chi phí theo đầu vào |
| `dataset audit` | Mới, mục 5.2 |
| `dataset validate` | Nhận bộ synthetic v2, kiểm mọi `expected_chunk_ids` tồn tại trong index của corpus (bắt lại lỗi 1.1 nếu tái xuất) |
| `bench`, `eval` | Đọc manifest; nếu `generated: true` thì bắt buộc xuất khối `trust`; kiểm trùng vai judge |

`dataset generate` phiên bản cũ giữ lại một release dưới tên `dataset generate-v1` để so sánh, rồi xoá.

---

## 8. Ngoài phạm vi v2

- **PDF, DOCX, HTML.** Parser là việc riêng; v2 vẫn chỉ nhận text và Markdown.
- **Câu hỏi multi-hop thật** cần chọn hai chunk liên quan rồi sinh câu nối chúng. Để v3, sau khi có pooling để biết
  chunk nào liên quan với nhau.
- **Golden set sinh bán tự động.** Bộ vàng là của người gán. Không có đường tắt.
- **Hỗ trợ nhãn cấp chunk có bậc trong metrics.** Metrics hiện đọc `relevance_by_doc_id`; nâng lên cấp chunk là
  việc của evaluation, không của generator.
- Không đổi tên namespace `ragbench`, theo quyết định đã hoãn trong `research_roadmap.md`.

---

## 9. Kế hoạch triển khai

Thứ tự đặt theo giá trị trên chi phí, và theo lời khuyên xuyên suốt của repo: **không xây thêm framework khi chưa
có số**.

| Mốc | Việc | Trạng thái |
|---|---|---|
| S0 | Sửa lỗi 1.1 và chặn judge trùng generator | **Xong** |
| S1 | Bước 1, 2, 3, 7: chunker chung, answer span, lọc answerability, manifest | **Xong** |
| S2 | Bước 5: pooling và gắn lại unanswerable | **Xong** |
| S3 | Bước 4, 6: overlap, viết lại, dedup | **Xong** |
| S4 | `dataset audit`, khối trust, ngưỡng verdict | **Xong** |
| S5 | Chạy trên một corpus thật, gán 80 câu vàng, hiệu chỉnh ngưỡng, cập nhật mục 5.4 | **Chưa** — cần người gán nhãn và chi phí API |

S5 không phải việc code: nó cần một corpus thật, tiền gọi API, và người ngồi gán 80 câu vàng. Cho tới khi nó chạy,
mọi ngưỡng ở mục 5.4 vẫn là con số phỏng đoán, và tài liệu này nói đúng như vậy.

Định nghĩa xong: một corpus thật đi hết S1 đến S5, report có khối trust với verdict
`synthetic_trusted_for_ranking`, và Kendall tau kèm CI95 được ghi vào tài liệu này làm điểm neo cho lần hiệu chỉnh
ngưỡng sau.

### 9.1 Bản đồ code

| Phần | Chỗ nằm |
|---|---|
| Bảy bước sinh đề, manifest, validator | `ragbench/datasets/synthetic.py` |
| Kendall tau-b, label precision, `run_audit` | `ragbench/evaluation/audit.py` |
| Khối trust, ngưỡng, verdict | `ragbench/evaluation/trust.py` |
| `retrieval_label_level` (chẩn đoán lỗi 1.1) | `ragbench/evaluation/metrics/__init__.py` |
| Gắn khối trust vào report, định tuyến validator | `ragbench/evaluation/runner.py` |
| `dataset generate` / `validate-synthetic` / `audit`, chặn trùng vai | `ragbench/cli/main.py` |
| Test | `tests/test_synthetic_benchmark_v2.py` (41 test) |

---

## 10. Những chỗ code khác thiết kế

Triển khai làm lộ ra bốn chỗ bản thiết kế đầu sai hoặc quá tay. Ghi lại vì lý do quan trọng hơn kết luận.

**Nhãn là cấp tài liệu, không phải cấp chunk.** Đây là sửa lớn nhất. Bản đầu nói "chunk bằng chunker của pipeline"
rồi ghi `expected_chunk_ids` thật. Nhưng *chunker nào*? `parent_child` chia khác `naive_rag`, nên `chunk_id` do một
technique sinh ra vô nghĩa với technique chia kiểu khác. Chunk id không chỉ sai giữa v1 và pipeline, nó sai giữa
mọi cặp technique. Chỉ `doc_id` sống sót qua mọi chunker, nên nhãn là cấp tài liệu, `expected_chunk_ids` để rỗng
đúng như đường fixed dataset vẫn làm, và độ liên quan có bậc đi vào `relevance_by_doc_id`. Chunker của pipeline vẫn
được dùng, nhưng để sinh đề và kiểm span, không để đặt nhãn.

*Hệ quả phải chấp nhận:* nhãn cấp tài liệu yếu đi khi corpus có ít file rất dài, vì recall gần như luôn bằng 1.
`extra_relevant_mean` và số tài liệu trong manifest là chỗ nhìn ra điều đó. Cách sửa thật là đối chiếu theo
`answer_span` thay vì theo id, và nó cần metrics hỗ trợ, nên vẫn nằm ngoài phạm vi.

**`lexical_overlap` là containment, không phải Jaccard.** Câu hỏi ngắn, đoạn văn dài, nên Jaccard chủ yếu đo độ dài
đoạn. Cái cần đo là *bao nhiêu phần của câu hỏi được bê từ đoạn ra*, tức tỉ lệ token câu hỏi xuất hiện trong đoạn.

**Khử trùng lặp bằng cosine từ vựng, không phải embedding.** Hai câu hỏi giống nhau tới mức cần loại thì dùng gần
hết cùng bộ token, nên thước đo rẻ tách chúng không kém, và bước sinh đề không phải phụ thuộc thêm nhà cung cấp
embedding.

**Pooling dense là tùy chọn.** Trả lời câu hỏi mở số 1 ở mục 11: mặc định pool chỉ bằng BM25, và khối trust cảnh
báo `single_retriever_pool`. Bật dense bằng `--pool-embedding-model`, và nên chọn embedding **khác** với embedding
đang được benchmark. Để mặc định là dense sẽ bắt mọi lần sinh đề trả một lượt embedding toàn corpus, mà tiền đó
chưa chắc đáng khi chưa ai đo được nó cải thiện nhãn bao nhiêu.

Hai điều nhỏ hơn: `audit` **đọc report có sẵn** thay vì tự chạy lại eval, nên nó rẻ và trung thực về thứ đã thực sự
đo; và `overall_kendall_tau` lấy **metric yếu nhất**, không lấy trung bình, vì một benchmark xếp đúng theo nDCG mà
xếp sai theo recall thì chưa có tiêu đề nào đáng tin.

Một lỗi tìm ra lúc viết test: pool rỗng (corpus chỉ có một đoạn) từng làm mất luôn nhãn của tài liệu nguồn, sinh ra
câu hỏi trả lời được mà không có nhãn nào. Nay tài liệu nguồn luôn được gắn nhãn kể cả khi không gọi pool.

---

## 11. Câu hỏi mở

1. ~~**Pooling dùng embedding nào?**~~ Đã chốt ở mục 10: BM25 mặc định, dense tùy chọn qua
   `--pool-embedding-model`, và nên khác embedding đang benchmark. Cảnh báo `single_retriever_pool` nói rõ khi chỉ
   có một họ retriever.
2. **Ngưỡng tau 0.5** là con số kinh nghiệm cho bốn tới chín technique. Với ít technique hơn, tau rất thô — với hai
   technique nó chỉ nhận hai giá trị, +1 hoặc -1. Có nên đổi sang "top-1 trùng nhau và top-3 trùng tập" khi số
   technique dưới bốn không. Chốt ở S5.
3. **`label_precision`** nay đo cả hai cấp: `pair_precision` và `question_precision`. Verdict dùng cấp cặp, đúng
   như đề xuất. Còn phải xem sau S5 là cấp cặp có phạt pooling nặng quá không.
4. **Chi phí.** Mỗi câu tốn 3 đến 4 lần gọi LLM. Với `--max-chunks 200` và 2 câu mỗi chunk là khoảng 1500 lần gọi.
   `dataset generate` **chưa có** `--max-estimated-cost-usd` như `bench`; `--max-chunks` hiện là cách chặn chi phí
   duy nhất. Nên thêm trước khi chạy corpus thật.
5. **Định dạng đầu vào.** Vẫn chỉ `.md`, `.markdown`, `.txt`. Tài liệu thật của người dùng phần lớn là PDF hoặc
   docx, nên đường này chưa dùng được với họ cho tới khi có parser.
