Mình nghĩ nên update theo hướng **giữ nguyên framework hiện tại, bổ sung 2 golden benchmark chính**, thay vì refactor lớn. Repo đã có adapter architecture, prepared dataset schema, retrieval metrics, abstention/citation metrics và LLM judge rồi.

### Target cuối cùng

```text
Benchmark v1
│
├── Zalo Legal Retrieval
│   └── chứng minh retrieval quality
│       BM25 / Dense / Hybrid / Reranker / Contextual...
│
├── UIT-ViQuAD 2.0
│   └── chứng minh full-RAG quality
│       correctness / faithfulness / abstention
│
└── Synthetic benchmark
    └── giữ nguyên track hiện tại
        raw docs → generated QA → relative technique ranking
```

Plan code mình đề xuất:

1. **Không đụng vào synthetic benchmark hiện tại.** Giữ `ragbench/datasets/synthetic.py` và trust model hiện tại. Synthetic tiếp tục chỉ dùng cho relative ranking; golden benchmark là track riêng.

2. **Thêm adapter `zalo_legal_retrieval.py`.** Tạo:

    ```text
    ragbench/datasets/adapters/zalo_legal_retrieval.py
    ```

    Adapter phải convert upstream data về canonical schema:

    ```text
    DocumentRecord
    QueryRecord
    QrelRecord
    ```

    và output:

    ```text
    documents.jsonl
    queries.jsonl
    qrels.jsonl
    docs/
    qa.jsonl
    dataset_card.md
    manifest.json
    ```

    Quan trọng nhất là giữ nguyên graded relevance nếu upstream có, không biến tất cả thành binary nếu không cần.

3. **Đăng ký adapter mới trong registry.** Thêm:

    ```python
    "zalo_legal_retrieval": prepare_zalo_legal_retrieval
    ```

    vào `ragbench/datasets/adapters/registry.py`. Registry hiện đã được thiết kế đúng kiểu này rồi.

4. **Nâng provenance của dataset.** Với cả Zalo và UIT-ViQuAD, `manifest.json` nên chứa tối thiểu:

    ```yaml
    source:
    source_revision:
    upstream_split:
    corpus_size:
    query_count:
    qrel_count:
    fingerprint:
    language:
    domain:
    annotation_type: human
    ```

    Không nên để benchmark phụ thuộc vào `main/latest` của Hugging Face. Pin revision hoặc commit SHA để một năm sau vẫn reproduce được.

5. **Review lại adapter `uit_viquad.py`, nhưng không viết lại.** Adapter hiện đã hỗ trợ `is_impossible → is_answerable`, ground-truth answer và chỉ tạo qrel cho answerable questions. Đây chính xác là foundation cần cho full-RAG benchmark.

    Chỉ cần bổ sung metadata rõ hơn:

    ```text
    annotation_type = human
    task = extractive_qa
    supports_unanswerable = true
    ```

    và provenance/version.

6. **Thêm deterministic QA metrics cho UIT-ViQuAD.** Đây là phần mình thấy code hiện tại còn thiếu.

    Hiện full-RAG correctness chủ yếu phụ thuộc vào LLM judge; judge đã tách correctness và faithfulness khá tốt.

    Nhưng với extractive QA như ViQuAD, nên thêm:

    ```text
    exact_match
    token_f1
    ```

    sau Vietnamese text normalization:

    ```text
    lowercase
    whitespace normalization
    punctuation normalization
    ```

    Như vậy:

    ```text
    EM / Token-F1       = deterministic primary evidence
    LLM correctness     = semantic supporting evidence
    Faithfulness        = grounding evidence
    ```

    Không nên để LLM-as-judge là metric correctness duy nhất.

7. **Giữ nguyên retrieval metrics hiện tại.** Repo đã có:

    ```text
    Recall@K
    Hit Rate
    MRR
    nDCG@K
    MAP@K
    Context Precision
    Evidence Complete/Partial/Zero
    ```

    nên Zalo không cần metric engine mới.

8. **Tạo protocol riêng cho Zalo, không sửa `vi_retrieval_core.yaml` ngay.** Hiện `vi_retrieval_core.yaml` vẫn đang trỏ vào `vi_wiki_retrieval`, tier `exploratory`.

    Tạo:

    ```text
    ragbench/evaluation/protocol/
      vi_legal_retrieval_v1.yaml
    ```

    Ví dụ concept:

    ```yaml
    id: vi_legal_retrieval_v1
    tier: exploratory

    dataset:
        docs: datasets/processed/zalo_legal/docs
        qa: datasets/processed/zalo_legal
        fingerprint: ...

    mode: retrieval_only
    profile: retrieval

    top_k: 20
    cutoffs: [1, 5, 10, 20]

    primary_metrics:
        - ndcg_at_10
        - recall_at_10
        - mrr

    required_baselines:
        - naive_rag
        - parent_child
        - bm25_hybrid_rerank
    ```

9. **Tạo protocol riêng cho UIT-ViQuAD full RAG.**

    ```text
    vi_full_rag_v1.yaml
    ```

    Concept:

    ```yaml
    mode: full_rag
    profile: single_hop_rag

    primary_metrics:
        - exact_match
        - token_f1
        - abstention_accuracy
        - answer_correctness
        - faithfulness

    required_baselines:
        - naive_rag
        - parent_child
        - bm25_hybrid_rerank
    ```

    Repo hiện đã có `single_hop_rag` profile và validation cho unanswerable questions nên không cần tạo profile mới.

10. **Không benchmark tất cả techniques ngay.** Benchmark v1 trước tiên chỉ chạy:

    ```text
    naive_rag
    parent_child
    bm25_hybrid_rerank
    contextual_retrieval_2024
    ```

    Sau khi infrastructure đáng tin mới thêm:

    ```text
    HyDE
    RAG-Fusion
    Agentic RAG
    ```

    Nếu chạy agentic từ đầu sẽ rất khó phân biệt lỗi benchmark với lỗi technique.

11. **Thêm dataset validation/audit gate.** Trước khi một dataset được dùng làm evidence:

    ```bash
    ragbench dataset validate ...
    ragbench dataset audit ...
    ```

    nên verify:

    ```text
    duplicate query/doc IDs
    dangling qrels
    empty docs/questions
    qrel coverage
    answerable → có ground truth
    unanswerable → không có positive qrel sai
    corpus/query/qrel counts
    fingerprint match
    ```

    Repo đã có `evaluation/audit.py`, nên nên mở rộng nó thay vì tạo framework audit mới.

12. **Thêm manual audit artifact.** Random cố định khoảng 50 examples/dataset:

    ```text
    benchmarks/audits/
      zalo_legal_v1.jsonl
      uit_viquad_v1.jsonl
    ```

    Mỗi sample ghi:

    ```json
    {
        "query_id": "...",
        "valid_question": true,
        "ground_truth_valid": true,
        "qrel_valid": true,
        "notes": ""
    }
    ```

    Không nhất thiết phải automate annotation; mục tiêu là có evidence rằng dataset đã được con người kiểm tra.

13. **Thêm tests ở 3 layer.**

    ```text
    Adapter unit tests
      → mapping upstream → canonical schema

    Dataset invariant tests
      → qrels valid, IDs unique, answerability consistent

    Protocol tests
      → mọi committed YAML load được
      → expected profile/dataset/metrics đúng
    ```

    Với test adapter, mock HF rows; **CI không nên download 60K legal documents**.

14. **Chạy một pilot nhỏ trước.**

    ```bash
    dataset prepare ... --limit 100
    ```

    rồi chạy 3 baselines. Pilot chỉ kiểm tra:

    ```text
    adapter đúng
    doc IDs survive ingestion
    qrels match retrieved doc IDs
    metrics không về 0 bất thường
    report generation đúng
    ```

    Đặc biệt check `retrieval_label_level == "doc"` vì metric code hiện phân biệt chunk-level và doc-level qrels rất rõ.

15. **Sau pilot mới freeze Benchmark v1.** Prepare full dataset với:

    ```text
    seed cố định
    source revision cố định
    fingerprint cố định
    protocol_split=test
    ```

    Sau đó mới chạy experiment chính.

16. **Cuối cùng mới cập nhật README/results.** README nên thể hiện rõ ba mức evidence:

    ```text
    smoke test
        ↓
    synthetic benchmark
        ↓
    human-labelled golden benchmark
    ```

    và tuyệt đối không để sample benchmark `Recall=1.0` hiện tại nhìn giống empirical result thật.

### Thứ tự implementation mình sẽ làm

```text
PR/Step 1
Zalo adapter + provenance + tests
        ↓
PR/Step 2
UIT-ViQuAD provenance cleanup
+ EM / Token-F1
+ tests
        ↓
PR/Step 3
Dataset validation + audit improvements
        ↓
PR/Step 4
vi_legal_retrieval_v1.yaml
vi_full_rag_v1.yaml
        ↓
PR/Step 5
100-query pilot benchmark
        ↓
PR/Step 6
fix issues discovered by pilot
freeze fingerprints/protocols
        ↓
PR/Step 7
full Benchmark v1
+ benchmark tables
+ failure analysis
```

Điểm mình muốn nhấn mạnh là **không cần rewrite repo nhiều**. Dataset abstraction và evaluation infrastructure hiện tại đã khá đúng hướng; phần thiếu thực sự là **golden dataset đủ mạnh, deterministic full-RAG metrics, provenance chặt và protocol chính thức**. Đây nên là scope của lần update code tiếp theo.
