# Chiến Lược Dataset Cho `rag-pipeline-lab`

Tài liệu này đề xuất bộ dataset nên dùng để biến `rag-pipeline-lab` thành một personal project RAG thuyết phục hơn.
Mục tiêu không phải là gom thật nhiều dataset, mà là xây một evaluation matrix đủ cân bằng giữa:

- Benchmark phổ biến để nhà tuyển dụng và AI engineer khác dễ hiểu kết quả.
- Dataset tiếng Việt để repo có điểm khác biệt rõ so với các RAG demo quốc tế.
- Dataset tự xây theo thị trường Việt Nam để chứng minh tư duy sản phẩm, data engineering và evaluation thực tế.

## Kết Luận Ngắn Gọn

Sau khi ưu tiên hướng "repo nổi bật nhờ thị trường Việt Nam", benchmark chính của repo nên là một **Vietnamese fixed
benchmark suite**. Các dataset quốc tế như BEIR, Natural Questions hay HotpotQA vẫn có giá trị tham khảo, nhưng không nên
là trung tâm của repo vì những dataset đó đã được paper gốc và nhiều framework lớn dùng rất nhiều.

Nên đi theo chiến lược ba tầng:

1. **Core smoke dataset trong repo**: giữ dataset nhỏ, tiếng Việt, tự tạo hoặc được phép commit để người xem clone repo là chạy được ngay.
2. **Fixed Vietnamese benchmark suite**: mọi paper/technique trong repo chạy trên cùng một list dataset tiếng Việt để kết quả so sánh công bằng.
3. **User dataset path**: người dùng vẫn có thể ingest/eval dataset riêng qua schema chuẩn `documents/queries/qrels`.

## Quyết Định: Fixed Vietnamese Benchmark Suite

Bộ benchmark mặc định nên có 6 dataset. Mỗi dataset kiểm tra một failure mode khác nhau của RAG, tránh việc một technique
thắng chỉ vì dataset quá hẹp.

Repo đã có lớp adapter để chuẩn hóa các dataset này về `documents.jsonl`, `queries.jsonl`, `qrels.jsonl`, đồng thời export
`docs/` và `qa.jsonl` để chạy được với ingest/eval hiện tại. Đây là lớp evaluation fixture, không liên quan tới luồng user
đưa tài liệu thô vào hệ thống.

| Pack ID | Dataset | Domain | Mode chính | Vì sao chọn | Trạng thái khuyến nghị |
| --- | --- | --- | --- | --- | --- |
| `vi_wiki_retrieval` | `mteb/VieQuADRetrieval` | Wikipedia/general knowledge | `retrieval_only` | Nhỏ, có corpus/query/qrels, phù hợp đo BM25/dense/hybrid/rerank tiếng Việt | Core P0 |
| `vi_mrc_abstention` | `taidng/UIT-ViQuAD2.0` | Wikipedia/general QA | `full_rag` | Có context, answer, `is_impossible`; tốt để đo answer correctness và abstention | Core P0 |
| `vi_legal_retrieval` | `YuITC/Vietnamese-Legal-Documents` | Luật Việt Nam | `retrieval_only` + `full_rag` | Domain Việt Nam mạnh, có query và relevant document ids, đủ lớn để test vector store/FAISS | Core P0 |
| `vi_legal_rag_small` | `NamSyntax/Vietnamese-Legal-QA-RAG` | Luật Việt Nam | `full_rag` | Có ground-truth context, answer, factoid/multi-hop/unanswerable; rất hợp judge faithfulness/hallucination | Core P1 |
| `vi_multihop_reasoning` | VIMQA | Wikipedia multi-hop tiếng Việt | `full_rag` | Có supporting facts cấp sentence, cần cho IRCoT/ReAct/GraphRAG/RAPTOR | Conditional P1 |
| `vi_finance_qa` | VNFinsQA | Tài chính/chứng khoán Việt Nam | `full_rag` | 790 câu hỏi tài chính tiếng Việt do analyst curate; tạo khác biệt thị trường rất rõ | Core P2 |

`Conditional P1` nghĩa là dataset phù hợp về mặt kỹ thuật nhưng cần kiểm tra điều kiện truy cập/license trước khi đưa vào
benchmark chạy tự động. Với VIMQA, full dataset yêu cầu ký user agreement, nên repo nên commit adapter và sample hợp lệ,
không commit raw data.

### Dataset Không Đưa Vào Core Ngay

| Dataset | Lý do chưa đưa vào core |
| --- | --- |
| BEIR / Natural Questions / HotpotQA tiếng Anh | Tốt cho sanity check quốc tế, nhưng không tạo khác biệt Việt Nam-domain |
| Vietnamese-Customer-Support-QA / CSConDa | Rất hợp thị trường, nhưng cần xác minh license/provenance kỹ hơn vì có nguồn từ real customer interactions |
| ViMedAQA | Hấp dẫn cho medical RAG, nhưng là high-stakes domain; nên làm sau khi repo có disclaimer, judge protocol và abstention ổn |
| Open-ViTabQA / OCR-VQA tiếng Việt | Tốt cho table/multimodal RAG, nhưng scope hiện tại của repo vẫn là text RAG |
| ALQAC full | Rất đáng dùng, nhưng access/format theo từng năm phức tạp hơn `YuITC/Vietnamese-Legal-Documents`; nên làm adapter sau |

Nếu chỉ chọn 5 dataset ưu tiên trong 1-2 tháng tới, nên chọn:

| Ưu tiên | Dataset | Vai trò trong repo | Lý do chọn |
| --- | --- | --- | --- |
| P0 | `sample_vi_enterprise` tự tạo | Local smoke, zero dependency | Clone repo chạy được trong vài phút, kiểm tra ingest/query/eval end-to-end |
| P0 | `mteb/VieQuADRetrieval` | Vietnamese retrieval benchmark | Nhỏ, dễ tải, đúng format retrieval, có qrels và phù hợp đánh giá embedding/retriever |
| P0 | `YuITC/Vietnamese-Legal-Documents` | Vietnam legal retrieval/RAG | Domain Việt Nam mạnh, có nhu cầu thực tế, rất hợp với retrieval + citation |
| P1 | `taidng/UIT-ViQuAD2.0` | Vietnamese full RAG + abstention | Có answerable/unanswerable, giúp kiểm tra hallucination và refusal |
| P1 | `NamSyntax/Vietnamese-Legal-QA-RAG` | Vietnamese legal full RAG | Nhỏ nhưng đúng RAG eval: context, answer, multi-hop, unanswerable |

## Nguyên Tắc Chọn Dataset

Một dataset đáng đưa vào repo cần đạt ít nhất ba trong các tiêu chí sau:

- Có corpus, queries và relevance labels rõ ràng.
- Có ground-truth answer hoặc supporting evidence để đánh giá full RAG.
- License hoặc điều kiện sử dụng đủ rõ để không commit nhầm dữ liệu hạn chế.
- Kích thước có thể downsample để CI/smoke chạy nhanh.
- Đại diện cho failure mode thật: lexical mismatch, multi-hop, long context, citation sensitivity, unanswerable, domain terminology.
- Có giá trị kể chuyện trong README: tại sao dataset này chứng minh một năng lực engineering cụ thể.

## Nhóm 1: Benchmark Quốc Tế Phổ Biến

### BEIR

**Nguồn**: [BEIR paper](https://arxiv.org/abs/2104.08663), [NeurIPS Datasets and Benchmarks](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract-round2.html)

**Nên dùng cho**: retrieval-only benchmark.

BEIR là lựa chọn tốt nhất để chứng minh repo không chỉ chạy toy examples. Benchmark này gom nhiều retrieval tasks và domain khác nhau, có corpus/query/qrels chuẩn. Với `rag-pipeline-lab`, BEIR nên được dùng để benchmark:

- BM25.
- Dense retriever.
- Hybrid retrieval.
- Reranker.
- RAG-Fusion/HyDE query transformation ở mode `retrieval_only`.

**Không nên dùng toàn bộ ngay** vì dataset có thể nặng và không phải task nào cũng phù hợp full RAG. Nên bắt đầu bằng adapter hỗ trợ một vài subset nhỏ:

- `fiqa`: finance retrieval, hợp với câu chuyện business.
- `nfcorpus`: biomedical-ish retrieval, nhỏ hơn nhiều bộ khác.
- `scifact`: scientific fact retrieval, hợp với citation support.
- `hotpotqa` trong BEIR hoặc bản gốc HotpotQA: multi-hop retrieval.

**Giá trị với nhà tuyển dụng**: cho thấy bạn biết dùng benchmark IR tiêu chuẩn, không tự chọn dataset quá dễ để làm đẹp số.

### MTEB / MMTEB

**Nguồn**: [MTEB paper](https://arxiv.org/abs/2210.07316), [MTEB Hugging Face organization](https://huggingface.co/mteb)

**Nên dùng cho**: embedding/retriever evaluation, đặc biệt khi repo có nhiều embedding backend.

MTEB không chỉ là một dataset mà là framework benchmark embedding đa nhiệm. Với `rag-pipeline-lab`, không cần tái triển khai toàn bộ MTEB. Nên làm adapter đọc task retrieval/reranking theo format chung:

- `corpus`
- `queries`
- `qrels`

Điểm mạnh là có thể so sánh embedding model tiếng Anh và đa ngôn ngữ. Điểm yếu là MTEB thiên về embedding evaluation, không trực tiếp kiểm tra answer generation và citation faithfulness.

**Giá trị với repo**: tạo cầu nối giữa RAG framework và embedding leaderboard thực tế.

### Natural Questions

**Nguồn**: [Google Research Natural Questions](https://research.google/pubs/natural-questions-a-benchmark-for-question-answering-research/), [GitHub dataset](https://github.com/google-research-datasets/natural-questions)

**Nên dùng cho**: open-domain QA và RAG answer benchmark.

Natural Questions dùng câu hỏi thật từ Google Search và answer annotation trên Wikipedia. Đây là dataset tốt để đánh giá RAG vì câu hỏi tự nhiên hơn nhiều dataset được viết thủ công theo template.

**Cách dùng đề xuất**:

- Không tải full dataset trong CI.
- Tạo script `raglab dataset prepare natural_questions --sample 200`.
- Convert sang schema nội bộ:
  - `question`
  - `ground_truth_answer`
  - `expected_doc_ids`
  - `expected_citations`
  - `is_answerable`

**Giá trị với repo**: chứng minh pipeline có thể xử lý open-domain QA chuẩn, không chỉ tài liệu nội bộ.

### HotpotQA

**Nguồn**: [HotpotQA](https://hotpotqa.github.io/)

**Nên dùng cho**: multi-hop retrieval, evidence aggregation và agentic RAG.

HotpotQA phù hợp khi triển khai:

- IRCoT.
- ReAct RAG.
- GraphRAG.
- RAPTOR.
- Query decomposition.

Điểm mạnh là có supporting facts, rất hợp với citation/evidence evaluation. Điểm yếu là Wikipedia-style, không phản ánh domain Việt Nam.

**Giá trị với repo**: tạo benchmark rõ cho các paper multi-hop, tránh triển khai agentic RAG mà chỉ test bằng câu hỏi single-hop.

### FRAMES

**Nguồn**: [FRAMES dataset summary](https://hyper.ai/en/datasets/34835), [paper](https://arxiv.org/abs/2409.12941)

**Nên dùng cho**: advanced RAG reasoning.

FRAMES có 824 câu hỏi multi-hop cần kết hợp thông tin từ 2-15 Wikipedia articles, có reasoning type như numerical, temporal, table, multiple constraints. Đây là benchmark rất hợp để phân biệt:

- Retriever mạnh nhưng generator yếu.
- Generator mạnh nhưng retrieval thiếu evidence.
- Long-context stuffing vs retrieval có chọn lọc.
- Graph/hierarchy retrieval.

**Khuyến nghị**: đưa vào roadmap P2/P3, chưa cần P0 vì adapter có thể phức tạp hơn và benchmark full RAG tốn chi phí.

## Nhóm 2: Dataset Tiếng Việt Có Sẵn

### UIT-ViQuAD2.0

**Nguồn**: [Hugging Face UIT-ViQuAD2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0)

**Nên dùng cho**: Vietnamese extractive QA và unanswerable QA.

UIT-ViQuAD2.0 có khoảng 39.6k rows, gồm train/validation/test, context tiếng Việt, question, answers, `is_impossible` và plausible answers. Đây là dataset tốt để kiểm tra:

- Khả năng trả lời có căn cứ từ passage.
- Khả năng abstain khi câu hỏi không trả lời được.
- Exact/substring answer matching.
- Judge metric tiếng Việt.

Điểm cần lưu ý: đây là machine reading comprehension theo context có sẵn, không phải open-domain RAG thuần. Muốn dùng cho RAG, nên deduplicate context thành corpus, rồi map question về context id.

**Khuyến nghị**: dùng làm P1 cho full RAG tiếng Việt, sau VieQuADRetrieval.

### VieQuADRetrieval

**Nguồn**: [mteb/VieQuADRetrieval](https://huggingface.co/datasets/mteb/VieQuADRetrieval)

**Nên dùng cho**: Vietnamese retrieval-only benchmark.

Đây là lựa chọn P0 cho tiếng Việt vì format đã gần với retrieval benchmark. Dataset card ghi validation có 2,490 documents, 2,048 queries và 4,096 relevant docs. Kích thước nhỏ, phù hợp local benchmark và CI mở rộng.

**Cách dùng đề xuất**:

- Thêm adapter `viequad_retrieval`.
- Chuẩn hóa về:
  - `documents.jsonl`
  - `queries.jsonl`
  - `qrels.jsonl`
- Benchmark các technique:
  - `naive_rag`
  - `bm25_hybrid_rerank`
  - `dpr_2020`
  - `rag_fusion_2024_v2`
  - `rewrite_retrieve_read_2023`

**Giá trị với repo**: chứng minh chất lượng retrieval tiếng Việt bằng benchmark có qrels thay vì chỉ synthetic QA.

### VIMQA

**Nguồn**: [VIMQA GitHub](https://github.com/vimqa/vimqa)

**Nên dùng cho**: Vietnamese multi-hop QA.

VIMQA có hơn 10,000 Vietnamese Wikipedia-based multi-hop QA pairs, có supporting facts cấp sentence. Đây là dataset Việt Nam rất đáng giá cho các technique:

- `ircot_2022`
- `react_rag`
- `graphrag_2024`
- `raptor_2024`
- query decomposition

Điểm cần lưu ý: full dataset yêu cầu ký user agreement. Vì vậy không nên commit data raw vào repo. Chỉ nên commit adapter, schema converter, sample demo nếu giấy phép cho phép.

**Giá trị với repo**: tạo khác biệt rất mạnh vì multi-hop tiếng Việt ít repo cá nhân làm tốt.

### ALQAC

**Nguồn**: [ALQAC official](https://alqac.github.io/), [nguyenlab/ALQAC](https://huggingface.co/datasets/nguyenlab/ALQAC)

**Nên dùng cho**: Vietnamese legal retrieval và legal QA.

ALQAC là lựa chọn rất hợp với RAG vì có legal document retrieval và legal question answering trên luật Việt Nam. Đây là domain có yêu cầu citation, grounding và abstention cao hơn QA thông thường.

Điểm cần lưu ý:

- Một số bản trên Hugging Face yêu cầu chấp nhận điều kiện truy cập.
- Legal data cần ghi rõ disclaimer: benchmark/research only, không phải tư vấn pháp lý.
- Nên lưu metadata văn bản pháp luật: law id, article id, effective date, source URL nếu có.

**Giá trị với repo**: thể hiện tư duy production RAG rõ hơn Wikipedia QA, vì legal RAG cần traceability và citation nghiêm túc.

### Vietnamese Legal Documents / BKAI Legal Retrieval

**Nguồn**: [YuITC/Vietnamese-Legal-Documents](https://huggingface.co/datasets/YuITC/Vietnamese-Legal-Documents)

**Nên dùng cho**: legal document retrieval quy mô vừa.

Dataset này được mô tả là benchmark retrieval tiếng Việt với corpus legal documents, train/test queries và relevant document ids. Dataset card ghi corpus có `cid`, `text`; split có `qid`, `question`, `cid`, `context_list`, và khoảng 119k rows.

So với ALQAC, dataset này có vẻ tiện hơn để làm retrieval adapter vì format đã rõ. Nên ưu tiên dùng cho:

- BM25 vs dense vs hybrid.
- Vietnamese embedding benchmark.
- Citation-aware full RAG trên điều/khoản luật.

**Giá trị với repo**: domain Việt Nam, có quy mô đủ lớn để FAISS/vector store layer thật sự có ý nghĩa.

## Nhóm 3: Dataset Tự Xây Cho Thị Trường Việt Nam

Dataset tự xây là phần tạo khác biệt lớn nhất. Nhiều repo RAG dùng Natural Questions/HotpotQA, nhưng ít repo có benchmark thực dụng cho tiếng Việt và thị trường Việt Nam. Nên xây các dataset nhỏ, sạch, có nguồn rõ ràng, thay vì scrape lớn nhưng khó kiểm soát.

### `vietnam_legal_citation_qa`

**Mục tiêu**: RAG pháp lý tiếng Việt có trích dẫn điều/khoản.

**Nguồn dữ liệu đề xuất**:

- Văn bản pháp luật từ [vbpl.vn](https://vbpl.vn/), cơ sở dữ liệu văn bản quy phạm pháp luật quốc gia.
- Chỉ dùng văn bản công khai, lưu source URL và ngày truy cập.

**Schema đề xuất**:

```json
{
  "question": "Người lao động có bao nhiêu ngày nghỉ hằng năm trong điều kiện bình thường?",
  "ground_truth_answer": "Theo Bộ luật Lao động, người lao động làm đủ 12 tháng trong điều kiện bình thường được nghỉ hằng năm 12 ngày làm việc.",
  "expected_citations": [
    {
      "document_title": "Bộ luật Lao động",
      "article": "Điều 113",
      "clause": "Khoản 1"
    }
  ],
  "difficulty": "factual",
  "question_type": "citation_sensitive",
  "market_domain": "legal"
}
```

**Vì sao đáng làm**:

- Hợp với citation accuracy, faithfulness, answer abstention.
- Dễ tạo failure analysis rõ: sai điều luật, thiếu khoản, dùng văn bản hết hiệu lực, hallucinate điều kiện.
- Rất hợp để demo GraphRAG/CRAG sau này.

### `vietnam_public_policy_qa`

**Mục tiêu**: hỏi đáp về chính sách công, thủ tục, giáo dục, y tế, bảo hiểm xã hội.

**Nguồn dữ liệu đề xuất**:

- [data.gov.vn](https://data.gov.vn/) và [open.data.gov.vn](https://open.data.gov.vn/) cho dữ liệu mở.
- Trang cơ quan nhà nước có tài liệu hướng dẫn công khai.
- FAQ công khai của các đơn vị nhà nước nếu điều khoản cho phép.

**Câu hỏi mẫu**:

- Điều kiện hưởng một chế độ.
- Hồ sơ cần chuẩn bị.
- Thời hạn xử lý.
- Cơ quan tiếp nhận.
- Trường hợp không đủ điều kiện.

**Vì sao đáng làm**:

- Gần với use case citizen assistant.
- Kiểm tra tốt unanswerable và citation support.
- Tạo điểm khác biệt Việt Nam-domain rõ hơn dataset Wikipedia.

### `vietnam_finance_filings_qa`

**Mục tiêu**: RAG trên báo cáo thường niên, báo cáo tài chính, công bố thông tin của doanh nghiệp niêm yết.

**Nguồn dữ liệu đề xuất**:

- Trang quan hệ nhà đầu tư của doanh nghiệp.
- Trang công bố thông tin chính thức của HOSE/HNX/UPCoM nếu điều khoản sử dụng cho phép.
- Không commit PDF/raw data nếu license không rõ; commit downloader metadata và adapter.

**Câu hỏi mẫu**:

- Doanh thu/lợi nhuận theo năm.
- Ban lãnh đạo, cổ đông lớn.
- Rủi ro kinh doanh.
- Thay đổi chiến lược.
- So sánh giữa hai kỳ báo cáo.

**Vì sao đáng làm**:

- Rất giống bài toán enterprise RAG thật.
- Có bảng, PDF, số liệu, citation và temporal reasoning.
- Phù hợp để sau này mở rộng sang document parsing, table extraction, multimodal/document RAG.

### `vietnam_education_admission_qa`

**Mục tiêu**: hỏi đáp tuyển sinh đại học/cao học, quy chế, học phí, học bổng, chương trình đào tạo.

Repo hiện đã có sample `quy_che_tuyen_sinh.md`, nên có thể mở rộng thành benchmark nhỏ 100-300 câu. Đây là domain an toàn hơn legal/finance, dễ tự tạo tài liệu public hoặc synthetic.

**Vì sao đáng làm**:

- Dễ demo cho người xem Việt Nam.
- Có nhiều câu hỏi citation-sensitive.
- Dễ tạo negative questions: hỏi thông tin không có trong quy chế.

### `vietnam_customer_support_qa`

**Mục tiêu**: mô phỏng RAG cho FAQ/sản phẩm/dịch vụ tiếng Việt.

**Nguồn dữ liệu đề xuất**:

- Tự viết mock docs, tránh dùng dữ liệu doanh nghiệp thật.
- Dùng 3-5 domain giả lập: ngân hàng số, bảo hiểm, logistics, thương mại điện tử, SaaS.

**Vì sao đáng làm**:

- Recruiter dễ hiểu business value.
- Không vướng license.
- Kiểm tra tốt query rewrite, hybrid retrieval, rerank và abstention.

## Evaluation Matrix Đề Xuất

| Benchmark pack | Dataset | Ngôn ngữ | Mode | Metrics chính | Technique phù hợp |
| --- | --- | --- | --- | --- | --- |
| `local_smoke` | `sample_vi_enterprise` | vi | full_rag | recall@k, citation_precision, citation_recall, citation_f1, faithfulness | all |
| `vi_retrieval_small` | VieQuADRetrieval | vi | retrieval_only | recall@5, mrr, ndcg@10 | BM25, dense, hybrid, RRF |
| `vi_legal_retrieval` | Vietnamese Legal Documents / ALQAC | vi | retrieval_only + full_rag | recall@k, citation_support, abstention | hybrid, rerank, CRAG |
| `en_ir_standard` | BEIR subset | en | retrieval_only | ndcg@10, recall@100, mrr | retrieval techniques |
| `en_open_qa` | Natural Questions sample | en | full_rag | answer_correctness, citation_precision, citation_recall, citation_f1 | RAG generation |
| `multi_hop` | HotpotQA / VIMQA / FRAMES | en/vi | full_rag | supporting_fact_recall, answer_correctness | IRCoT, ReAct, GraphRAG |
| `long_context` | LongGenBench-style synthetic local | en/vi | full_rag | key_point_recall, faithfulness | RAPTOR, context ordering |

Với định hướng mới, bảng benchmark chính trong README nên chỉ hiển thị các pack tiếng Việt trước. Các pack tiếng Anh nên
đưa xuống phần "international sanity checks" hoặc "optional adapters".

## Mapping Dataset Theo Paper/Technique

Một danh sách dataset cố định chỉ hữu ích nếu mỗi paper được đánh giá trên cùng subset có ý nghĩa. Đề xuất mapping:

| Technique | Dataset bắt buộc | Dataset phụ | Lý do |
| --- | --- | --- | --- |
| `bm25_hybrid_rerank` | `vi_wiki_retrieval`, `vi_legal_retrieval` | `vi_finance_qa` | Cần đo retrieval thuần trên general + legal domain; finance để kiểm tra thuật ngữ chuyên ngành |
| `dpr_2020` | `vi_wiki_retrieval`, `vi_legal_retrieval` | `vi_mrc_abstention` | Dense retrieval phải so với BM25/hybrid trên qrels rõ ràng |
| `lost_in_middle_context_ordering` | `vi_mrc_abstention`, `vi_legal_rag_small` | `vi_finance_qa` | Cần full RAG và context dài/nhiều evidence để thấy ảnh hưởng thứ tự context |
| `rewrite_retrieve_read_2023` | `vi_wiki_retrieval`, `vi_legal_retrieval`, `vi_finance_qa` | `vi_mrc_abstention` | Query rewrite nên cải thiện câu hỏi tự nhiên, thuật ngữ pháp lý/tài chính và query không khớp lexical |
| `rag_fusion_2024_v2` | `vi_wiki_retrieval`, `vi_legal_retrieval` | `vi_multihop_reasoning` | Fusion cần qrels để đo recall/MRR ổn định; multi-hop kiểm tra query expansion |
| `ragas_eval` | `vi_mrc_abstention`, `vi_legal_rag_small` | `vi_finance_qa` | Evaluation cần answer, context và faithfulness/citation labels |
| `crag_2024` | `vi_legal_rag_small`, `vi_finance_qa` | `vi_mrc_abstention` | Corrective RAG cần case retrieval kém, answer thiếu support và unanswerable |
| `adaptive_rag_2024` | `vi_wiki_retrieval`, `vi_legal_rag_small`, `vi_finance_qa` | `vi_multihop_reasoning` | Router cần nhiều độ khó/domain để quyết định simple vs complex retrieval |
| `raptor_2024` | `vi_multihop_reasoning` | `vi_legal_retrieval`, long-context synthetic từ legal/finance docs | Hierarchical retrieval cần tài liệu dài và câu hỏi nhiều evidence |
| `graphrag_2024` | `vi_multihop_reasoning`, `vi_legal_retrieval` | `vi_finance_qa` | GraphRAG cần entity/relation và global/multi-hop questions |
| `ircot_2022` | `vi_multihop_reasoning` | `vi_legal_rag_small` | IRCoT chỉ có ý nghĩa khi câu hỏi cần nhiều bước reasoning/retrieval |
| `react_rag` | `vi_multihop_reasoning`, `vi_finance_qa` | `vi_legal_rag_small` | Agentic RAG cần trace, tool calls và câu hỏi cần decomposition |

## Scorecard Chọn Dataset

| Dataset | RAG fit | Vietnam market fit | Reproducibility | Risk | Kết luận |
| --- | --- | --- | --- | --- | --- |
| VieQuADRetrieval | Cao cho retrieval | Trung bình | Cao | Thấp | Đưa vào core ngay |
| UIT-ViQuAD2.0 | Cao cho QA/abstention | Trung bình | Cao | Thấp | Đưa vào core ngay |
| Vietnamese-Legal-Documents | Rất cao | Rất cao | Cao | Trung bình do legal caveat | Đưa vào core ngay |
| Vietnamese-Legal-QA-RAG | Rất cao | Rất cao | Trung bình | Cần kiểm tra license/provenance | Đưa vào core P1 nếu license ổn |
| VIMQA | Rất cao cho multi-hop | Trung bình | Trung bình | User agreement | Conditional core |
| VNFinsQA | Cao | Rất cao | Trung bình | Dataset mới, cần kiểm tra schema/license | Core P2 |
| ViMedAQA | Cao | Cao | Cao | High-stakes medical | Optional sau |
| Vietnamese-Customer-Support-QA | Cao | Rất cao | Thấp/Trung bình | License/provenance/privacy | Chưa đưa vào core |

## Schema Chuẩn Nên Hỗ Trợ

Để không bị khóa vào từng dataset, repo nên chuẩn hóa về ba file:

```text
datasets/processed/<dataset_id>/
  documents.jsonl
  queries.jsonl
  qrels.jsonl
  dataset_card.md
```

`documents.jsonl`:

```json
{
  "doc_id": "law_45_2019_article_113",
  "title": "Bộ luật Lao động 2019 - Điều 113",
  "text": "Nội dung văn bản...",
  "source": "https://...",
  "metadata": {
    "language": "vi",
    "domain": "legal",
    "document_type": "law_article",
    "effective_date": "2021-01-01"
  }
}
```

`queries.jsonl`:

```json
{
  "query_id": "q_0001",
  "question": "Người lao động có bao nhiêu ngày nghỉ hằng năm?",
  "ground_truth_answer": "12 ngày làm việc trong điều kiện bình thường nếu làm đủ 12 tháng.",
  "is_answerable": true,
  "difficulty": "factual",
  "question_type": "citation_sensitive",
  "metadata": {
    "language": "vi",
    "domain": "legal"
  }
}
```

`qrels.jsonl`:

```json
{
  "query_id": "q_0001",
  "doc_id": "law_45_2019_article_113",
  "relevance": 2,
  "evidence_span": "người lao động làm đủ 12 tháng ... được nghỉ hằng năm 12 ngày làm việc",
  "citation": {
    "article": "Điều 113",
    "clause": "Khoản 1"
  }
}
```

## Implementation Roadmap

### Phase 1: Dataset Foundation

- Thêm `raglab/datasets/schema.py` với dataclass/Pydantic model cho `DocumentRecord`, `QueryRecord`, `QrelRecord`.
- Thêm `raglab/datasets/adapters/`:
  - `local_jsonl.py`
  - `viequad_retrieval.py`
  - `uit_viquad.py`
  - `beir.py`
- Thêm CLI:
  - `raglab dataset prepare <dataset_name>`
  - `raglab dataset validate <processed_dataset_dir>`
  - `raglab dataset sample <processed_dataset_dir> --n 50`
- Không commit raw external datasets; chỉ commit processed sample nhỏ nếu license cho phép.

### Phase 2: Vietnamese Benchmark Pack

- Implement `VieQuADRetrieval` adapter trước.
- Thêm benchmark config `benchmarks/configs/vi_retrieval_small.yaml`.
- Chạy `naive_rag`, dense, hybrid/rerank nếu đã có.
- Publish report ở `benchmarks/results/vi_retrieval_small/`.

### Phase 3: Legal RAG Pack

- Implement adapter cho `YuITC/Vietnamese-Legal-Documents` hoặc ALQAC.
- Thêm metadata legal-specific: article, clause, effective date, source.
- Thêm judge prompt riêng cho citation support tiếng Việt.
- Thêm failure taxonomy:
  - `wrong_article`
  - `missing_clause`
  - `outdated_law`
  - `unsupported_legal_advice`
  - `should_abstain`

### Phase 4: International Benchmark Pack

- Add BEIR subset adapter.
- Add Natural Questions small sample converter.
- Add HotpotQA/VIMQA converter cho multi-hop.
- README có bảng benchmark tách rõ:
  - English standard retrieval.
  - Vietnamese retrieval.
  - Vietnamese legal RAG.
  - Multi-hop RAG.

### Phase 5: Repo-Owned Vietnam Market Dataset

- Tự xây `datasets/public/vietnam_market_rag_v1/`.
- Chỉ dùng dữ liệu tự viết hoặc nguồn công khai có source rõ.
- Tạo 300-500 QA:
  - 100 legal/policy.
  - 100 education/admission.
  - 100 customer support.
  - 50 finance filings.
  - 50 negative/unanswerable.
- Mỗi QA bắt buộc có expected citations.

## Dataset Nên Tránh Ở Giai Đoạn Đầu

- Dataset scrape báo chí nếu chưa rõ bản quyền.
- Dataset social media tiếng Việt vì dễ chứa dữ liệu cá nhân, độc hại và khó xin quyền dùng.
- Dataset PDF tài chính từ nguồn không rõ license.
- Dataset quá lớn làm quickstart chậm.
- Dataset chỉ có question/answer nhưng không có corpus hoặc evidence, vì khó đánh giá RAG đúng nghĩa.

## README Nên Kể Câu Chuyện Như Thế Nào

Một README ấn tượng nên có đoạn:

> This repo evaluates RAG techniques on both standard international retrieval benchmarks and Vietnam-specific RAG datasets. The local smoke path runs without API keys. Research benchmarks include BEIR/MTEB-style retrieval, Vietnamese VieQuAD retrieval, and legal-domain Vietnamese RAG with citation-aware evaluation.

Sau đó có bảng:

| Dataset pack | Domain | Language | Size | Purpose | Status |
| --- | --- | --- | --- | --- | --- |
| `local_smoke` | enterprise docs | vi | tiny | CI + quickstart | implemented |
| `vi_retrieval_small` | Wikipedia QA retrieval | vi | small | Vietnamese retriever benchmark | planned |
| `vi_legal_rag` | law | vi | medium | citation-aware legal RAG | planned |
| `beir_small` | mixed IR | en | small/medium | standard IR comparison | planned |
| `multi_hop` | Wikipedia reasoning | en/vi | medium | IRCoT/ReAct/GraphRAG | planned |

Điểm quan trọng: đừng chỉ nói "supports many datasets". Hãy nói rõ dataset nào kiểm tra năng lực nào, vì đó là dấu hiệu engineering maturity.

## Khuyến Nghị Cuối

Nếu mục tiêu là làm repo nổi bật với Middle AI Engineer hiring panel, thứ tự tốt nhất là:

1. Implement `VieQuADRetrieval` adapter và benchmark retrieval tiếng Việt.
2. Implement legal retrieval/RAG pack bằng `Vietnamese-Legal-Documents` hoặc ALQAC.
3. Add BEIR subset để có chuẩn quốc tế.
4. Add HotpotQA hoặc VIMQA trước khi làm IRCoT/ReAct/GraphRAG.
5. Tự xây `vietnam_market_rag_v1` nhỏ nhưng sạch, có dataset card, source tracking và citation labels.

Chiến lược này giúp repo có cả độ tin cậy research lẫn khác biệt thị trường Việt Nam. Đây là phần mà nhiều RAG personal projects thiếu: họ demo pipeline, nhưng không chứng minh được pipeline đó được đánh giá nghiêm túc trên dữ liệu có ý nghĩa.
