# Từ Baseline Đến Agentic RAG: Thứ Tự Triển Khai 12 Kỹ Thuật Nên Có Trong Một RAG Lab

RAG thường bị trình bày như một pipeline đơn giản: chunk tài liệu, embedding, retrieve top-k, nhét context vào prompt rồi
gọi LLM. Cách nhìn đó đủ cho demo, nhưng không đủ cho một hệ thống cần benchmark nghiêm túc, citation đáng tin, khả năng
mở rộng nhiều corpus, và quan trọng hơn là khả năng trả lời câu hỏi: kỹ thuật nào thật sự tốt hơn kỹ thuật nào?

Vì vậy, thứ tự triển khai trong `rag-pipeline-lab` không nên bắt đầu bằng các phương pháp phức tạp nhất như GraphRAG hay
agentic RAG. Nếu làm vậy, repo sẽ có nhiều technique nghe rất hay nhưng thiếu baseline đủ mạnh để chứng minh giá trị.
Một roadmap tốt phải đi từ nền móng đo lường, đến retrieval mạnh, đến query transformation, đến evaluation, rồi mới sang
adaptive, hierarchical và agentic RAG.

Bài viết này giải thích chi tiết thứ tự triển khai đề xuất:

1. `bm25_hybrid_rerank`
2. `dpr_2020`
3. `lost_in_middle_context_ordering`
4. `rewrite_retrieve_read_2023`
5. `rag_fusion_2024_v2`
6. `ragas_eval`
7. `crag_2024`
8. `adaptive_rag_2024`
9. `raptor_2024`
10. `graphrag_2024`
11. `ircot_2022`
12. `react_rag`

Mục tiêu không phải là copy paper một cách cơ học. Mục tiêu là biến mỗi ý tưởng thành một technique chạy được trong cùng
một framework, cùng artifact format, cùng benchmark runner, cùng evaluation protocol.

## Vì Sao Cần Một Thứ Tự Triển Khai?

Một RAG lab có thể thất bại theo hai cách phổ biến.

Cách thứ nhất là chỉ có các baseline quá đơn giản. Khi đó mọi technique mới đều có vẻ tốt, nhưng kết quả không nói được
nhiều. Nếu chỉ so sánh với naive retrieval, một query rewriting pipeline hoặc một fusion pipeline gần như chắc chắn nhìn
sẽ ấn tượng. Nhưng so với hybrid retrieval có reranker tốt, lợi ích có thể nhỏ hơn nhiều.

Cách thứ hai là nhảy thẳng vào kỹ thuật phức tạp. Ví dụ, triển khai GraphRAG trước khi có retrieval-only evaluation,
failure taxonomy, citation scoring và cost tracking sẽ làm hệ thống khó debug. Khi answer sai, ta không biết lỗi đến từ
chunking, entity extraction, community summary, retriever, prompt, generator hay judge.

Thứ tự dưới đây được thiết kế để tránh cả hai vấn đề:

- Xây baseline mạnh trước.
- Tách retrieval quality khỏi generation quality.
- Thêm query transformation sau khi có retrieval metrics ổn.
- Thêm evaluation research trước các kỹ thuật controller phức tạp.
- Chỉ thêm graph, hierarchy và agentic loop khi hệ thống đã có trace, cost guard và failure analysis.

## 1. `bm25_hybrid_rerank`: Baseline Thực Dụng Đầu Tiên

Technique đầu tiên nên triển khai là một production baseline mạnh: BM25 + dense retrieval + score fusion + optional
reranker. Đây không phải một paper cụ thể, nhưng lại là baseline quan trọng nhất cho mọi kết quả sau này.

Trong thực tế, nhiều corpus không thể chỉ dùng dense retrieval. Dense retrieval tốt cho semantic similarity, nhưng có thể
bỏ lỡ keyword quan trọng như mã điều khoản, tên sản phẩm, số hiệu văn bản, tên riêng, lỗi chính tả hoặc cụm từ hiếm. BM25
lại mạnh ở lexical matching, nhưng yếu khi câu hỏi và tài liệu dùng cách diễn đạt khác nhau. Hybrid retrieval kết hợp hai
nguồn tín hiệu này.

Reranker là lớp thứ ba. Retriever ban đầu nên ưu tiên recall, lấy nhiều candidate hơn một chút. Reranker sau đó xếp lại
candidate theo relevance sát hơn với câu hỏi. Dù dùng lexical-overlap reranker đơn giản hay cross-encoder về sau, pattern
này tạo ra một baseline đủ mạnh để các technique tiếp theo phải thật sự có giá trị mới vượt qua được.

Trong repo, `bm25_hybrid_rerank` nên được triển khai như một production pattern:

- `retriever`: hybrid BM25 + dense.
- `fusion`: normalize score hoặc reciprocal rank fusion đơn giản.
- `reranker`: lexical-overlap trước, cross-encoder optional sau.
- `context_builder`: citation-aware context.
- `evaluation`: ưu tiên `retrieval_only` trước khi chạy `full_rag`.

Điểm cần benchmark:

- Recall@5 và Recall@10.
- MRR.
- Citation accuracy.
- Latency tăng thêm do rerank.
- Cost nếu reranker dùng model.

Kết quả mong đợi không phải là technique này luôn thắng. Kết quả mong đợi là nó trở thành baseline đủ mạnh để mọi kỹ thuật
sau này có một mốc so sánh nghiêm túc.

## 2. `dpr_2020`: Dense Retrieval Như Một Nền Tảng

Sau hybrid baseline, bước tiếp theo là `dpr_2020`, dựa trên Dense Passage Retrieval. DPR quan trọng vì nó định hình cách
nhiều hệ thống open-domain QA hiện đại nhìn retrieval: encode query và passage vào cùng vector space, sau đó tìm nearest
neighbors.

Trong repo này, ta không cần reproduction đầy đủ DPR training. Điều đó sẽ quá nặng và không cần thiết cho mục tiêu ban
đầu. Thay vào đó, nên triển khai theo hướng paper-inspired:

- Tách rõ query encoder và passage encoder ở interface, dù ban đầu có thể cùng dùng một embedding model.
- Chuẩn hóa dense retriever để dùng vector store.
- Ghi metadata embedding model vào artifact manifest.
- Cho phép thay backend `json_memory` bằng `faiss_local`.

Lý do `dpr_2020` đứng trước các technique query transformation là vì nó làm rõ dense retrieval baseline. Nếu dense path
còn mơ hồ, các kỹ thuật như HyDE, RAG-Fusion hay Rewrite-Retrieve-Read sẽ khó đánh giá công bằng.

Khi triển khai, cần cẩn thận phân biệt ba mức:

- Faithful DPR: training dual encoder với negative sampling, dùng dataset gốc. Không nên làm ngay.
- Paper-inspired DPR: dense retrieval pipeline có query/passage encoding rõ. Nên làm.
- Production dense retrieval: dùng embedding API hiện đại, vector DB, batch embedding, artifact manifest. Nên hỗ trợ.

Tiêu chí hoàn thành:

- Có config riêng cho dense retrieval.
- Có benchmark so với `bm25_hybrid_rerank`.
- Có artifact inspect hiển thị embedding model, vector store backend, node count.
- Chạy được `retrieval_only` để tách retrieval khỏi generation.

## 3. `lost_in_middle_context_ordering`: Tối Ưu Context Trước Khi Tăng Độ Phức Tạp

Nhiều người cố cải thiện RAG bằng cách retrieve nhiều hơn, nhưng không xử lý câu hỏi quan trọng hơn: context được sắp xếp
như thế nào trong prompt?

Paper Lost in the Middle chỉ ra rằng mô hình ngôn ngữ thường dùng thông tin ở đầu và cuối context tốt hơn thông tin nằm ở
giữa. Với RAG, điều này có hệ quả trực tiếp. Nếu passage quan trọng bị nhét vào giữa một context dài, retrieval có thể đã
đúng nhưng generation vẫn sai.

Technique `lost_in_middle_context_ordering` nên được triển khai ở stage `context_construction`, không phải retrieval.
Điều này rất đáng giá vì nó giúp repo chứng minh rằng RAG quality không chỉ là top-k retrieval.

Các biến thể nên thử:

- `score_desc`: passage điểm cao nhất đứng đầu.
- `edge_biased`: passage quan trọng nhất đặt ở đầu và cuối.
- `interleaved`: xen kẽ passage mạnh/yếu để tránh evidence quan trọng nằm giữa quá lâu.
- `citation_grouped`: nhóm theo document hoặc section để giữ coherence.

Điểm thú vị của technique này là cost gần như bằng 0. Không cần thêm model call, không cần thêm embedding. Chỉ thay đổi
cách build context. Vì vậy, nếu nó cải thiện answer correctness hoặc citation support, đây là một optimization rất đáng
giá.

Tiêu chí hoàn thành:

- Thêm context builder hoặc context ordering strategy.
- Cùng retrieved results nhưng nhiều ordering policy khác nhau.
- Report phải cho thấy retrieval metrics giữ nguyên, nhưng generation/evaluation metrics có thể thay đổi.
- Có failure analysis cho case retrieval đúng nhưng answer sai.

## 4. `rewrite_retrieve_read_2023`: Query Rewriting Trước Retrieval

Sau khi có retrieval baseline và context ordering, bước tiếp theo là query rewriting. Rewrite-Retrieve-Read là một ý tưởng
thực dụng: thay vì dùng trực tiếp câu hỏi người dùng để retrieve, ta cho model rewrite câu hỏi thành query tốt hơn cho
retrieval.

Điểm mạnh của kỹ thuật này là nó giải quyết mismatch giữa ngôn ngữ người dùng và ngôn ngữ tài liệu. Người dùng có thể hỏi
"nghỉ phép tối đa bao nhiêu ngày", trong khi tài liệu viết "số ngày nghỉ thường niên được hưởng". Query rewriting có thể
chuyển câu hỏi tự nhiên thành một hoặc nhiều truy vấn sát tài liệu hơn.

Trong repo, `rewrite_retrieve_read_2023` nên thêm một stage trước retriever:

```text
question
  -> query_rewriter
  -> retriever
  -> reranker
  -> context_builder
  -> generator
```

Kỹ thuật này cần debug payload thật rõ. Nếu answer sai, ta phải nhìn được query đã rewrite là gì. Nếu rewrite drift khỏi
ý định ban đầu, retrieval có thể tệ hơn baseline.

Các biến thể nên hỗ trợ:

- Rewrite một query duy nhất.
- Rewrite thành nhiều query nhưng chưa fusion.
- Rewrite có giữ nguyên named entities/số liệu.
- Rewrite theo cùng ngôn ngữ với câu hỏi.
- Rewrite theo ngôn ngữ của corpus nếu corpus khác ngôn ngữ người dùng.

Tiêu chí hoàn thành:

- `retrieval_runtime.debug_payload.rewritten_query` hoặc tương đương.
- So sánh với dense/hybrid baseline trên cùng dataset.
- Có metric latency và cost riêng cho rewrite call.
- Có fallback dùng original query nếu rewrite rỗng hoặc lỗi parse.

## 5. `rag_fusion_2024_v2`: Fusion Tổng Quát Thay Vì Một Retriever Riêng Lẻ

Repo đã có `rag_fusion_2024`, nhưng bản v2 nên biến RAG-Fusion thành một cơ chế tổng quát hơn. Ý tưởng chính của
RAG-Fusion là tạo nhiều query biến thể, retrieve cho từng query, rồi fuse các ranked list bằng Reciprocal Rank Fusion.

Phiên bản đầu thường hard-code dense retrieval. Phiên bản tốt hơn nên cho phép mỗi query biến thể chạy qua cùng một
retriever backend có thể cấu hình:

- BM25.
- Dense.
- Hybrid.
- OpenAI dense.
- FAISS-backed dense.

Điểm quan trọng là tách hai khái niệm:

- Query expansion: tạo nhiều query.
- Rank fusion: hợp nhất nhiều result list.

Nếu tách tốt, RRF có thể dùng lại cho nhiều technique khác, kể cả hybrid retrieval hoặc multi-hop retrieval.

Trong `rag_fusion_2024_v2`, debug payload nên ghi:

- Original query.
- Generated queries.
- Result list theo từng query.
- RRF score cuối.
- Matched queries cho mỗi retrieved node.

Tiêu chí hoàn thành:

- RRF là component reusable.
- Có config chọn sub-retriever.
- Có ablation số lượng generated queries.
- Có benchmark latency/cost vì RAG-Fusion thường đắt hơn single-query retrieval.

Kỹ thuật này nên đứng sau Rewrite-Retrieve-Read vì nó phức tạp hơn: không chỉ rewrite một query mà sinh nhiều query và
fusion nhiều ranked lists.

## 6. `ragas_eval`: Evaluation Trước Khi Controller Phức Tạp

Trước khi thêm CRAG, Adaptive-RAG, RAPTOR hay GraphRAG, repo cần evaluation mạnh hơn. Nếu không, các technique phức tạp
sẽ rất khó đánh giá. `ragas_eval` nên được triển khai như một optional metric adapter lấy cảm hứng từ RAGAS.

Các chiều đo quan trọng:

- Faithfulness: answer có được support bởi context không?
- Answer relevance: answer có trả lời đúng câu hỏi không?
- Context precision: retrieved context có nhiều phần thật sự liên quan không?
- Context recall: context có chứa đủ evidence cần thiết không?

Repo hiện đã có metrics retrieval và optional judge. `ragas_eval` không nhất thiết phải phụ thuộc trực tiếp vào thư viện
RAGAS ở bước đầu. Có thể bắt đầu bằng adapter nội bộ mô phỏng các chiều đo này, sau đó thêm external integration nếu cần.

Điểm cần tránh là biến evaluation thành black box. Mỗi judge metric nên ghi:

- Judge model.
- Prompt version.
- Input fields.
- Raw/parsed output.
- Parse failure notes.
- Estimated cost.

Tiêu chí hoàn thành:

- Evaluation report có thêm group metric theo retrieval, generation, faithfulness.
- Judge prompt được version hóa.
- Có test parser fallback.
- Có cách chạy không cần judge để giữ CI rẻ và deterministic.

`ragas_eval` đứng ở vị trí thứ sáu vì sau năm technique đầu, repo đã có đủ retrieval variants để evaluation nâng cao bắt
đầu có ý nghĩa.

## 7. `crag_2024`: Corrective RAG Khi Retrieval Không Đủ Tốt

Corrective RAG đặt ra một câu hỏi rất production: nếu retrieved documents không đủ tốt thì sao?

Naive RAG thường giả định retriever trả về context đúng. Nhưng trong thực tế, retriever có thể trả về evidence thiếu,
nhiễu hoặc sai. CRAG thêm một bước đánh giá retrieval quality và quyết định có cần correction hay không.

Trong repo, `crag_2024` nên được triển khai theo hướng practical workflow:

```text
question
  -> initial retrieval
  -> retrieval evaluator
  -> if good: normal generation
  -> if ambiguous/poor: corrective retrieval or expansion
  -> final context
  -> generation
```

Vì repo không nên phụ thuộc web search ngay từ đầu, fallback nên có hai mode:

- Local expansion: retrieve thêm bằng query rewrite, broader top-k hoặc alternative retriever.
- External expansion: optional hook cho web/search connector về sau.

Điểm quan trọng nhất là CRAG phải ghi decision trace. Nếu hệ thống quyết định correction, report phải cho biết vì sao:
score thấp, citation thiếu, context không support answer, hoặc judge đánh giá evidence yếu.

Tiêu chí hoàn thành:

- Có retrieval evaluator stage.
- Có decision: `accept`, `expand`, `correct`, `abstain`.
- Có trace trong report.
- Có benchmark trên câu hỏi mà baseline retrieval miss.

CRAG nên triển khai sau `ragas_eval`, vì cần evaluation signal đủ tốt để biết retrieval có vấn đề thật hay không.

## 8. `adaptive_rag_2024`: Route Theo Độ Phức Tạp Của Câu Hỏi

Không phải câu hỏi nào cũng cần pipeline phức tạp. Một câu hỏi fact lookup đơn giản có thể trả lời bằng BM25 hoặc hybrid
retrieval. Một câu hỏi multi-hop có thể cần query decomposition hoặc iterative retrieval. Một câu hỏi global summary có
thể cần graph/hierarchy.

Adaptive-RAG tập trung vào routing: phân loại độ phức tạp của câu hỏi và chọn strategy phù hợp.

Trong repo, `adaptive_rag_2024` nên là bước đầu tiên đưa `controller` vào framework, nhưng chỉ sau khi CRAG đã chứng minh
cần một lớp quyết định dùng chung.

Route ban đầu nên đơn giản:

- `simple`: dùng hybrid baseline.
- `semantic`: dùng dense hoặc query rewrite.
- `ambiguous`: dùng RAG-Fusion hoặc CRAG.
- `multi_hop`: dùng strategy multi-hop khi có.
- `global`: dùng graph/hierarchical retrieval khi có.

Ở giai đoạn đầu, route `multi_hop` và `global` có thể fallback về strategy hiện có, nhưng report vẫn phải ghi route
decision. Điều này giúp benchmark sau này đo được router có chọn đúng không.

Tiêu chí hoàn thành:

- Có query complexity classifier.
- Có route config rõ ràng.
- Có per-question route trace.
- Có metric route distribution và route-level performance.
- Có fallback khi classifier lỗi hoặc confidence thấp.

Adaptive-RAG là cầu nối giữa retrieval technique và hệ thống RAG có orchestration.

## 9. `raptor_2024`: Hierarchical Retrieval Cho Tài Liệu Dài

RAPTOR giải quyết một vấn đề khác: flat chunks không phải lúc nào cũng đủ. Với tài liệu dài, evidence có thể nằm ở nhiều
đoạn, nhiều section hoặc nhiều cấp abstraction. Nếu chỉ retrieve các chunk nhỏ, hệ thống có thể thiếu bức tranh tổng thể.
Nếu chỉ retrieve chunk lớn, retrieval sẽ kém chính xác và context tốn token.

RAPTOR xây cây summary: chunk nhỏ ở leaf, summary ở các level cao hơn. Khi query đến, hệ thống có thể retrieve ở nhiều
cấp khác nhau.

Trong repo, `raptor_2024` nên được triển khai sau khi artifact manifest đã đủ tốt, vì nó cần lưu thêm cấu trúc cây:

- Leaf nodes.
- Summary nodes.
- Parent-child links.
- Level metadata.
- Summary generation cost.

Implementation target nên là paper-inspired, không cần reproduction đầy đủ. Điều quan trọng là artifact inspect được và
benchmark được.

Các câu hỏi RAPTOR nên cải thiện:

- Câu hỏi cần overview.
- Câu hỏi cần tổng hợp nhiều section.
- Câu hỏi mà evidence nằm rải rác.
- Câu hỏi cần hiểu cấu trúc tài liệu dài.

Tiêu chí hoàn thành:

- Có tree index artifact.
- Có summary generation stage.
- Có multi-level retriever.
- Report có build stats: số summary nodes, số level, build latency, estimated cost.
- Có benchmark riêng cho long-document hoặc multi-section QA.

## 10. `graphrag_2024`: Graph Retrieval Cho Global Questions

GraphRAG nên đứng sau RAPTOR vì nó còn phức tạp hơn. Nếu RAPTOR xây hierarchy từ text summaries, GraphRAG xây graph từ
entities, relations và communities. Nó đặc biệt hữu ích với câu hỏi global trên corpus lớn: những câu hỏi không thể trả
lời tốt bằng cách retrieve vài chunk top-k.

Ví dụ:

- "Các chính sách nhân sự nào thường liên quan đến lương thưởng?"
- "Những nhóm vấn đề chính trong toàn bộ corpus là gì?"
- "Các entity nào kết nối nhiều chủ đề nhất?"

Trong repo, `graphrag_2024` nên bắt đầu bằng bản concept/practical, không cố reproduction toàn bộ pipeline GraphRAG.

Các component cần có:

- Entity extractor.
- Relation extractor.
- Graph artifact.
- Community hoặc cluster summary.
- Local query mode.
- Global query mode.

GraphRAG có rủi ro cao về cost và complexity. Vì vậy, acceptance criteria phải chặt:

- Artifact inspect phải hiển thị nodes, edges, communities.
- Build report phải có cost.
- Query report phải ghi mode local/global.
- Phải có benchmark mà flat retrieval thật sự yếu hơn.

Nếu không có global-question dataset, GraphRAG sẽ rất dễ trở thành demo đẹp nhưng không chứng minh được giá trị.

## 11. `ircot_2022`: Interleaving Reasoning Và Retrieval

IRCoT đưa repo sang nhóm multi-hop retrieval. Ý tưởng chính là không retrieve một lần duy nhất. Thay vào đó, hệ thống xen
kẽ reasoning và retrieval: suy luận bước tiếp theo cần biết gì, tạo subquery, retrieve evidence, cập nhật reasoning, rồi
lặp lại.

Technique này phù hợp cho câu hỏi mà evidence không nằm trong một passage duy nhất.

Ví dụ:

- Cần nối thông tin từ hai tài liệu.
- Cần tìm một entity trung gian.
- Cần so sánh chính sách ở hai section.
- Cần trả lời sau nhiều bước truy xuất.

Trong repo, `ircot_2022` nên được triển khai với trace schema chặt:

```text
step_id
thought
subquery
retrieved_contexts
intermediate_answer
stop_reason
```

Không có trace thì multi-hop RAG gần như không debug được. Khi answer sai, ta cần biết sai ở bước reasoning, subquery,
retrieval hay synthesis cuối.

Tiêu chí hoàn thành:

- Có max-step guard.
- Có cost guard.
- Có structured trace trong report.
- Có synthetic multi-hop QA hoặc dataset riêng.
- Có comparison với single-shot retrieval.

IRCoT nên làm trước ReAct vì nó tập trung hẹp hơn vào retrieval + reasoning, ít rủi ro hơn agent tool loop tổng quát.

## 12. `react_rag`: Agentic RAG Có Kiểm Soát

ReAct là bước cuối trong danh sách này vì nó mở ra agentic workflow: model vừa reasoning vừa chọn action. Trong RAG, action
có thể là search local corpus, inspect artifact, retrieve by metadata, query graph, hoặc gọi một verifier.

ReAct rất hấp dẫn, nhưng cũng dễ biến thành hệ thống khó kiểm soát nếu không có guardrails. Vì vậy, chỉ nên triển khai khi
repo đã có:

- Retrieval baseline mạnh.
- Query transformation.
- Evaluation/judge.
- Corrective/adaptive controller.
- Hierarchical/graph retrieval.
- Multi-hop trace.
- Cost và step guard.

Trong repo, `react_rag` nên được giới hạn rõ:

- Tool set nhỏ và explicit.
- JSON action schema.
- Max step.
- Max cost.
- No hidden tool call.
- Full trace trong report.

Action ban đầu nên đơn giản:

- `retrieve`.
- `rewrite_query`.
- `inspect_context`.
- `verify_answer`.
- `final_answer`.

Tiêu chí hoàn thành:

- Agent trace đầy đủ.
- Tool call parse fallback.
- Guard khi model lặp vô hạn.
- Evaluation trên multi-hop/adversarial questions.
- So sánh với IRCoT và Adaptive-RAG.

ReAct không nên được xem là "kỹ thuật tốt hơn mọi thứ". Nó là một orchestration pattern mạnh, nhưng chỉ đáng dùng khi câu
hỏi thật sự cần nhiều bước và nhiều action.

## Nhìn Tổng Thể: Repo Sẽ Tiến Hóa Như Thế Nào?

Nếu triển khai theo thứ tự này, `rag-pipeline-lab` sẽ phát triển qua bốn giai đoạn rõ ràng.

Giai đoạn đầu là retrieval credibility:

- `bm25_hybrid_rerank`
- `dpr_2020`
- `lost_in_middle_context_ordering`

Sau giai đoạn này, repo có baseline đủ mạnh và hiểu rõ retrieval/context construction ảnh hưởng tới answer ra sao.

Giai đoạn hai là recall và evaluation:

- `rewrite_retrieve_read_2023`
- `rag_fusion_2024_v2`
- `ragas_eval`

Sau giai đoạn này, repo có nhiều cách cải thiện recall và đủ evaluation để không bị đánh lừa bởi demo hay.

Giai đoạn ba là control và long-document reasoning:

- `crag_2024`
- `adaptive_rag_2024`
- `raptor_2024`
- `graphrag_2024`

Sau giai đoạn này, repo bắt đầu giống một research platform thật sự: có controller, correction, hierarchy và graph.

Giai đoạn bốn là multi-hop và agentic RAG:

- `ircot_2022`
- `react_rag`

Đây là lúc repo có thể xử lý câu hỏi nhiều bước, nhưng vẫn giữ được trace, cost guard và evaluation.

## Kết Luận

Thứ tự triển khai này cố tình không chạy theo kỹ thuật nghe "hot" nhất trước. Nó đi từ thứ khó thay thế nhất trong một RAG
lab: baseline mạnh, retrieval metrics, context construction, benchmark reproducibility và evaluation.

Khi nền móng đó đã chắc, các kỹ thuật như CRAG, Adaptive-RAG, RAPTOR, GraphRAG, IRCoT hay ReAct mới có đất để chứng minh
giá trị thật. Ngược lại, nếu triển khai chúng quá sớm, repo có thể trông phức tạp nhưng rất khó trả lời câu hỏi quan trọng
nhất: kỹ thuật này có thật sự tốt hơn baseline không, tốt hơn ở loại câu hỏi nào, và cái giá phải trả là gì?

Một RAG research repo tốt không chỉ implement nhiều paper. Nó phải giúp người đọc hiểu tradeoff. Roadmap này được thiết kế
để làm đúng điều đó.
