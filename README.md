# RAG Pipeline Lab

`rag-pipeline-lab` is a **paper-driven, evaluation-first RAG Lab** for turning RAG research ideas into runnable end-to-end pipelines.

Repo này là một paper-driven, evaluation-first RAG Lab nhằm chuyển các ý tưởng và chiến lược RAG từ research papers thành những pipeline RAG hoàn chỉnh, có thể chạy end-to-end từ raw documents đến final answers. Thay vì chỉ implement lại code của từng paper một cách rời rạc, repo chuẩn hóa các strategy ở nhiều giai đoạn như data processing, chunking, enrichment, indexing, retrieval, reranking, context construction, generation và verification, sau đó benchmark chúng trên cùng một evaluation protocol. Mục tiêu là giúp AI Engineer/AI Researcher so sánh thực nghiệm các pipeline RAG theo chất lượng trả lời, citation accuracy, faithfulness, latency và cost, từ đó hiểu rõ strategy nào phù hợp với từng loại tài liệu và use case production.

## Repository Shape

```text
raglab/              # Engine: stable interfaces and base components
  core/
  processing/
  indexing/
  inference/
  cli/

techniques/          # Paper/method implementations
  _template/
  naive_rag/
  rag_sequence_2020/
  parent_child/

evaluation/          # First-class evaluation protocol and runner
  metrics/
  protocol/
  runner.py

datasets/            # Sample data plus bring-your-own-data area
  sample/
  user_data/

benchmarks/          # Benchmark runner and ignored local results
  run_all.py
  results/
```

## Technique Catalog

Each technique owns its paper analysis, metadata, runnable config, and optional custom code:

```text
techniques/<technique_id>/
  README.md
  technique.yaml
  config.yaml
  custom/
    register.py
```

Current techniques:

| Technique | Type | Base | Config |
| --- | --- | --- | --- |
| `naive_rag` | baseline | none | `techniques/naive_rag/config.yaml` |
| `rag_sequence_2020` | paper-inspired RAG-Sequence from [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) | `naive_rag` | `techniques/rag_sequence_2020/config.yaml` |
| `parent_child` | production RAG pattern | `naive_rag` | `techniques/parent_child/config.yaml` |
| `hyde_2022` | paper-inspired HyDE from [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) | `rag_sequence_2020` | `techniques/hyde_2022/config.yaml` |
| `rag_fusion_2024` | paper-inspired RAG-Fusion from [arXiv:2402.03367](https://arxiv.org/abs/2402.03367) | `rag_sequence_2020` | `techniques/rag_fusion_2024/config.yaml` |
| `self_rag_2023` | concept-only Self-RAG-inspired critique verifier from [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) | `parent_child` | `techniques/self_rag_2023/config.yaml` |

List techniques from CLI:

```bash
python -m raglab.cli.main techniques list
python -m raglab.cli.main techniques show rag_sequence_2020
```

## Quick Start

Run the local baseline on the sample dataset:

```bash
python -m raglab.cli.main ingest \
  --config techniques/naive_rag/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/naive_rag

python -m raglab.cli.main eval \
  --config techniques/naive_rag/config.yaml \
  --artifact artifacts/naive_rag \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/naive_rag_eval.json
```

Run multiple techniques:

```bash
python benchmarks/run_all.py \
  --techniques naive_rag parent_child \
  --docs datasets/sample/docs \
  --qa datasets/sample/qa.jsonl \
  --output benchmarks/results/sample
```

## Real OpenAI-Compatible Path

Some paper-driven techniques require real embeddings or model generation. Create `.env` from `.env.example`:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini
```

Then run a paper-driven pipeline:

```bash
python -m raglab.cli.main ingest \
  --config techniques/rag_sequence_2020/config.yaml \
  --input datasets/sample/docs \
  --output artifacts/rag_sequence_2020

python -m raglab.cli.main eval \
  --config techniques/rag_sequence_2020/config.yaml \
  --artifact artifacts/rag_sequence_2020 \
  --dataset datasets/sample/qa.jsonl \
  --output benchmarks/results/rag_sequence_2020_eval.json
```

## Bring Your Own Dataset

Use the format described in [datasets/README.md](datasets/README.md). For meaningful comparison, use at least 50-100 labeled QA pairs.

```bash
python benchmarks/run_all.py \
  --techniques naive_rag parent_child rag_sequence_2020 hyde_2022 rag_fusion_2024 self_rag_2023 \
  --docs datasets/user_data/my_docs \
  --qa datasets/user_data/my_qa.jsonl \
  --output benchmarks/results/my_dataset
```

## Add A New Paper Technique

1. Copy `techniques/_template/`.
2. Rename the folder with a stable descriptive id, e.g. `hyde_2022`.
3. Fill `technique.yaml`, including `base`, `stage`, `requires`, `best_for`, and `weak_for`.
4. Write the paper analysis in `README.md`.
5. Compose the pipeline in `config.yaml`.
6. Add custom strategy code under `custom/` only when base components are not enough.
7. Register custom code through `custom/register.py`.
8. Run against the same dataset as the baselines.

Be explicit about whether a technique is a **faithful reproduction**, **paper-inspired adaptation**, **production pattern**, or **baseline**.
