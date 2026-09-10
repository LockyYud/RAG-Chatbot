"""Tests for synthetic benchmark generation v2 — see docs/synthetic_benchmark_v2.md."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ragbench.core.io import read_json, read_jsonl, write_json, write_jsonl
from ragbench.core.schema import EvalItem, RAGAnswer, RetrievalResult
from ragbench.datasets.synthetic import (
    GenerationConfig,
    SyntheticBenchmarkBuilder,
    chunk_corpus,
    lexical_overlap,
    load_synthetic_manifest,
    role_collisions,
    rouge_l_recall,
    span_in_text,
    validate_synthetic_dataset,
)
from ragbench.evaluation.audit import (
    emit_label_sample,
    kendall_tau_b,
    label_precision,
    load_recorded_audit,
    rank_agreement,
    row_metric_key,
    run_audit,
    write_label_sample,
)
from ragbench.evaluation.metrics import evaluate_prediction_rows, retrieval_label_level
from ragbench.evaluation.trust import TRUSTED_FOR_RANKING, UNTRUSTED, build_trust_block


@dataclass
class _Completion:
    text: str


class FakeChatClient:
    """Scripted stand-in for LLMClient, routed by which prompt is being answered."""

    def __init__(
        self,
        *,
        questions_per_chunk: int = 1,
        answerable: bool = True,
        pool_grades: list[dict[str, int]] | None = None,
        question_text: str | None = None,
        question_type: str = "factual",
        rewrite_to: str | None = None,
    ) -> None:
        self.questions_per_chunk = questions_per_chunk
        self.answerable = answerable
        self.pool_grades = pool_grades or []
        self.question_text = question_text
        self.question_type = question_type
        self.rewrite_to = rewrite_to
        self.calls: list[str] = []

    def create_chat_completion(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.0, max_tokens: int = 600
    ) -> _Completion:
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if system.startswith("Generate Vietnamese RAG evaluation questions"):
            self.calls.append("generate")
            return _Completion(json.dumps(self._questions(user)))
        if system.startswith("Answer the question using ONLY the context"):
            self.calls.append("verify")
            span = self._span(user.split("CONTEXT:\n", 1)[1].split("\n\nQUESTION:", 1)[0])
            return _Completion(json.dumps({"answerable": self.answerable, "answer": span}))
        if system.startswith("Rewrite the question"):
            self.calls.append("rewrite")
            return _Completion(json.dumps({"question": self.rewrite_to or "cau hoi da viet lai"}))
        if system.startswith("Grade how well each passage"):
            self.calls.append("pool")
            return _Completion(json.dumps(self.pool_grades))
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    def _questions(self, user: str) -> list[dict[str, Any]]:
        context = user.split("CONTEXT:\n", 1)[1]
        span = self._span(context)
        question = self.question_text or f"cau hoi ve {span}"
        return [
            {
                "question": question,
                "answer_span": "" if self.question_type == "unanswerable" else span,
                "ground_truth_answer": span,
                "question_type": self.question_type,
                "difficulty": "medium",
            }
            for _ in range(self.questions_per_chunk)
        ]

    @staticmethod
    def _span(context: str) -> str:
        return " ".join(context.split()[:6])


def _trust(manifest: dict[str, Any] | None, audit: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
    block = build_trust_block(manifest, audit, **kwargs)
    assert block is not None
    return block


def _write_corpus(root: Path, documents: dict[str, str]) -> Path:
    for name, text in documents.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _config(**overrides: Any) -> GenerationConfig:
    defaults = {
        "generator_model": "model-generator",
        "verifier_model": "model-verifier",
        "questions_per_chunk": 1,
        "chunk_size": 40,
        "chunk_overlap": 0,
        "pool_k": 5,
    }
    return GenerationConfig(**{**defaults, **overrides})


# ─── Text measures ───────────────────────────────────────────────────────────


def test_span_in_text_matches_on_tokens_not_raw_substring() -> None:
    """Chunk text has been through split_tokens (lowercased, punctuation gone),
    so a verbatim span from the source document is not a raw substring of it."""
    chunk_text = "nhan vien thu viec duoc nghi phep nam 12 ngay"
    assert span_in_text("Nhân viên thử việc, được nghỉ phép", "nhan vien thu viec duoc nghi phep") is False
    assert span_in_text("duoc nghi phep nam", chunk_text) is True
    assert span_in_text("nghi phep 12", chunk_text) is False  # not contiguous


def test_lexical_overlap_is_question_containment_not_jaccard() -> None:
    """A short question fully lifted from a long passage must score 1.0; Jaccard
    would report ~0.1 purely because the passage is long."""
    passage = " ".join(["tu"] * 50 + ["nghi", "phep", "nam"])
    assert lexical_overlap("nghi phep nam", passage) == 1.0
    assert lexical_overlap("hoan toan khac biet", passage) == 0.0


def test_rouge_l_recall_measures_reference_coverage() -> None:
    assert rouge_l_recall("a b c", "a b c") == 1.0
    assert rouge_l_recall("a b c", "x a x c x") == pytest.approx(2 / 3)
    assert rouge_l_recall("a b c", "") == 0.0


# ─── The v1 defect ───────────────────────────────────────────────────────────


def test_chunk_labels_from_a_foreign_chunker_zero_every_metric_silently() -> None:
    """Regression documenting exactly what v1 did.

    v1 wrote its own ``doc:synthetic:0001`` ids into expected_chunk_ids while
    pipelines emit ``doc:c1:<sha1>``. Because metrics prefer chunk labels
    whenever present, the two families never intersect and every retrieval
    score is 0 — while retrieval_evaluated stays True, so nothing looks wrong.
    """
    item = EvalItem(
        question_id="q1",
        question="q",
        expected_doc_ids=["policy"],
        expected_chunk_ids=["policy:synthetic:0001"],
    )
    prediction = RAGAnswer(
        query="q",
        answer="",
        contexts=[
            RetrievalResult(node_id="n", chunk_id="policy:c1:abcdef0123", doc_id="policy", text="t", score=1.0, rank=1)
        ],
    )
    (row,) = evaluate_prediction_rows([item], [prediction], k=5)
    assert row["retrieval_evaluated"] is True
    assert row["recall"] == 0.0 and row["mrr"] == 0.0 and row["ndcg"] == 0.0
    # The diagnostic that makes this visible instead of silent.
    assert retrieval_label_level([item]) == "chunk"


def test_document_labels_score_correctly_across_chunkers() -> None:
    """The v2 shape: doc-level labels survive any chunker, so the same
    retrieval that scored 0 above scores 1.0."""
    item = EvalItem(question_id="q1", question="q", expected_doc_ids=["policy"], expected_chunk_ids=[])
    prediction = RAGAnswer(
        query="q",
        answer="",
        contexts=[
            RetrievalResult(node_id="n", chunk_id="policy:c1:abcdef0123", doc_id="policy", text="t", score=1.0, rank=1)
        ],
    )
    (row,) = evaluate_prediction_rows([item], [prediction], k=5)
    assert row["recall"] == 1.0 and row["mrr"] == 1.0
    assert retrieval_label_level([item]) == "doc"


def test_generated_rows_are_scorable_against_pipeline_retrieval(tmp_path: Path) -> None:
    """End to end: what the builder writes must actually score above zero when
    fed to the metrics with a retrieval that found the right document."""
    docs = _write_corpus(tmp_path / "corpus", {"policy.md": "nhan vien thu viec duoc nghi phep nam muoi hai ngay"})
    builder = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient())
    builder.build(docs, tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "qa.jsonl")
    assert rows, "generation produced no questions"
    items = [EvalItem.from_dict(row) for row in rows]
    predictions = [
        RAGAnswer(
            query=item.question,
            answer="",
            contexts=[
                RetrievalResult(
                    node_id="n",
                    chunk_id="policy:c1:deadbeef00",
                    doc_id=item.expected_doc_ids[0],
                    text="t",
                    score=1.0,
                    rank=1,
                )
            ],
        )
        for item in items
    ]
    scored = evaluate_prediction_rows(items, predictions, k=5)
    assert all(row["recall"] == 1.0 for row in scored)


# ─── Generation stages ───────────────────────────────────────────────────────


def test_builder_refuses_to_let_one_model_hold_two_roles(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam"})
    config = _config(generator_model="same-model", verifier_model="same-model")
    with pytest.raises(ValueError, match="Model roles must differ"):
        SyntheticBenchmarkBuilder(config, client=FakeChatClient()).build(docs, tmp_path / "out")


def test_role_collisions_reports_every_shared_pair() -> None:
    assert role_collisions({"generator": "m", "verifier": "m", "judge": "other"}) == ["generator==verifier"]
    assert role_collisions({"generator": "m", "verifier": "n", "judge": "m"}) == ["generator==judge"]
    assert role_collisions({"generator": "a", "verifier": "b"}) == []


def test_span_that_is_not_in_the_passage_is_rejected_without_an_api_call(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})

    class Fabricating(FakeChatClient):
        def _questions(self, user: str) -> list[dict[str, Any]]:
            return [
                {
                    "question": "cau hoi",
                    "answer_span": "hoan toan khong co trong doan van",
                    "ground_truth_answer": "x",
                    "question_type": "factual",
                    "difficulty": "medium",
                }
            ]

    client = Fabricating()
    builder = SyntheticBenchmarkBuilder(_config(), client=client)
    manifest = builder.build(docs, tmp_path / "out")
    assert manifest["stages"]["span_reject"] >= 1
    assert manifest["stages"]["final"] == 0
    assert "verify" not in client.calls  # rejected before any verifier spend


def test_unverifiable_question_is_dropped_and_counted(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    builder = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient(answerable=False))
    manifest = builder.build(docs, tmp_path / "out")
    assert manifest["stages"]["answerability_reject"] >= 1
    assert manifest["stages"]["final"] == 0
    assert manifest["stages"]["answerability_reject_rate"] > 0


def test_pooling_adds_relevant_documents_beyond_the_source(tmp_path: Path) -> None:
    """A corpus that repeats a fact must not label only the passage that
    happened to seed the question — that is what penalises good retrievers."""
    text = "nhan vien duoc nghi phep nam muoi hai ngay theo quy dinh cua cong ty"
    docs = _write_corpus(tmp_path / "corpus", {"policy.md": text, "handbook.md": text})
    client = FakeChatClient(pool_grades=[{"index": 0, "grade": 2}])
    builder = SyntheticBenchmarkBuilder(_config(), client=client)
    manifest = builder.build(docs, tmp_path / "out")
    rows = read_jsonl(tmp_path / "out" / "qa.jsonl")
    assert rows
    assert any(len(row["expected_doc_ids"]) > 1 for row in rows)
    assert manifest["stages"]["extra_relevant_mean"] > 0
    graded = rows[0]["metadata"]["relevance_by_doc_id"]
    assert set(graded) == set(rows[0]["expected_doc_ids"])


def test_unanswerable_question_is_relabelled_when_the_pool_answers_it(tmp_path: Path) -> None:
    """ "Unanswerable" from the generator only ever meant "unanswerable from one
    passage"; if the corpus answers it, the label was wrong."""
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam", "b.md": "chin muoi mot hai"})
    client = FakeChatClient(question_type="unanswerable", pool_grades=[{"index": 0, "grade": 2}])
    builder = SyntheticBenchmarkBuilder(_config(), client=client)
    manifest = builder.build(docs, tmp_path / "out")
    assert manifest["stages"]["relabelled_from_unanswerable"] >= 1
    rows = read_jsonl(tmp_path / "out" / "qa.jsonl")
    relabelled = [row for row in rows if row["metadata"]["relabelled_from_unanswerable"]]
    assert relabelled
    assert relabelled[0]["metadata"]["is_answerable"] is True
    assert relabelled[0]["expected_doc_ids"]


def test_high_overlap_questions_are_rewritten_and_recorded(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    # The scripted question copies the passage wording outright, so overlap is 1.0.
    client = FakeChatClient(question_text="mot hai ba", rewrite_to="mot hai ba")
    builder = SyntheticBenchmarkBuilder(_config(overlap_threshold=0.5), client=client)
    manifest = builder.build(docs, tmp_path / "out")
    assert manifest["stages"]["rewrite"] >= 1
    # The rewrite came back identical, so the original is kept and the bias flagged.
    assert manifest["stages"]["rewrite_reject"] >= 1
    rows = read_jsonl(tmp_path / "out" / "qa.jsonl")
    assert rows[0]["metadata"]["high_overlap"] is True
    assert manifest["stages"]["lexical_overlap_p90"] > 0.5


def test_near_identical_questions_are_deduplicated(tmp_path: Path) -> None:
    docs = _write_corpus(
        tmp_path / "corpus",
        {"a.md": "mot hai ba bon nam sau bay tam", "b.md": "mot hai ba bon nam sau bay tam"},
    )
    client = FakeChatClient(question_text="cung mot cau hoi giong het nhau")
    builder = SyntheticBenchmarkBuilder(_config(), client=client)
    manifest = builder.build(docs, tmp_path / "out")
    assert manifest["stages"]["dedup_removed"] >= 1
    rows = read_jsonl(tmp_path / "out" / "qa.jsonl")
    assert len({row["question"] for row in rows}) == len(rows)


def test_manifest_records_provenance_and_forces_dev_split(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    manifest = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(
        docs, tmp_path / "out", dataset_id="unit"
    )
    assert manifest["metadata"] == {"protocol_split": "dev", "generated": True}
    assert manifest["corpus"]["fingerprint"].startswith("sha256:")
    assert manifest["models"] == {
        "generator": "model-generator",
        "verifier": "model-verifier",
        "pool_embedding": None,
    }
    assert manifest["pool_retrievers"] == ["bm25"]
    assert read_json(tmp_path / "out" / "manifest.json")["dataset_id"] == "unit"


def test_regenerating_the_same_corpus_reproduces_the_fingerprint(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    first = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "one")
    second = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "two")
    assert first["corpus"]["fingerprint"] == second["corpus"]["fingerprint"]


def test_chunk_corpus_disambiguates_same_named_files_in_subdirectories(tmp_path: Path) -> None:
    _write_corpus(tmp_path, {"legal/report.md": "legal " * 50, "finance/report.md": "finance " * 50})
    chunks = chunk_corpus(tmp_path, chunk_size=1000, chunk_overlap=0)
    assert {chunk.doc_id for chunk in chunks} == {"legal/report", "finance/report"}


# ─── Validation ──────────────────────────────────────────────────────────────


def test_validate_synthetic_rejects_chunk_labels(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "qa.jsonl").write_text(
        json.dumps(
            {
                "question_id": "syn_0001",
                "question": "q",
                "expected_doc_ids": ["a"],
                "expected_chunk_ids": ["a:synthetic:0001"],
                "metadata": {"is_answerable": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps({"metadata": {"generated": True, "protocol_split": "dev"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="expected_chunk_ids must be empty"):
        validate_synthetic_dataset(target)


def test_load_synthetic_manifest_ignores_a_fixed_dataset(tmp_path: Path) -> None:
    target = tmp_path / "fixed"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps({"metadata": {"protocol_split": "test"}}), encoding="utf-8")
    assert load_synthetic_manifest(target) is None


def test_generated_dataset_passes_its_own_validator(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "out")
    summary = validate_synthetic_dataset(tmp_path / "out")
    assert summary["questions"] >= 1
    assert summary["generator_version"] == "v2"


# ─── Audit ───────────────────────────────────────────────────────────────────


def test_kendall_tau_b_agrees_and_disagrees_as_expected() -> None:
    assert kendall_tau_b([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0
    assert kendall_tau_b([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == -1.0
    assert kendall_tau_b([1.0], [1.0]) is None  # one technique cannot be an ordering
    assert kendall_tau_b([1.0, 1.0], [1.0, 1.0]) is None  # all ties


def test_row_metric_key_strips_only_a_real_cutoff_suffix() -> None:
    assert row_metric_key("ndcg_at_10") == "ndcg"
    assert row_metric_key("recall_at_5") == "recall"
    assert row_metric_key("mrr") == "mrr"
    assert row_metric_key("hit_rate") == "hit_rate"


def _rows(values: dict[str, float]) -> list[dict[str, Any]]:
    return [{"question_id": qid, "ndcg": value} for qid, value in values.items()]


def test_rank_agreement_reports_agreement_between_two_orderings() -> None:
    synthetic = {"a": _rows({"q1": 0.9, "q2": 0.9}), "b": _rows({"q1": 0.5, "q2": 0.5})}
    golden = {"a": _rows({"g1": 0.8, "g2": 0.8}), "b": _rows({"g1": 0.4, "g2": 0.4})}
    result = rank_agreement(synthetic, golden, metrics=("ndcg_at_10",), samples=20)
    assert result["techniques"] == ["a", "b"]
    assert result["per_metric"]["ndcg_at_10"]["kendall_tau"] == 1.0


def test_rank_agreement_detects_a_reversed_ordering() -> None:
    synthetic = {"a": _rows({"q1": 0.9}), "b": _rows({"q1": 0.5})}
    golden = {"a": _rows({"g1": 0.2}), "b": _rows({"g1": 0.7})}
    result = rank_agreement(synthetic, golden, metrics=("ndcg_at_10",), samples=20)
    assert result["per_metric"]["ndcg_at_10"]["kendall_tau"] == -1.0


def test_rank_agreement_needs_two_shared_techniques() -> None:
    with pytest.raises(ValueError, match="at least two techniques"):
        rank_agreement({"a": _rows({"q1": 1.0})}, {"a": _rows({"g1": 1.0})}, metrics=("ndcg_at_10",), samples=5)


def test_label_sample_is_emitted_per_question_document_pair() -> None:
    rows = [
        {
            "question_id": "syn_0001",
            "question": "q",
            "expected_doc_ids": ["a", "b"],
            "expected_citations": ["a"],
            "metadata": {"relevance_by_doc_id": {"a": 2, "b": 1}},
        }
    ]
    sample = emit_label_sample(rows, sample_size=10)
    assert len(sample) == 2
    assert {row["doc_id"] for row in sample} == {"a", "b"}
    assert all(row["correct"] is None for row in sample)
    assert sum(1 for row in sample if row["is_source"]) == 1


def test_label_precision_ignores_unreviewed_rows() -> None:
    reviewed = label_precision(
        [
            {"question_id": "q1", "correct": True},
            {"question_id": "q1", "correct": False},
            {"question_id": "q2", "correct": True},
            {"question_id": "q3", "correct": None},
        ]
    )
    assert reviewed["reviewed_pairs"] == 3
    assert reviewed["pair_precision"] == pytest.approx(2 / 3)
    # q1 had a wrong label, so only q2 is a fully correct question.
    assert reviewed["question_precision"] == pytest.approx(1 / 2)


def test_label_precision_of_an_untouched_file_is_not_a_perfect_score() -> None:
    assert label_precision([{"question_id": "q1", "correct": None}])["pair_precision"] is None


# ─── Trust ───────────────────────────────────────────────────────────────────


def _manifest(**overrides: Any) -> dict[str, Any]:
    base = {
        "models": {"generator": "gen", "verifier": "ver"},
        "same_model_roles": [],
        "pool_retrievers": ["bm25", "dense"],
        "stages": {
            "answerability_reject_rate": 0.15,
            "lexical_overlap_p90": 0.5,
            "extra_relevant_mean": 0.8,
            "final": 300,
        },
    }
    base.update(overrides)
    return base


def _audit(tau: float = 0.8, ci_low: float = 0.4, precision: float = 0.9) -> dict[str, Any]:
    return {
        "audit_path": "/tmp/audit.json",
        "golden_queries": 80,
        "overall_kendall_tau": {"kendall_tau": tau, "ci95_low": ci_low, "ci95_high": 1.0},
        "rank_agreement": {},
        "label_precision": {"pair_precision": precision, "reviewed_pairs": 40},
    }


def test_a_dataset_with_no_audit_is_never_trusted() -> None:
    block = _trust(_manifest())
    assert block["verdict"] == UNTRUSTED
    assert block["reasons"] == ["not_audited"]
    assert block["audited"] is False


def test_a_clean_audit_earns_ranking_trust_only() -> None:
    block = _trust(_manifest(), _audit())
    assert block["verdict"] == TRUSTED_FOR_RANKING
    assert block["reasons"] == []


def test_low_rank_agreement_blocks_trust() -> None:
    block = _trust(_manifest(), _audit(tau=0.3))
    assert block["verdict"] == UNTRUSTED
    assert "rank_disagreement" in block["reasons"]


def test_a_tau_whose_interval_spans_zero_blocks_trust() -> None:
    block = _trust(_manifest(), _audit(tau=0.6, ci_low=-0.2))
    assert block["verdict"] == UNTRUSTED
    assert "rank_disagreement" in block["reasons"]


def test_noisy_labels_block_trust() -> None:
    block = _trust(_manifest(), _audit(precision=0.5))
    assert block["verdict"] == UNTRUSTED
    assert "label_noise" in block["reasons"]


def test_a_weak_generator_blocks_trust() -> None:
    manifest = _manifest(
        stages={"answerability_reject_rate": 0.6, "lexical_overlap_p90": 0.4, "extra_relevant_mean": 1.0, "final": 10}
    )
    block = _trust(manifest, _audit())
    assert block["verdict"] == UNTRUSTED
    assert "weak_generator" in block["reasons"]


def test_a_judge_that_wrote_the_questions_is_circular() -> None:
    block = _trust(_manifest(), _audit(), judge_model="gen")
    assert block["verdict"] == UNTRUSTED
    assert "circular_judge" in block["reasons"]
    assert "generator==judge" in block["same_model_roles"]


def test_lexical_bias_warns_without_blocking() -> None:
    manifest = _manifest(
        stages={"answerability_reject_rate": 0.1, "lexical_overlap_p90": 0.9, "extra_relevant_mean": 0.5, "final": 300}
    )
    block = _trust(manifest, _audit())
    assert block["verdict"] == TRUSTED_FOR_RANKING
    assert "lexical_bias" in block["warnings"]


def test_bm25_only_pooling_warns_without_blocking() -> None:
    block = _trust(_manifest(pool_retrievers=["bm25"]), _audit())
    assert block["verdict"] == TRUSTED_FOR_RANKING
    assert "single_retriever_pool" in block["warnings"]


def test_a_human_labelled_dataset_gets_no_trust_block() -> None:
    assert build_trust_block(None) is None


# ─── Audit → dataset → report wiring ─────────────────────────────────────────


def _write_report(path: Path, values: dict[str, float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"metrics": {}, "query_metrics": _rows(values)})
    return path


def test_running_an_audit_records_it_on_the_dataset_it_audited(tmp_path: Path) -> None:
    """The dataset carries its own audit, so a later eval finds it without the
    caller having to remember which audit belonged to which dataset."""
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "out")
    qa_path = tmp_path / "out" / "qa.jsonl"

    audit = run_audit(
        synthetic_runs={
            "a": _write_report(tmp_path / "r" / "syn_a.json", {"q1": 0.9}),
            "b": _write_report(tmp_path / "r" / "syn_b.json", {"q1": 0.4}),
        },
        golden_runs={
            "a": _write_report(tmp_path / "r" / "gold_a.json", {"g1": 0.8}),
            "b": _write_report(tmp_path / "r" / "gold_b.json", {"g1": 0.3}),
        },
        output_dir=tmp_path / "audit",
        synthetic_qa_path=qa_path,
        samples=10,
    )
    assert audit["overall_kendall_tau"]["kendall_tau"] == 1.0

    manifest = load_synthetic_manifest(qa_path)
    assert manifest is not None
    assert manifest["audit_path"] == audit["audit_path"]
    recovered = load_recorded_audit(manifest)
    assert recovered is not None and recovered["golden_queries"] == 1

    block = _trust(manifest, recovered)
    assert block["audited"] is True
    # No label review was supplied, so ranking trust is still withheld.
    assert block["verdict"] == UNTRUSTED
    assert "label_noise" in block["reasons"]


def test_an_unaudited_dataset_reports_no_audit_rather_than_failing(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    manifest = SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "out")
    assert load_recorded_audit(manifest) is None
    assert _trust(manifest, None)["reasons"] == ["not_audited"]


def test_a_full_review_pass_earns_ranking_trust(tmp_path: Path) -> None:
    docs = _write_corpus(tmp_path / "corpus", {"a.md": "mot hai ba bon nam sau bay tam chin muoi"})
    SyntheticBenchmarkBuilder(_config(), client=FakeChatClient()).build(docs, tmp_path / "out")
    qa_path = tmp_path / "out" / "qa.jsonl"

    sample = write_label_sample(qa_path, tmp_path / "review.jsonl", sample_size=10)
    assert sample["pairs"] >= 1
    reviewed = [{**row, "correct": True, "reviewer": "duy"} for row in read_jsonl(tmp_path / "review.jsonl")]
    write_jsonl(tmp_path / "review.jsonl", reviewed)

    audit = run_audit(
        synthetic_runs={
            "a": _write_report(tmp_path / "r" / "syn_a.json", {"q1": 0.9}),
            "b": _write_report(tmp_path / "r" / "syn_b.json", {"q1": 0.4}),
        },
        golden_runs={
            "a": _write_report(tmp_path / "r" / "gold_a.json", {"g1": 0.8}),
            "b": _write_report(tmp_path / "r" / "gold_b.json", {"g1": 0.3}),
        },
        output_dir=tmp_path / "audit",
        synthetic_qa_path=qa_path,
        label_review_path=tmp_path / "review.jsonl",
        samples=200,
    )
    assert audit["label_precision"]["pair_precision"] == 1.0
    manifest = load_synthetic_manifest(qa_path)
    block = _trust(manifest, audit)
    assert block["verdict"] == TRUSTED_FOR_RANKING
    # Pooling used BM25 alone here, which is a warning and never a blocker.
    assert block["warnings"] == ["single_retriever_pool"]
