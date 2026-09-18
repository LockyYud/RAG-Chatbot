from __future__ import annotations

from pathlib import Path
from typing import Any

from ragbench.benchmarks.runner import _row
from ragbench.core.schema import ArtifactManifest, EvalItem, RAGAnswer
from ragbench.evaluation.metrics import evaluate_prediction_rows, evaluate_predictions


def _prediction_with_relevant_doc_at_rank(rank: int, total: int = 10) -> RAGAnswer:
    """A prediction whose *context* is empty (so context-based recall@k is 0
    regardless of k) but whose raw retriever ranking — as ``run_eval()``
    records it once, at ``max(cutoffs)`` depth, in
    ``metadata.retrieved_doc_ids`` — has the relevant doc at 1-indexed
    ``rank``. Isolates the raw-metric cutoff bug from the context-based one.
    """
    doc_ids = [f"other-{i}" for i in range(total)]
    doc_ids[rank - 1] = "relevant-doc"
    return RAGAnswer(
        query="Q",
        answer="",
        contexts=[],
        metadata={"retrieved_chunk_ids": [], "retrieved_doc_ids": doc_ids},
    )


def test_raw_recall_respects_cutoff_not_the_full_metadata_ranking() -> None:
    """Regression: ``run_eval()`` evaluates multiple cutoffs from one
    max-depth prediction (``_ensure_evaluation_depth`` raises query depth to
    ``max(cutoffs)`` once, then ``metrics_by_cutoff`` scores that same
    prediction at every cutoff). ``_attach_raw_retrieval_scores`` must slice
    its metadata ranking to ``k`` the same way the context-based metrics
    slice ``prediction.contexts[:k]`` — before this fix it read the full
    ranking regardless of ``k``, so a relevant doc at rank 5 counted as a hit
    even at raw_recall@1.
    """
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["relevant-doc"])
    prediction = _prediction_with_relevant_doc_at_rank(rank=5)

    rows_at_1 = evaluate_prediction_rows([item], [prediction], k=1)
    rows_at_5 = evaluate_prediction_rows([item], [prediction], k=5)
    rows_at_10 = evaluate_prediction_rows([item], [prediction], k=10)

    assert rows_at_1[0]["raw_recall"] == 0.0
    assert rows_at_5[0]["raw_recall"] == 1.0
    assert rows_at_10[0]["raw_recall"] == 1.0


def test_raw_ndcg_and_map_respect_cutoff() -> None:
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["relevant-doc"])
    prediction = _prediction_with_relevant_doc_at_rank(rank=5)

    rows_at_1 = evaluate_prediction_rows([item], [prediction], k=1)
    rows_at_5 = evaluate_prediction_rows([item], [prediction], k=5)

    assert rows_at_1[0]["raw_ndcg"] == 0.0
    assert rows_at_5[0]["raw_ndcg"] > 0.0
    assert rows_at_1[0]["raw_map"] == 0.0
    assert rows_at_5[0]["raw_map"] > 0.0


def test_raw_metrics_absent_when_metadata_predates_the_field() -> None:
    """Older checkpoints/fixtures without ``retrieved_chunk_ids``/
    ``retrieved_doc_ids`` must not crash or silently score as 0 — the raw
    keys should simply be missing from the row."""
    item = EvalItem(question_id="q1", question="Q", expected_doc_ids=["relevant-doc"])
    prediction = RAGAnswer(query="Q", answer="", contexts=[], metadata={})

    rows = evaluate_prediction_rows([item], [prediction], k=5)
    assert "raw_recall" not in rows[0]

    metrics = evaluate_predictions([item], [prediction], k=5)
    assert "raw_recall_at_5" not in metrics


def _fake_manifest() -> ArtifactManifest:
    return {
        "pipeline": {
            "id": "parent_child",
            "implementation_level": "test",
            "config": {},
            "config_fingerprint": "sha256:cfg",
        },
        "corpus": {
            "fingerprint": "sha256:corpus",
            "documents": ["d1"],
            "document_count": 1,
            "block_count": 1,
            "chunk_count": 1,
            "node_count": 1,
        },
        "store": {"backend": "json_memory"},
    }


def test_row_flattens_raw_metrics_for_non_headline_cutoffs(tmp_path: Path) -> None:
    """Regression: ``_row()`` only copied ``recall_at_``/``ndcg_at_``/
    ``map_at_``/``context_precision_at_`` prefixed keys out of
    ``metrics_by_cutoff`` into the benchmark row. A cutoff other than the
    headline ``top_k`` (whose metrics arrive separately via
    ``evaluation["metrics"]``) had its ``raw_recall_at_*`` /
    ``raw_ndcg_at_*`` / ``raw_map_at_*`` values silently dropped, so
    ``_comparisons()`` could never report a delta/CI for them.
    """
    evaluation: dict[str, Any] = {
        "predictions": [],
        "index": {},
        "metrics": {"recall_at_10": 0.9, "raw_recall_at_10": 0.9},
        "metrics_by_cutoff": {
            "2": {
                "recall_at_2": 0.5,
                "raw_recall_at_2": 0.5,
                "ndcg_at_2": 0.4,
                "raw_ndcg_at_2": 0.4,
                "map_at_2": 0.3,
                "raw_map_at_2": 0.3,
            },
            "10": {"recall_at_10": 0.9, "raw_recall_at_10": 0.9},
        },
    }
    row = _row(
        technique="parent_child",
        artifact=tmp_path,
        report=tmp_path / "report.json",
        manifest=_fake_manifest(),
        evaluation=evaluation,
        status="ok",
    )
    assert row["raw_recall_at_2"] == 0.5
    assert row["raw_ndcg_at_2"] == 0.4
    assert row["raw_map_at_2"] == 0.3
