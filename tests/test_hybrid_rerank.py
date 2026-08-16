from __future__ import annotations

import pytest

from raglab.core.schema import RetrievalResult
from raglab.indexing.retrievers import reciprocal_rank_fusion
from raglab.inference.rerankers.cross_encoder import CrossEncoderReranker


def _result(node_id: str, rank: int, score: float = 0.0, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        node_id=node_id,
        chunk_id=node_id,
        doc_id="doc",
        text=text or node_id,
        score=score,
        rank=rank,
    )


def test_rrf_rewards_agreement_across_lists() -> None:
    # "a" is rank 1 in BOTH lists — the doc both retrievers agree on.
    dense = [_result("a", 1), _result("b", 2), _result("c", 3), _result("d", 4)]
    sparse = [_result("a", 1), _result("c", 2), _result("b", 3), _result("d", 4)]

    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60.0))

    # Agreement at the top wins decisively over single-list strength.
    assert max(fused, key=lambda key: fused[key]) == "a"
    # b and c are symmetric (rank 2 in one, rank 3 in the other) → tie.
    assert fused["b"] == fused["c"]
    # d trails everyone (rank 4 in both).
    assert min(fused, key=lambda key: fused[key]) == "d"


def test_rrf_score_matches_formula() -> None:
    dense = [_result("d", 3)]
    sparse = [_result("d", 7)]
    fused = dict(reciprocal_rank_fusion([dense, sparse], k=60.0))
    expected = 1.0 / (60.0 + 3) + 1.0 / (60.0 + 7)
    assert abs(fused["d"] - expected) < 1e-12


def test_rrf_handles_missing_and_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    # "x" only in one list still ranks.
    fused = dict(reciprocal_rank_fusion([[_result("x", 1)], []], k=60.0))
    assert fused["x"] == 1.0 / 61.0


def test_cross_encoder_falls_back_to_lexical_without_extra(monkeypatch) -> None:
    # Force the optional dependency to look absent.
    reranker = CrossEncoderReranker(strict=False)
    object.__setattr__(reranker, "_encoder", None)
    assert reranker.available is False

    # Equal base scores so lexical overlap is the tie-breaker.
    results = [
        _result("a", 1, score=0.5, text="hoàn toàn không liên quan"),
        _result("b", 2, score=0.5, text="điều kiện xét tuyển trí tuệ nhân tạo"),
    ]
    reranked = reranker.rerank("điều kiện xét tuyển trí tuệ nhân tạo", results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].rank == 1
    # Lexical overlap should lift "b" (full term overlap) above "a".
    assert reranked[0].node_id == "b"
    assert reranked[0].metadata["reranker"] == "lexical_overlap_fallback"


def test_cross_encoder_is_strict_by_default(monkeypatch) -> None:
    def unavailable(model: str):
        raise RuntimeError("offline")

    monkeypatch.setattr(CrossEncoderReranker, "_load_encoder", staticmethod(unavailable))
    with pytest.raises(RuntimeError, match="strict mode"):
        CrossEncoderReranker()
