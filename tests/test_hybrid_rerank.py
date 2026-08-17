from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from raglab.core.base import load_pipeline
from raglab.core.doctor import diagnose_technique
from raglab.core.schema import RetrievalResult
from raglab.indexing.retrievers import reciprocal_rank_fusion
from raglab.inference.rerankers.cross_encoder import CrossEncoderReranker, effective_reranker_name
from raglab.providers.llm_client import LLMClient, capture_provider_usage


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


class _FakeRerankResponse:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results


def _fake_litellm_rerank(handler: Any) -> Any:
    def rerank(model: str, query: str, documents: list[str], top_n: int, timeout: float) -> _FakeRerankResponse:
        return handler(model, query, documents, top_n)

    return SimpleNamespace(rerank=rerank)


def test_cross_encoder_api_backend_never_loads_a_local_model(monkeypatch) -> None:
    # backend="api" must not touch sentence-transformers at construction time —
    # a bad/unreachable API model id should only fail on the first real call.
    monkeypatch.setattr(
        CrossEncoderReranker,
        "_load_encoder",
        staticmethod(lambda model: (_ for _ in ()).throw(RuntimeError("should never be called"))),
    )
    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api", strict=True)
    assert reranker.available is True


def test_cross_encoder_api_backend_scores_and_reorders(monkeypatch) -> None:
    def handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        # Reverse relevance vs input order, proving re-sorting actually happens.
        return _FakeRerankResponse(
            [{"index": i, "relevance_score": float(len(documents) - i)} for i in range(len(documents))]
        )

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(handler))

    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api")
    results = [_result("a", 1, text="doc a"), _result("b", 2, text="doc b"), _result("c", 3, text="doc c")]

    reranked = reranker.rerank("q", results, top_k=2)

    assert [r.node_id for r in reranked] == ["a", "b"]
    assert reranked[0].rank == 1
    assert reranked[0].metadata["reranker"] == "cohere/rerank-english-v3.0"


def test_cross_encoder_api_backend_strict_raises_on_call_failure(monkeypatch) -> None:
    def handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        raise RuntimeError("network down")

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(handler))
    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api", strict=True)

    with pytest.raises(RuntimeError, match="API cross-encoder"):
        reranker.rerank("q", [_result("a", 1)], top_k=1)


def test_cross_encoder_api_backend_falls_back_when_not_strict(monkeypatch) -> None:
    def handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        raise RuntimeError("network down")

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(handler))
    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api", strict=False)

    results = [
        _result("a", 1, score=0.5, text="hoàn toàn không liên quan"),
        _result("b", 2, score=0.5, text="điều kiện xét tuyển trí tuệ nhân tạo"),
    ]
    reranked = reranker.rerank("điều kiện xét tuyển trí tuệ nhân tạo", results, top_k=2)

    assert reranked[0].node_id == "b"
    assert reranked[0].metadata["reranker"] == "lexical_overlap_fallback"


def test_create_rerank_tracks_cost_ledger(monkeypatch) -> None:
    def handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        return _FakeRerankResponse([{"index": 0, "relevance_score": 1.0}])

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(handler))
    monkeypatch.setenv("LLM_RERANK_COST_PER_CALL", "0.002")

    with capture_provider_usage() as ledger:
        scored = LLMClient().create_rerank(model="cohere/rerank-english-v3.0", query="q", documents=["d"], top_n=1)

    assert scored == [(0, 1.0)]
    usage = ledger.to_dict()
    assert usage["rerank_calls"] == 1
    assert usage["rerank_cost"] == pytest.approx(0.002)
    assert usage["cost_status"] == "estimated"


def test_effective_reranker_name_reflects_per_call_api_fallback_not_static_available(monkeypatch) -> None:
    """Regression: reranker.available is static (True for the whole lifetime of a
    backend="api" instance), so it cannot distinguish "this call used the real API"
    from "this call silently fell back to lexical overlap" — a technique reading
    `reranker.available` reports the API model as effective even when this specific
    call actually fell back. effective_reranker_name() must read the outcome from
    the call's own returned/tagged results instead.
    """

    def failing_handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        raise RuntimeError("network down")

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(failing_handler))
    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api", strict=False)
    assert reranker.available is True  # static flag: nothing has failed yet, from its own point of view

    reranked = reranker.rerank("q", [_result("a", 1)], top_k=1)

    assert effective_reranker_name(reranker, reranked) == "lexical_overlap_fallback"


def test_effective_reranker_name_reports_real_model_on_api_success(monkeypatch) -> None:
    def handler(model: str, query: str, documents: list[str], top_n: int) -> _FakeRerankResponse:
        return _FakeRerankResponse([{"index": i, "relevance_score": 1.0} for i in range(len(documents))])

    monkeypatch.setattr("raglab.providers.llm_client._litellm", lambda: _fake_litellm_rerank(handler))
    reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api")

    reranked = reranker.rerank("q", [_result("a", 1)], top_k=1)

    assert effective_reranker_name(reranker, reranked) == "cohere/rerank-english-v3.0"


def test_effective_reranker_name_handles_empty_candidate_pool() -> None:
    # No candidates to rerank at all — nothing succeeded or fell back; fall back to
    # the static `available` flag, which is the best available signal in that case.
    local_unavailable = CrossEncoderReranker(strict=False)
    object.__setattr__(local_unavailable, "_encoder", None)
    assert effective_reranker_name(local_unavailable, []) == "lexical_overlap_fallback"

    api_reranker = CrossEncoderReranker(model="cohere/rerank-english-v3.0", backend="api")
    assert effective_reranker_name(api_reranker, []) == "cohere/rerank-english-v3.0"


def test_bm25_hybrid_rerank_reports_lexical_fallback_when_api_reranker_fails(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end regression for the same bug through a real pipeline: with
    reranker_backend="api" and the rerank API failing, allow_fallback=True must make
    query() report components.effective_reranker == "lexical_overlap_fallback", not
    the requested API model id (which is what `reranker.available`-based logic used
    to report, since it never turns False for an api-backed reranker)."""

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    def failing_rerank(model: str, query: str, documents: list[str], top_n: int, timeout: float) -> Any:
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "raglab.providers.llm_client._litellm",
        lambda: SimpleNamespace(embedding=fake_embedding, rerank=failing_rerank),
    )
    monkeypatch.setattr("raglab.providers.llm_client.check_provider_ready", lambda model: None)

    artifact = tmp_path / "artifact"
    params = {
        "reranker_backend": "api",
        "reranker_model": "cohere/rerank-english-v3.0",
        "allow_fallback": True,
    }
    ingest_pipeline = load_pipeline("bm25_hybrid_rerank", params=params)
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))

    query_pipeline = load_pipeline("bm25_hybrid_rerank", params=params)
    assert query_pipeline is not None
    query_pipeline.load(str(artifact))

    answer = query_pipeline.query("điều kiện xét tuyển ngành trí tuệ nhân tạo", mode="retrieval_only")

    assert answer.metadata["components"]["requested_reranker"] == "cohere/rerank-english-v3.0"
    assert answer.metadata["components"]["effective_reranker"] == "lexical_overlap_fallback"


def test_doctor_checks_api_reranker_provider_key(monkeypatch) -> None:
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    pipeline = load_pipeline(
        "bm25_hybrid_rerank",
        params={"reranker_backend": "api", "reranker_model": "cohere/rerank-english-v3.0"},
    )
    assert pipeline is not None

    report = diagnose_technique(pipeline, mode="retrieval_only")
    cross_encoder_check = next(c for c in report["checks"] if c["name"] == "cross_encoder")
    assert cross_encoder_check["status"] == "failed"
    assert "COHERE_API_KEY" in cross_encoder_check["detail"]

    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    report = diagnose_technique(pipeline, mode="retrieval_only")
    cross_encoder_check = next(c for c in report["checks"] if c["name"] == "cross_encoder")
    assert cross_encoder_check["status"] == "ok"
