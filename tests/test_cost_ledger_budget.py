from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import evaluation.runner as runner_module
from evaluation.runner import BudgetExceededError, run_eval
from raglab.core.base import load_pipeline
from raglab.core.schema import RAGAnswer
from raglab.providers.llm_client import LLMClient, ProviderUsageLedger, capture_provider_usage


class _FakeCompletionResponse:
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]


def _fake_litellm() -> Any:
    return SimpleNamespace(
        embedding=lambda model, input, timeout: _FakeEmbeddingResponse([[0.1, 0.2] for _ in input]),
        completion=lambda model, messages, temperature, max_tokens, timeout: _FakeCompletionResponse(
            "hello", prompt_tokens=10, completion_tokens=5
        ),
    )


def test_ledger_requires_pricing_for_every_call_type_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """One priced call type must not mask an unpriced one as though the whole run's cost were known.

    This is the exact bug that was previously reported: an embedding call with
    configured pricing OR'd into a single global flag, so a run that also made
    an unpriced chat call was still reported as "estimated" (with the chat
    cost silently counted as $0).
    """
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "0.0001")
    # Set to empty rather than delenv: load_dotenv() uses os.environ.setdefault,
    # so deleting the var would let the project's real .env silently refill it.
    monkeypatch.setenv("LLM_CHAT_INPUT_COST_PER_1K", "")
    monkeypatch.setenv("LLM_CHAT_OUTPUT_COST_PER_1K", "")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_embeddings("fake-embed", ["a", "b"])
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])

    usage = ledger.to_dict()
    assert usage["embedding_calls"] == 1
    assert usage["chat_calls"] == 1
    assert usage["embedding_cost"] > 0
    assert usage["chat_cost"] == 0.0
    assert usage["cost_status"] == "unknown"


def test_ledger_reports_estimated_once_every_used_call_type_is_priced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "0.0001")
    monkeypatch.setenv("LLM_CHAT_INPUT_COST_PER_1K", "0.001")
    monkeypatch.setenv("LLM_CHAT_OUTPUT_COST_PER_1K", "0.002")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_embeddings("fake-embed", ["a"])
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])

    usage = ledger.to_dict()
    assert usage["embedding_cost"] > 0
    assert usage["chat_cost"] > 0
    assert usage["cost_status"] == "estimated"
    assert usage["estimated_cost"] == round(usage["embedding_cost"] + usage["chat_cost"], 8)


def test_chat_pricing_requires_both_input_and_output_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only one of the two chat rates configured must not count as "priced" — a
    real chat call always spends both prompt and completion tokens."""
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "")
    monkeypatch.setenv("LLM_CHAT_INPUT_COST_PER_1K", "0.001")
    monkeypatch.setenv("LLM_CHAT_OUTPUT_COST_PER_1K", "")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])

    assert ledger.to_dict()["cost_status"] == "unknown"


def test_ledger_with_no_calls_is_trivially_estimated() -> None:
    assert ProviderUsageLedger().to_dict()["cost_status"] == "estimated"


def test_explicit_zero_rate_counts_as_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local/free model validly costs $0 — that must read as "estimated", not "unknown"."""
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "0")
    monkeypatch.setenv("LLM_CHAT_INPUT_COST_PER_1K", "0")
    monkeypatch.setenv("LLM_CHAT_OUTPUT_COST_PER_1K", "0")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_embeddings("fake-embed", ["a"])
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])

    usage = ledger.to_dict()
    assert usage["embedding_cost"] == 0.0
    assert usage["chat_cost"] == 0.0
    assert usage["cost_status"] == "estimated"


def test_missing_rate_stays_unknown_even_when_result_would_be_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinguish "explicitly $0" from "never configured" even though both yield a $0 total."""
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_embeddings("fake-embed", ["a"])

    usage = ledger.to_dict()
    assert usage["embedding_cost"] == 0.0
    assert usage["cost_status"] == "unknown"


def test_negative_rate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("raglab.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "-0.001")
    client = LLMClient()

    with pytest.raises(RuntimeError, match="must not be negative"):
        with capture_provider_usage():
            client.create_embeddings("fake-embed", ["a"])


class _FakeRerankResponse:
    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results


def test_rerank_cost_appears_in_report_cost_estimate_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: ProviderUsageLedger.to_dict()["estimated_cost"] already included
    rerank_cost (so the budget guard was never wrong), but evaluation/runner.py's
    per-prediction cost_estimate/evaluation_cost_estimate and the report-level
    cost_summary only ever copied embedding_cost/chat_cost — silently dropping
    rerank spend from the report even though the total was correct. This made "how
    much did the API cross-encoder cost" unanswerable from the report alone.
    """

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    def fake_rerank(model: str, query: str, documents: list[str], top_n: int, timeout: float) -> Any:
        return _FakeRerankResponse([{"index": i, "relevance_score": 1.0} for i in range(len(documents))])

    monkeypatch.setattr(
        "raglab.providers.llm_client._litellm",
        lambda: SimpleNamespace(embedding=fake_embedding, rerank=fake_rerank),
    )
    monkeypatch.setattr("raglab.providers.llm_client.check_provider_ready", lambda model: None)
    monkeypatch.setenv("LLM_RERANK_COST_PER_CALL", "0.01")
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "0")

    artifact = tmp_path / "artifact"
    params = {"reranker_backend": "api", "reranker_model": "cohere/rerank-english-v3.0"}
    ingest_pipeline = load_pipeline("bm25_hybrid_rerank", params=params)
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))

    query_pipeline = load_pipeline("bm25_hybrid_rerank", params=params)
    assert query_pipeline is not None
    output_path = tmp_path / "eval.json"

    report = run_eval(
        query_pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(output_path),
        mode="retrieval_only",
    )

    for prediction in report["predictions"]:
        assert prediction["metadata"]["cost_estimate"]["rerank_cost"] == pytest.approx(0.01)
    query_count = len(report["predictions"])
    assert report["cost_summary"]["pipeline_cost"]["rerank_cost_total"] == pytest.approx(0.01 * query_count)
    assert report["cost_summary"]["pipeline_cost"]["total"] == pytest.approx(
        report["cost_summary"]["pipeline_cost"]["rerank_cost_total"]
        + report["cost_summary"]["pipeline_cost"]["embedding_cost_total"]
        + report["cost_summary"]["pipeline_cost"]["chat_cost_total"]
    )


def test_budget_guard_stops_run_and_preserves_completed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any
    ) -> RAGAnswer:
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "cost_estimate": {
                    "currency": "USD",
                    "amount": 1.0,
                    "embedding_cost": 0.0,
                    "chat_cost": 1.0,
                    "status": "estimated",
                },
                "components": {},
            },
        )

    monkeypatch.setattr(runner_module, "_run_single_query", fake_run_single_query)
    output_path = tmp_path / "eval.json"

    with pytest.raises(BudgetExceededError, match=r"exceeded max_estimated_cost_usd"):
        run_eval(
            pipeline,
            str(artifact),
            "datasets/sample/qa.jsonl",
            str(output_path),
            max_estimated_cost_usd=1.5,
        )

    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    assert checkpoint_path.exists()
    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    prediction_lines = [line for line in lines if '"type": "prediction"' in line or '"type":"prediction"' in line]
    assert len(prediction_lines) == 2  # stopped after the 2nd query (cumulative cost 2.0 > 1.5)
    assert not output_path.exists()  # the run aborted before a final report was ever written


def test_budget_guard_is_noop_when_cost_status_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A priced embedding call alongside an unpriced chat call yields a positive but
    incomplete total — the guard must not abort a run on an incomplete number, no
    matter how far past the cap that partial total looks."""
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any
    ) -> RAGAnswer:
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "cost_estimate": {
                    "currency": "USD",
                    "amount": 100.0,  # would obviously blow any reasonable cap...
                    "embedding_cost": 100.0,
                    "chat_cost": 0.0,
                    "status": "unknown",  # ...but chat pricing was never configured, so it's not trustworthy
                },
                "components": {},
            },
        )

    monkeypatch.setattr(runner_module, "_run_single_query", fake_run_single_query)
    output_path = tmp_path / "eval.json"

    report = run_eval(
        pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(output_path),
        max_estimated_cost_usd=1.5,
    )

    assert output_path.exists()
    assert len(report["predictions"]) == 3  # all 3 queries ran; the guard never fired
