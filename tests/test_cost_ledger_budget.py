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
