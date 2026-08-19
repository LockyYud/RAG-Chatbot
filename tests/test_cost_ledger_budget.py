from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ragbench.evaluation.runner as runner_module
from ragbench.core.base import load_pipeline
from ragbench.core.schema import RAGAnswer
from ragbench.evaluation.runner import BudgetExceededError, run_eval
from ragbench.providers.llm_client import LLMClient, ProviderUsageLedger, capture_provider_usage, generation_seed


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
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
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
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
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
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
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
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
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
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "")
    client = LLMClient()

    with capture_provider_usage() as ledger:
        client.create_embeddings("fake-embed", ["a"])

    usage = ledger.to_dict()
    assert usage["embedding_cost"] == 0.0
    assert usage["cost_status"] == "unknown"


def test_generation_seed_is_forwarded_to_the_provider_only_when_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: trial_seed used to change nothing about LLM generation at
    all (see benchmarks.runner's removed global random.seed() call) — the
    only thing "seeded" was retry jitter, by accident, via mutating global
    random state. generation_seed() is the real mechanism: it must reach
    litellm.completion() as `seed=`, and must NOT be present when inactive
    (so a provider that errors on an unexpected kwarg is unaffected by
    default)."""
    received_kwargs: list[dict] = []

    def fake_completion(**kwargs: Any) -> Any:
        received_kwargs.append(kwargs)
        return _FakeCompletionResponse("hello", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(completion=fake_completion)
    )
    client = LLMClient()

    client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])
    assert "seed" not in received_kwargs[-1]

    with generation_seed(42):
        result = client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])
    assert received_kwargs[-1]["seed"] == 42
    assert result.generation_seed_requested == 42

    # Scoped: outside the block, the next call is unseeded again.
    client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])
    assert "seed" not in received_kwargs[-1]


def test_generation_seed_requested_does_not_claim_determinism_was_achieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A changed system_fingerprint across same-seed calls is the signal that
    the provider's serving snapshot moved — generation_seed_requested alone
    must never be read as "the output was reproduced"."""

    def fake_completion(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            system_fingerprint="fp_v1",
        )

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(completion=fake_completion)
    )
    client = LLMClient()

    with capture_provider_usage() as ledger, generation_seed(7):
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])
        client.create_chat_completion("fake-chat", [{"role": "user", "content": "hi"}])

    assert ledger.to_dict()["system_fingerprints"] == ["fp_v1"]


def test_negative_rate_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragbench.providers.llm_client._litellm", _fake_litellm)
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
        "ragbench.providers.llm_client._litellm",
        lambda: SimpleNamespace(embedding=fake_embedding, rerank=fake_rerank),
    )
    monkeypatch.setattr("ragbench.providers.llm_client.check_provider_ready", lambda model: None)
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
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
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


def test_budget_guard_counts_measurement_cost_not_just_pipeline_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the guard used to sum only pipeline_cost_total + judge_cost_total.

    latency_repetitions > 1 makes extra real API calls per query (see
    _run_single_query's measurement_cost_estimate) whose cost was previously
    dropped entirely — a run capped at max_estimated_cost_usd could spend
    well past that cap on repeated-measurement calls alone. total_spend
    (pipeline + measurement + warmup + judge) is what the guard must use.
    """
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
    ) -> RAGAnswer:
        return RAGAnswer(
            query=item.question,
            answer="stub",
            contexts=[],
            metadata={
                "provider_usage": {"retries": 0},
                "cost_estimate": {"currency": "USD", "amount": 0.1, "status": "estimated"},
                # Each query's extra latency_repetitions calls cost as much as
                # the scored request itself — summing cost_estimate alone
                # would report half the true spend.
                "measurement_cost_estimate": {"currency": "USD", "amount": 0.1, "status": "estimated"},
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
            max_estimated_cost_usd=0.15,  # below even the 2nd query's combined 0.2/query total
        )

    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    prediction_lines = [line for line in lines if '"type": "prediction"' in line or '"type":"prediction"' in line]
    assert len(prediction_lines) == 1  # tripped after the 1st query: 0.1 + 0.1 = 0.2 > 0.15


def test_cost_summary_reports_total_spend_across_all_four_buckets() -> None:
    """total_spend must equal technique + measurement + warmup + judge cost —
    not just technique cost (the old ``total_estimated_cost`` alias, kept for
    backward compatibility but no longer what the budget guard enforces)."""
    predictions = [
        RAGAnswer(
            query="q",
            answer="a",
            contexts=[],
            metadata={
                "cost_estimate": {"amount": 1.0, "embedding_cost": 0.4, "chat_cost": 0.6, "rerank_cost": 0.0},
                "measurement_cost_estimate": {"amount": 0.5, "status": "estimated"},
                "evaluation_cost_estimate": {
                    "amount": 2.0,
                    "embedding_cost": 0.0,
                    "chat_cost": 2.0,
                    "rerank_cost": 0.0,
                },
            },
        )
    ]
    summary = runner_module._cost_summary(predictions, warmup_cost_total=0.3)
    assert summary["pipeline_cost"]["total"] == pytest.approx(1.0)
    assert summary["measurement_cost"]["total"] == pytest.approx(0.5)
    assert summary["warmup_cost"]["total"] == pytest.approx(0.3)
    assert summary["judge_cost"]["total"] == pytest.approx(2.0)
    assert summary["total_spend"] == pytest.approx(1.0 + 0.5 + 0.3 + 2.0)
    # Back-compat alias keeps its pre-existing "technique cost only" meaning.
    assert summary["total_estimated_cost"] == pytest.approx(1.0)


def test_budget_guard_is_noop_when_cost_status_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A priced embedding call alongside an unpriced chat call yields a positive but
    incomplete total — the guard must not abort a run on an incomplete number, no
    matter how far past the cap that partial total looks."""
    artifact = tmp_path / "artifact"
    pipeline = load_pipeline("parent_child")
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))

    def fake_run_single_query(
        pipeline: Any, item: Any, *, mode: str, latency_repetitions: int, judge: Any, seed: Any = None
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


def test_warmup_queries_spend_is_counted_in_cost_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: warm-up queries make real embedding/chat calls (see run_eval's
    warm-up loop) but that spend used to be captured and immediately discarded —
    excluded from every cost total, including the budget guard. It must now show
    up in cost_summary.warmup_cost and count toward total_spend."""

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(embedding=fake_embedding)
    )
    monkeypatch.setattr("ragbench.providers.llm_client.check_provider_ready", lambda model: None)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "1.0")

    artifact = tmp_path / "artifact"
    ingest_pipeline = load_pipeline("naive_rag")
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))

    query_pipeline = load_pipeline("naive_rag")
    assert query_pipeline is not None
    output_path = tmp_path / "eval.json"

    report = run_eval(
        query_pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(output_path),
        mode="retrieval_only",
        warmup_queries=2,
    )

    assert report["cost_summary"]["warmup_cost"]["total"] > 0
    assert report["cost_summary"]["total_spend"] == pytest.approx(
        report["cost_summary"]["pipeline_cost"]["total"]
        + report["cost_summary"]["measurement_cost"]["total"]
        + report["cost_summary"]["warmup_cost"]["total"]
        + report["cost_summary"]["judge_cost"]["total"]
    )
    # total_spend must exceed the technique-only alias once warm-ups cost anything.
    assert report["cost_summary"]["total_spend"] > report["cost_summary"]["total_estimated_cost"]


def test_budget_guard_checks_after_every_warmup_call_not_only_after_all_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: warm-up calls used to run to completion — all of them —
    before the budget guard got a single chance to check anything (the guard
    only lived inside record(), called for real predictions). A cap smaller
    than even one warm-up call's cost must stop after the first one, not
    after every configured warm-up call has already spent money."""
    embedding_calls: list[int] = []

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        embedding_calls.append(1)
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(embedding=fake_embedding)
    )
    monkeypatch.setattr("ragbench.providers.llm_client.check_provider_ready", lambda model: None)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "1000000")  # deliberately huge per-call cost

    artifact = tmp_path / "artifact"
    ingest_pipeline = load_pipeline("naive_rag")
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))
    embedding_calls.clear()  # ingest embeds the corpus too; only count query-time calls below

    query_pipeline = load_pipeline("naive_rag")
    assert query_pipeline is not None
    output_path = tmp_path / "eval.json"

    with pytest.raises(BudgetExceededError, match=r"warm-up spend"):
        run_eval(
            query_pipeline,
            str(artifact),
            "datasets/sample/qa.jsonl",
            str(output_path),
            mode="retrieval_only",
            warmup_queries=5,
            max_estimated_cost_usd=0.01,
        )

    assert len(embedding_calls) == 1  # tripped after the 1st warm-up call, not all 5
    assert not output_path.exists()  # aborted before a final report was ever written


def test_resuming_an_attempt_that_already_tripped_the_cap_during_warmup_aborts_before_any_new_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a resumed run seeded warmup_cost_total correctly from the
    checkpoint but only ever re-checked the budget from *inside* the
    warm-up loop body — an index already in resumed_warmup is skipped with
    a bare `continue` and no check at all. If every configured warm-up call
    was already checkpointed (as here: warmup_queries=1, and that one
    attempt is exactly what tripped the cap), resume sailed straight past
    warm-up with zero budget checks and ran a real prediction query before
    the next guard (inside record()) ever caught it — spending past a cap
    a prior attempt had already exceeded."""
    embedding_calls: list[int] = []

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        embedding_calls.append(1)
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(embedding=fake_embedding)
    )
    monkeypatch.setattr("ragbench.providers.llm_client.check_provider_ready", lambda model: None)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "1000000")  # deliberately huge per-call cost

    artifact = tmp_path / "artifact"
    ingest_pipeline = load_pipeline("naive_rag")
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))
    embedding_calls.clear()  # ingest embeds the corpus too; only count query-time calls below
    output_path = tmp_path / "eval.json"

    first_pipeline = load_pipeline("naive_rag")
    assert first_pipeline is not None
    with pytest.raises(BudgetExceededError, match=r"warm-up spend"):
        run_eval(
            first_pipeline,
            str(artifact),
            "datasets/sample/qa.jsonl",
            str(output_path),
            mode="retrieval_only",
            warmup_queries=1,
            max_estimated_cost_usd=0.01,
        )
    assert len(embedding_calls) == 1  # the 1st (only) warm-up call ran and tripped the cap

    with pytest.raises(BudgetExceededError, match=r"recovered from a prior checkpointed attempt"):
        run_eval(
            first_pipeline,
            str(artifact),
            "datasets/sample/qa.jsonl",
            str(output_path),
            mode="retrieval_only",
            warmup_queries=1,
            max_estimated_cost_usd=0.01,
        )
    # No new API call at all: the already-checkpointed warm-up call is
    # skipped (it's the only configured one), and the resumed run must abort
    # before reaching the first real prediction query.
    assert len(embedding_calls) == 1


def test_warmup_calls_are_checkpointed_and_not_repaid_for_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: warm-up usage was never checkpointed, so a resume after a
    crash mid-warmup silently dropped the crashed attempt's warm-up spend
    from total_spend AND re-ran (re-paid for) every warm-up call from
    scratch. Each warm-up call must be checkpointed like a prediction —
    resumed calls are skipped, and their cost still counts."""
    embedding_calls: list[int] = []

    def fake_embedding(model: str, input: list[str], timeout: float) -> Any:
        embedding_calls.append(1)
        return SimpleNamespace(data=[{"index": i, "embedding": [0.1, 0.2]} for i in range(len(input))])

    monkeypatch.setattr(
        "ragbench.providers.llm_client._litellm", lambda: SimpleNamespace(embedding=fake_embedding)
    )
    monkeypatch.setattr("ragbench.providers.llm_client.check_provider_ready", lambda model: None)
    monkeypatch.setenv("LLM_EMBEDDING_INPUT_COST_PER_1K", "1.0")
    monkeypatch.setenv("RAGLAB_EMBEDDING_CACHE", "0")  # every call below must reach the fake provider, not a cache hit

    artifact = tmp_path / "artifact"
    ingest_pipeline = load_pipeline("naive_rag")
    assert ingest_pipeline is not None
    ingest_pipeline.ingest("datasets/sample/docs", str(artifact))
    embedding_calls.clear()  # ingest embeds the corpus too; only count query-time calls below

    crashing_pipeline = load_pipeline("naive_rag")
    assert crashing_pipeline is not None
    original_query = crashing_pipeline.query
    warmup_call_count = {"n": 0}

    def flaky_query(question: str, mode: str = "full_rag") -> RAGAnswer:
        warmup_call_count["n"] += 1
        if warmup_call_count["n"] == 2:
            raise RuntimeError("simulated crash during warm-up")
        return original_query(question, mode=mode)

    monkeypatch.setattr(crashing_pipeline, "query", flaky_query)
    output_path = tmp_path / "eval.json"

    with pytest.raises(RuntimeError, match="simulated crash during warm-up"):
        run_eval(
            crashing_pipeline,
            str(artifact),
            "datasets/sample/qa.jsonl",
            str(output_path),
            mode="retrieval_only",
            warmup_queries=2,
        )

    assert len(embedding_calls) == 1  # only the 1st warm-up call actually ran before the crash
    checkpoint_path = Path(f"{output_path}.checkpoint.jsonl")
    checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    warmup_records = [line for line in checkpoint_lines if '"type": "warmup"' in line]
    assert len(warmup_records) == 1  # the completed warm-up call was checkpointed before the crash

    resuming_pipeline = load_pipeline("naive_rag")
    assert resuming_pipeline is not None
    calls_before_resume = len(embedding_calls)
    report = run_eval(
        resuming_pipeline,
        str(artifact),
        "datasets/sample/qa.jsonl",
        str(output_path),
        mode="retrieval_only",
        warmup_queries=2,
    )

    # The already-checkpointed 1st warm-up call must not run again — the
    # resumed run should only make one *new* embedding call for warm-up
    # (index 1) plus one per real prediction that follows (this dataset has
    # 3 questions); if index 0 were re-run, this would be one call too many.
    dataset_size = 3
    new_calls_during_resume = len(embedding_calls) - calls_before_resume
    assert new_calls_during_resume == 1 + dataset_size
    # Both warm-up calls' cost must be in total_spend: the one from the
    # crashed attempt (recovered from the checkpoint) plus the one from this
    # resumed attempt — not just whichever attempt happened to run last.
    # Verified independently of the report by summing the checkpoint's own
    # two "warmup" records, so this isn't circular with how the report
    # itself was computed.
    final_checkpoint_lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    final_warmup_records = [json.loads(line) for line in final_checkpoint_lines if '"type": "warmup"' in line]
    assert len(final_warmup_records) == 2
    expected_warmup_total = sum(record["usage"]["estimated_cost"] for record in final_warmup_records)
    assert expected_warmup_total > 0
    assert report["cost_summary"]["warmup_cost"]["total"] == pytest.approx(expected_warmup_total)
