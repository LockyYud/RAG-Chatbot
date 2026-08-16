from __future__ import annotations

import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

import raglab.inference.context_builders.citation_context as citation_context_module
from raglab.core.base import load_pipeline

QUESTION_A = "Điều kiện xét tuyển ngành trí tuệ nhân tạo là gì?"
QUESTION_B = "Hồ sơ đăng ký xét tuyển gồm những giấy tờ nào?"


class _FakeEmbeddingResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]


def _fake_embedding(model: str, input: list[str], timeout: float) -> _FakeEmbeddingResponse:
    # Any consistent-dimension, content-derived vector — correctness of the
    # retrieved ranking isn't what this test is checking.
    vectors = [[float((hash(text) >> shift) % 97) / 97 for shift in range(0, 64, 8)] for text in input]
    return _FakeEmbeddingResponse(vectors)


def _extract_question(messages: list[dict[str, str]]) -> str:
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    match = re.search(r"(?:Question|Original query): (.*)", user_content)
    assert match is not None
    return match.group(1).split("\n")[0]


def _make_marker_completion(text_builder: Any):
    """Every call returns instantly — no artificial delay here. The
    interleaving in these tests is forced deterministically via a blocked
    CitationContextBuilder.build_context(), not by racing real wall-clock
    timing against the GIL's scheduling (which is unreliable: the actual
    vulnerable window — after a retriever's internal write, before the
    pipeline reads it back — is short, in-memory, pure-Python work with no
    blocking call inside it, so relying on incidental thread-switch timing to
    land inside that window is flaky at best)."""

    def completion(model: str, messages: list[dict[str, str]], temperature: float, max_tokens: int, timeout: float):
        question = _extract_question(messages)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text_builder(question)))],
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )

    return completion


def _run_ingest(technique_id: str, params: dict[str, Any], artifact: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    # ingest() never calls create_chat_completion for these techniques — only embed_nodes.
    monkeypatch.setattr(
        "raglab.providers.llm_client._litellm",
        lambda: SimpleNamespace(embedding=_fake_embedding, completion=None),
    )
    pipeline = load_pipeline(technique_id, params=params)
    assert pipeline is not None
    pipeline.ingest("datasets/sample/docs", str(artifact))
    query_pipeline = load_pipeline(technique_id, params=params)
    assert query_pipeline is not None
    query_pipeline.load(str(artifact))
    return query_pipeline


def _run_forced_interleave(pipeline: Any) -> tuple[Any, Any]:
    """Run query(question_a) and query(question_b) on the SAME shared, already-
    loaded pipeline instance, deterministically forcing question_b's entire
    query (including whatever it writes to any shared retriever state) to run
    to completion *during* question_a's window between "retrieval finished"
    and "the pipeline reads retrieval's runtime metadata back" — the exact
    window a self.last_metadata-style side channel would be vulnerable in.
    """
    b_finished = threading.Event()
    a_reached_context_build = threading.Event()
    original_build_context = citation_context_module.CitationContextBuilder.build_context

    def patched_build_context(self: Any, query: str, results: Any) -> Any:
        if query == QUESTION_A:
            a_reached_context_build.set()
            assert b_finished.wait(timeout=5), "question_b never completed — test setup is broken"
        return original_build_context(self, query, results)

    answers: dict[str, Any] = {}
    lock = threading.Lock()

    def run_a() -> None:
        answer = pipeline.query(QUESTION_A, mode="retrieval_only")
        with lock:
            answers["a"] = answer

    def run_b() -> None:
        assert a_reached_context_build.wait(timeout=5), "question_a never reached the blocking point"
        answer = pipeline.query(QUESTION_B, mode="retrieval_only")
        with lock:
            answers["b"] = answer
        b_finished.set()

    with mock.patch.object(citation_context_module.CitationContextBuilder, "build_context", patched_build_context):
        thread_a = threading.Thread(target=run_a)
        thread_b = threading.Thread(target=run_b)
        thread_a.start()
        thread_b.start()
        thread_a.join(timeout=10)
        thread_b.join(timeout=10)

    assert "a" in answers and "b" in answers, "one of the concurrent queries never completed"
    return answers["a"], answers["b"]


@pytest.fixture(autouse=True)
def _no_provider_checks():
    with mock.patch("raglab.providers.llm_client.check_provider_ready", lambda model: None):
        yield


def test_hyde_concurrent_queries_do_not_cross_contaminate_retrieval_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression test: HyDERetriever used to write self.last_metadata inside
    retrieve() and have the pipeline read it back afterward. Two concurrent
    query() calls on the same shared, already-loaded pipeline (raglab
    eval/bench --concurrency) could interleave in that window, silently
    attaching one question's runtime metadata to a different question's answer."""
    artifact = tmp_path / "artifact"
    query_pipeline = _run_ingest("hyde_2022", {"hyde_samples": 1}, artifact, monkeypatch)

    monkeypatch.setattr(
        "raglab.providers.llm_client._litellm",
        lambda: SimpleNamespace(
            embedding=_fake_embedding,
            completion=_make_marker_completion(lambda q: f"HYPO_DOC_MARKER::{q}"),
        ),
    )

    answer_a, answer_b = _run_forced_interleave(query_pipeline)

    assert answer_a.metadata["retrieval_runtime"]["generated_texts"] == [f"HYPO_DOC_MARKER::{QUESTION_A}"]
    assert answer_b.metadata["retrieval_runtime"]["generated_texts"] == [f"HYPO_DOC_MARKER::{QUESTION_B}"]


def test_rag_fusion_concurrent_queries_do_not_cross_contaminate_retrieval_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same regression as HyDE, for RAGFusionRetriever's self.last_metadata."""
    artifact = tmp_path / "artifact"
    query_pipeline = _run_ingest("rag_fusion_2024", {"fusion_queries": 1}, artifact, monkeypatch)

    monkeypatch.setattr(
        "raglab.providers.llm_client._litellm",
        lambda: SimpleNamespace(
            embedding=_fake_embedding,
            completion=_make_marker_completion(lambda q: f"ALT_QUERY_MARKER::{q}"),
        ),
    )

    answer_a, answer_b = _run_forced_interleave(query_pipeline)

    for question, answer in ((QUESTION_A, answer_a), (QUESTION_B, answer_b)):
        all_queries = answer.metadata["retrieval_runtime"]["queries"]
        assert all_queries[0] == question
        for alternative in all_queries[1:]:
            assert alternative == f"ALT_QUERY_MARKER::{question}", (
                f"question {question!r} got someone else's alternative query: {alternative!r}"
            )
