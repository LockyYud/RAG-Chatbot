from __future__ import annotations

from ragbench.core.schema import Chunk
from ragbench.processing.enrichers.contextual import ContextualEnricher


def _chunk(chunk_id: str, doc_id: str, text: str, **metadata: object) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, metadata=dict(metadata))


def test_prepends_context_to_embedding_text_only() -> None:
    chunks = [_chunk("c1", "d1", "Doanh thu tăng 3% trong quý.", parent_text="Báo cáo tài chính ACME Q2 2023.")]
    documents = {"d1": "Báo cáo tài chính ACME Q2 2023. Doanh thu tăng 3% trong quý."}

    enricher = ContextualEnricher(
        documents=documents,
        context_fn=lambda doc, chunk: "Trích từ báo cáo ACME Q2 2023.",
    )
    nodes = enricher.enrich(chunks)

    node = nodes[0]
    # Context prepended for retrieval...
    assert node.text_for_embedding.startswith("Trích từ báo cáo ACME Q2 2023.")
    assert "Doanh thu tăng 3% trong quý." in node.text_for_embedding
    # ...but generation text stays the original (parent) text — never the synthetic context.
    assert node.text_for_generation == "Báo cáo tài chính ACME Q2 2023."
    assert node.metadata["contextualized"] is True
    assert node.metadata["contextual_prefix"] == "Trích từ báo cáo ACME Q2 2023."


def test_failed_context_call_degrades_to_plain_chunk() -> None:
    def boom(doc: str, chunk: str) -> str:
        raise RuntimeError("LLM unavailable")

    chunks = [_chunk("c1", "d1", "nội dung gốc")]
    nodes = ContextualEnricher(documents={"d1": "tài liệu"}, context_fn=boom).enrich(chunks)

    assert nodes[0].text_for_embedding == "nội dung gốc"
    assert nodes[0].metadata["contextualized"] is False
    assert "contextual_prefix" not in nodes[0].metadata


def test_document_truncation_caps_tokens_sent_to_llm() -> None:
    seen: list[str] = []

    def capture(doc: str, chunk: str) -> str:
        seen.append(doc)
        return "ctx"

    long_doc = " ".join(f"w{i}" for i in range(5000))
    chunks = [_chunk("c1", "d1", "chunk")]
    ContextualEnricher(
        documents={"d1": long_doc},
        context_fn=capture,
        max_doc_tokens=100,
    ).enrich(chunks)

    assert len(seen) == 1
    assert len(seen[0].split()) == 100


def test_missing_document_falls_back_to_chunk_text() -> None:
    captured: list[str] = []
    chunks = [_chunk("c1", "d-unknown", "chỉ có chunk")]

    def capture_document(doc: str, chunk: str) -> str:
        captured.append(doc)
        return "ctx"

    ContextualEnricher(
        documents={},  # no document for this doc_id
        context_fn=capture_document,
    ).enrich(chunks)

    # When the document map lacks the doc_id, the chunk text itself is used as document.
    assert captured == ["chỉ có chunk"]
