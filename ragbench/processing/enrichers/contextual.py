"""Contextual Retrieval enrichment (Anthropic, 2024).

The problem
-----------
Chunking destroys context.  A 200-token chunk that reads "The revenue grew 3%
that quarter" is useless to a retriever — *which* company, *which* quarter?  The
surrounding document knew, but the chunk no longer does.

The fix
-------
Before indexing, ask an LLM to write a short (50–100 token) snippet that situates
each chunk inside its full document, then **prepend** that snippet to the chunk's
indexing text.  Because both the dense embedder and BM25 in this repo read
``IndexedNode.text_for_embedding``, prepending there gives us *both* of
Anthropic's variants at once: "Contextual Embeddings" and "Contextual BM25".

Anthropic reports the contextualized hybrid index cuts top-20 retrieval failures
by ~49% (and ~67% once a reranker is added) versus naive chunking.
See: https://www.anthropic.com/news/contextual-retrieval

Design notes
------------
- ``text_for_generation`` is left as the *original* chunk/parent text — the
  synthetic context is a retrieval aid, not something the answer should quote.
- The LLM call is injectable (``context_fn``) so the enricher can be unit-tested
  with no network, and so a pipeline can swap providers.
- One LLM call per chunk.  The full document is truncated to ``max_doc_tokens``
  to bound cost; a per-chunk failure degrades to the plain chunk rather than
  aborting the whole ingest.
"""

from __future__ import annotations

from collections.abc import Callable

from ragbench.core.interfaces import BaseEnricher
from ragbench.core.schema import Chunk, IndexedNode
from ragbench.core.text import tokenize

ContextFn = Callable[[str, str], str]
"""Signature: ``(document_text, chunk_text) -> situating_context``."""

PROMPT_TEMPLATE = (
    "<document>\n{document}\n</document>\n\n"
    "Here is the chunk we want to situate within the whole document:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Give a short, succinct context (1-2 sentences) to situate this chunk within "
    "the overall document, for the purpose of improving search retrieval of the "
    "chunk. Answer ONLY with the succinct context and nothing else. "
    "Answer in the same language as the chunk."
)


class ContextualEnricher(BaseEnricher):
    """Prepend an LLM-generated, document-aware context to each chunk's index text.

    Parameters
    ----------
    documents:
        Map of ``doc_id -> full document text``.  Built by the pipeline from the
        parsed blocks before chunking is enriched.
    context_fn:
        Callable that produces the situating context for one ``(document, chunk)``
        pair.  Defaults to an LLM call via :class:`LLMClient`.  Inject a stub in
        tests to run offline.
    context_model:
        Chat model id used when ``context_fn`` is not supplied.
    max_doc_tokens:
        Truncate each document to this many tokens before sending it to the LLM
        (cost guard for long documents).
    """

    def __init__(
        self,
        documents: dict[str, str],
        context_fn: ContextFn | None = None,
        context_model: str = "gpt-4.1-mini",
        max_doc_tokens: int = 4000,
        context_max_tokens: int = 160,
        **_: object,
    ) -> None:
        self.documents = documents
        self.context_model = context_model
        self.max_doc_tokens = max_doc_tokens
        self.context_max_tokens = context_max_tokens
        self._context_fn = context_fn or self._default_context_fn

    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        nodes: list[IndexedNode] = []
        for chunk in chunks:
            document = self.documents.get(chunk.doc_id, chunk.text)
            document = self._truncate(document, self.max_doc_tokens)
            context = self._safe_context(document, chunk.text)

            embedding_text = f"{context}\n\n{chunk.text}" if context else chunk.text
            generation_text = chunk.metadata.get("parent_text", chunk.text)

            metadata = dict(chunk.metadata)
            metadata["contextualized"] = bool(context)
            if context:
                metadata["contextual_prefix"] = context

            nodes.append(
                IndexedNode(
                    node_id=chunk.chunk_id,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text_for_embedding=embedding_text,
                    text_for_generation=generation_text,
                    metadata=metadata,
                )
            )
        return nodes

    # ── internals ──────────────────────────────────────────────────────────

    def _safe_context(self, document: str, chunk_text: str) -> str:
        try:
            return self._context_fn(document, chunk_text).strip()
        except Exception:
            # A single failed context call must not abort the whole ingest.
            return ""

    def _default_context_fn(self, document: str, chunk_text: str) -> str:
        from ragbench.providers.llm_client import LLMClient

        completion = LLMClient().create_chat_completion(
            model=self.context_model,
            temperature=0.0,
            max_tokens=self.context_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(document=document, chunk=chunk_text),
                }
            ],
        )
        return completion.text

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        tokens = tokenize(text)
        if len(tokens) <= max_tokens:
            return text
        # Token-count is whitespace-based here; rejoin a safe prefix of the raw text.
        return " ".join(text.split()[:max_tokens])
