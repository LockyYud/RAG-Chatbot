from __future__ import annotations

from raglab.core.schema import IndexedNode
from raglab.providers.llm_client import LLMClient, default_embed_model


class Embedder:
    """Embed texts using any provider supported by litellm.

    The model is taken from the ``EMBED_MODEL`` env var by default
    (``text-embedding-3-small`` if unset).  Pass *model* explicitly to
    override per-pipeline.

    Example::

        embedder = Embedder()                          # uses EMBED_MODEL env
        embedder = Embedder(model="ollama/nomic-embed-text")  # local
    """

    def __init__(
        self,
        model: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = model or default_embed_model()
        self.batch_size = batch_size
        self._client = LLMClient()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._client.create_embeddings(self.model, texts, self.batch_size)

    def embed_nodes(self, nodes: list[IndexedNode]) -> list[IndexedNode]:
        vectors = self.embed_texts([node.text_for_embedding for node in nodes])
        for node, vector in zip(nodes, vectors, strict=True):
            node.embedding = vector
            node.metadata["embedding_model"] = self.model
        return nodes
