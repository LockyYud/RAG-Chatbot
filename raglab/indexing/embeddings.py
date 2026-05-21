from __future__ import annotations

from raglab.core.schema import IndexedNode
from raglab.providers.openai_compatible import OpenAICompatibleClient


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        api_key_env: str = "OPENAI_API_KEY",
        base_url_env: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAICompatibleClient(api_key_env=api_key_env, base_url_env=base_url_env, base_url=base_url)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.client.create_embeddings(self.model, texts, self.batch_size)

    def embed_nodes(self, nodes: list[IndexedNode]) -> list[IndexedNode]:
        vectors = self.embed_texts([node.text_for_embedding for node in nodes])
        for node, vector in zip(nodes, vectors, strict=True):
            node.embedding = vector
            node.metadata["embedding_model"] = self.model
        return nodes
