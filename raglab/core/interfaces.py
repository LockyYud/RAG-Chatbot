from __future__ import annotations

from abc import ABC, abstractmethod

from raglab.core.schema import (
    BuiltContext,
    Chunk,
    DocumentBlock,
    IndexedNode,
    RAGAnswer,
    RetrievalResult,
    VerificationReport,
)


class BaseParser(ABC):
    @abstractmethod
    def parse(self, path: str) -> list[DocumentBlock]:
        raise NotImplementedError


class BaseCleaner(ABC):
    @abstractmethod
    def clean(self, blocks: list[DocumentBlock]) -> list[DocumentBlock]:
        raise NotImplementedError


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, blocks: list[DocumentBlock]) -> list[Chunk]:
        raise NotImplementedError


class BaseEnricher(ABC):
    @abstractmethod
    def enrich(self, chunks: list[Chunk]) -> list[IndexedNode]:
        raise NotImplementedError


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int) -> list[RetrievalResult]:
        raise NotImplementedError


class BaseContextBuilder(ABC):
    @abstractmethod
    def build_context(self, query: str, results: list[RetrievalResult]) -> BuiltContext:
        raise NotImplementedError


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context: BuiltContext) -> RAGAnswer:
        raise NotImplementedError


class BaseVerifier(ABC):
    @abstractmethod
    def verify(self, answer: RAGAnswer, context: BuiltContext) -> VerificationReport:
        raise NotImplementedError
