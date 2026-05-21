from __future__ import annotations

from typing import Any, Callable


Factory = Callable[..., Any]


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Factory] = {}

    def register(self, name: str, factory: Factory) -> None:
        self._items[name] = factory

    def create(self, spec: dict[str, Any] | str | None, **extra: Any) -> Any:
        if spec is None:
            raise KeyError("Missing strategy spec")
        if isinstance(spec, str):
            type_name = spec
            params: dict[str, Any] = {}
        else:
            type_name = str(spec.get("type"))
            params = dict(spec.get("params", {}))
        params.update(extra)
        if type_name not in self._items:
            known = ", ".join(sorted(self._items))
            raise KeyError(f"Unknown strategy '{type_name}'. Known: {known}")
        return self._items[type_name](**params)


parsers = Registry()
cleaners = Registry()
chunkers = Registry()
enrichers = Registry()
retrievers = Registry()
rerankers = Registry()
context_builders = Registry()
generators = Registry()
verifiers = Registry()


def register_defaults() -> None:
    from raglab.indexing.retrievers import BM25Retriever, DenseRetriever, HybridRetriever, OpenAIDenseRetriever, OpenAIHybridRetriever
    from raglab.inference.context_builders.citation_context import CitationContextBuilder
    from raglab.inference.context_builders.topk_context import TopKContextBuilder
    from raglab.inference.generators.extractive import CitationExtractiveGenerator, ExtractiveGenerator
    from raglab.inference.generators.openai_chat import OpenAIChatGenerator
    from raglab.inference.rerankers.no_reranker import NoReranker
    from raglab.inference.rerankers.lexical_overlap import LexicalOverlapReranker
    from raglab.inference.verifiers.citation_coverage import CitationCoverageVerifier
    from raglab.processing.chunkers.fixed_size import FixedSizeChunker
    from raglab.processing.chunkers.heading_aware import HeadingAwareChunker
    from raglab.processing.chunkers.parent_child import ParentChildChunker
    from raglab.processing.chunkers.recursive import RecursiveChunker
    from raglab.processing.cleaners.basic import VietnameseNormalizer, WhitespaceCleaner
    from raglab.processing.enrichers.basic import NoEnricher, SectionTitleEnricher
    from raglab.processing.parsers.text_parser import TextParser

    parsers.register("text", TextParser)
    parsers.register("text_parser", TextParser)
    parsers.register("markdown", TextParser)
    parsers.register("markdown_parser", TextParser)

    cleaners.register("vietnamese_normalizer", VietnameseNormalizer)
    cleaners.register("whitespace_cleaner", WhitespaceCleaner)

    chunkers.register("fixed_size", FixedSizeChunker)
    chunkers.register("recursive", RecursiveChunker)
    chunkers.register("heading_aware", HeadingAwareChunker)
    chunkers.register("parent_child", ParentChildChunker)

    enrichers.register("none", NoEnricher)
    enrichers.register("no_enrichment", NoEnricher)
    enrichers.register("section_title", SectionTitleEnricher)
    enrichers.register("title_injection", SectionTitleEnricher)

    retrievers.register("dense", DenseRetriever)
    retrievers.register("bm25", BM25Retriever)
    retrievers.register("hybrid", HybridRetriever)
    retrievers.register("openai_dense", OpenAIDenseRetriever)
    retrievers.register("openai_hybrid", OpenAIHybridRetriever)

    rerankers.register("none", NoReranker)
    rerankers.register("no_reranker", NoReranker)
    rerankers.register("lexical_overlap", LexicalOverlapReranker)
    rerankers.register("overlap", LexicalOverlapReranker)

    context_builders.register("topk", TopKContextBuilder)
    context_builders.register("topk_context", TopKContextBuilder)
    context_builders.register("citation_context", CitationContextBuilder)

    generators.register("basic", ExtractiveGenerator)
    generators.register("extractive", ExtractiveGenerator)
    generators.register("citation_required", CitationExtractiveGenerator)
    generators.register("openai_chat", OpenAIChatGenerator)

    verifiers.register("citation_coverage", CitationCoverageVerifier)
    verifiers.register("grounded_citation", CitationCoverageVerifier)
