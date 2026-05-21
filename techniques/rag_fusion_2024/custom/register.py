from rag_fusion_retriever import RAGFusionRetriever


def register(registry):
    registry.retrievers.register("rag_fusion", RAGFusionRetriever)
    registry.retrievers.register("rag_fusion_retriever", RAGFusionRetriever)
