from hyde_retriever import HyDERetriever


def register(registry):
    registry.retrievers.register("hyde", HyDERetriever)
    registry.retrievers.register("hyde_retriever", HyDERetriever)
