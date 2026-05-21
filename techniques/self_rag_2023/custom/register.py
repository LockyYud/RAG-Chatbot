from self_rag_verifier import SelfRAGCritiqueVerifier


def register(registry):
    registry.verifiers.register("self_rag_critique", SelfRAGCritiqueVerifier)
    registry.verifiers.register("self_rag_verifier", SelfRAGCritiqueVerifier)
