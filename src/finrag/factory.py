"""Monta os componentes a partir da configuração (usado pela CLI e pela API)."""

from finrag.agent import FinRAGAgent
from finrag.config import Settings, get_settings
from finrag.graph.store import GraphStore
from finrag.llm import build_llm
from finrag.retrieval.dense import DenseStore
from finrag.retrieval.embeddings import build_embedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.rerank import build_reranker


def build_dense(settings: Settings | None = None) -> DenseStore:
    s = settings or get_settings()
    embedder = build_embedder(s.embedding_provider, s.embedding_model)
    return DenseStore(embedder, s.chroma_path, s.collection_name)


def build_graph(settings: Settings | None = None) -> GraphStore | None:
    s = settings or get_settings()
    if not s.graph_enabled:
        return None
    from finrag.graph.store import Neo4jGraphStore

    return Neo4jGraphStore(s.neo4j_uri, s.neo4j_user, s.neo4j_password)


def build_agent(settings: Settings | None = None) -> FinRAGAgent:
    s = settings or get_settings()
    retriever = HybridRetriever(
        build_dense(s),
        reranker=build_reranker(s.reranker, s.reranker_model),
        candidates=s.candidates_per_retriever,
    )
    return FinRAGAgent(build_llm(s), retriever, graph=build_graph(s), top_k=s.top_k)
