"""Recuperação híbrida: densa + esparsa, fusão por RRF e re-ranking opcional."""

from typing import Literal

from finrag.retrieval.fusion import reciprocal_rank_fusion
from finrag.retrieval.rerank import NoopReranker, Reranker
from finrag.retrieval.sparse import SparseIndex
from finrag.retrieval.vectorstore import VectorStore
from finrag.schemas import RetrievedChunk

Mode = Literal["dense", "sparse", "hybrid"]


class HybridRetriever:
    def __init__(
        self,
        dense: VectorStore,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ) -> None:
        self.dense = dense
        self.sparse = SparseIndex(dense.all_chunks())
        self.reranker = reranker or NoopReranker()
        self.candidates = candidates

    def refresh(self) -> None:
        """Reconstrói o BM25 a partir do banco vetorial, a fonte única de verdade dos chunks."""
        self.sparse = SparseIndex(self.dense.all_chunks())

    def retrieve(
        self, query: str, k: int = 5, mode: Mode = "hybrid", rerank: bool = True
    ) -> list[RetrievedChunk]:
        if mode == "dense":
            results = self.dense.search(query, self.candidates)
        elif mode == "sparse":
            results = self.sparse.search(query, self.candidates)
        else:
            results = reciprocal_rank_fusion(
                [
                    self.dense.search(query, self.candidates),
                    self.sparse.search(query, self.candidates),
                ]
            )
        if rerank:
            results = self.reranker.rerank(query, results)
        return results[:k]
