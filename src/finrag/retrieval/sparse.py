"""Busca esparsa: BM25 sobre tokens normalizados.

Complementa a busca densa em termos exatos que embeddings tendem a diluir, como números
de artigos, siglas (FIDC, FIP) e CNPJs.
"""

from rank_bm25 import BM25Okapi

from finrag.retrieval.text import tokenize
from finrag.schemas import Chunk, RetrievedChunk


class SparseIndex:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._bm25 = BM25Okapi([tokenize(c.search_text) for c in chunks]) if chunks else None

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), retriever="sparse")
            for i in ranked[:k]
            if scores[i] > 0
        ]
