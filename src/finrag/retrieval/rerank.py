"""Re-ranking dos candidatos com cross-encoder.

O cross-encoder lê pergunta e trecho juntos, o que é mais preciso que comparar vetores
independentes, porém caro. Por isso só é aplicado aos poucos candidatos da fusão.
"""

from typing import Protocol

from finrag.schemas import RetrievedChunk


class Reranker(Protocol):
    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoopReranker:
    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return items


class CrossEncoderReranker:
    def __init__(self, model: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model)

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not items:
            return []
        scores = self.model.predict([(query, it.chunk.search_text) for it in items])
        scored = sorted(zip(items, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            RetrievedChunk(chunk=it.chunk, score=float(s), retriever="rerank") for it, s in scored
        ]


def build_reranker(kind: str, model: str) -> Reranker:
    if kind == "cross-encoder":
        return CrossEncoderReranker(model)
    if kind == "none":
        return NoopReranker()
    raise ValueError(f"Reranker desconhecido: {kind}")
