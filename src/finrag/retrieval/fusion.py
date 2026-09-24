"""Reciprocal Rank Fusion (Cormack et al., 2009).

Combina rankings pela posição, não pelo score. Isso evita calibrar escalas diferentes
(similaridade de cosseno x BM25) e é robusto quando um dos retrievers erra feio.
"""

from finrag.schemas import RetrievedChunk


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk.id] = scores.get(item.chunk.id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(item.chunk.id, item)
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    return [
        RetrievedChunk(chunk=by_id[cid].chunk, score=scores[cid], retriever="hybrid")
        for cid in ordered
    ]
