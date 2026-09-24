"""Busca densa com ChromaDB (modo embarcado, persistido em disco). Padrão para desenvolvimento."""

import chromadb

from finrag.retrieval.embeddings import Embedder
from finrag.schemas import Chunk, RetrievedChunk


class ChromaStore:
    def __init__(self, embedder: Embedder, path: str | None, collection: str) -> None:
        client = chromadb.PersistentClient(path=path) if path else chromadb.EphemeralClient()
        self.embedder = embedder
        self.collection = client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.collection.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embedder.embed([c.search_text for c in batch]),
                metadatas=[
                    {
                        "source": c.source,
                        "article": c.article,
                        "position": c.position,
                        "context": c.context,
                    }
                    for c in batch
                ],
            )

    def all_chunks(self) -> list[Chunk]:
        data = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk(id=i, text=doc, **meta)
            for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True)
        ]

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk=Chunk(id=i, text=doc, **meta),
                score=1.0 - dist,
                retriever="dense",
            )
            for i, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


# Nome antigo mantido por compatibilidade.
DenseStore = ChromaStore
