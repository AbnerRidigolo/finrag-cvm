"""Contrato comum dos bancos vetoriais (Chroma e Pinecone).

O retriever híbrido só depende deste protocolo, então trocar de banco vetorial é uma
mudança de configuração (VECTOR_STORE), sem tocar na lógica de busca.
"""

from typing import Protocol

from finrag.schemas import Chunk, RetrievedChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def search(self, query: str, k: int) -> list[RetrievedChunk]: ...

    def all_chunks(self) -> list[Chunk]:
        """Todos os chunks indexados, usados para reconstruir o índice BM25."""
        ...
