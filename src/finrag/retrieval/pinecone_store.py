"""Busca densa com Pinecone (serverless), alternativa gerenciada ao Chroma para produção.

Diferenças de modelagem em relação ao Chroma:
- O Pinecone não guarda o texto separado do vetor, então o texto do chunk vai nos
  metadados (o limite é de 40 KB por vetor, folgado para chunks de ~1.200 caracteres).
- IDs precisam ser ASCII. Os IDs do projeto contêm o nome do arquivo, que pode ter
  acentos, então o ID no Pinecone é um hash e o ID original fica nos metadados.
- Para reconstruir o BM25, os IDs são listados página a página e buscados em lote
  (list + fetch), já que o Pinecone não tem um "get all".
"""

import hashlib
from collections.abc import Iterable, Iterator
from typing import Any

from finrag.retrieval.embeddings import Embedder
from finrag.schemas import Chunk, RetrievedChunk

UPSERT_BATCH = 100
FETCH_BATCH = 100


def pinecone_id(chunk_id: str) -> str:
    return hashlib.sha1(chunk_id.encode()).hexdigest()


def _chunk_from_metadata(meta: dict[str, Any]) -> Chunk:
    return Chunk(
        id=meta["chunk_id"],
        text=meta["text"],
        source=meta.get("source", ""),
        article=meta.get("article", ""),
        position=int(meta.get("position", 0)),
        context=meta.get("context", ""),
    )


def _page_ids(page: Any) -> list[str]:
    # SDKs recentes devolvem objetos com `.vectors` (itens com `.id`); versões antigas,
    # listas de IDs. Aceitar os dois evita quebrar numa atualização do SDK.
    if isinstance(page, list | tuple):
        return [str(i) for i in page]
    return [item.id for item in page.vectors]


def _batched(items: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class PineconeStore:
    def __init__(
        self,
        embedder: Embedder,
        index_name: str,
        namespace: str = "cvm-docs",
        cloud: str = "aws",
        region: str = "us-east-1",
        index: Any = None,
    ) -> None:
        self.embedder = embedder
        self.namespace = namespace
        if index is None:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone()  # lê PINECONE_API_KEY do ambiente
            if not pc.has_index(index_name):
                dimension = len(embedder.embed(["dimensão"])[0])
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region),
                )
            index = pc.Index(index_name)
        self.index = index

    def add(self, chunks: list[Chunk]) -> None:
        for i in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[i : i + UPSERT_BATCH]
            vectors = self.embedder.embed([c.search_text for c in batch])
            self.index.upsert(
                vectors=[
                    {
                        "id": pinecone_id(c.id),
                        "values": v,
                        "metadata": {
                            "chunk_id": c.id,
                            "text": c.text,
                            "source": c.source,
                            "article": c.article,
                            "position": c.position,
                            "context": c.context,
                        },
                    }
                    for c, v in zip(batch, vectors, strict=True)
                ],
                namespace=self.namespace,
            )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        response = self.index.query(
            vector=self.embedder.embed([query])[0],
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )
        return [
            RetrievedChunk(
                chunk=_chunk_from_metadata(match.metadata), score=match.score, retriever="dense"
            )
            for match in response.matches
            if match.metadata
        ]

    def _all_ids(self) -> list[str]:
        pages: Iterable[Any] = self.index.list(namespace=self.namespace)
        return [vid for page in pages for vid in _page_ids(page)]

    def all_chunks(self) -> list[Chunk]:
        chunks = []
        for ids in _batched(self._all_ids(), FETCH_BATCH):
            fetched = self.index.fetch(ids=ids, namespace=self.namespace)
            chunks.extend(
                _chunk_from_metadata(vec.metadata)
                for vec in fetched.vectors.values()
                if vec.metadata
            )
        return sorted(chunks, key=lambda c: (c.source, c.position))
