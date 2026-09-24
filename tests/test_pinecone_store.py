"""Testa o adaptador Pinecone com um índice falso que imita o SDK (sem rede)."""

from types import SimpleNamespace

from finrag.retrieval.embeddings import HashingEmbedder
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.pinecone_store import PineconeStore, _page_ids, pinecone_id


class FakeIndex:
    """Índice em memória com a mesma interface usada do SDK: upsert, query, list, fetch."""

    def __init__(self, page_size: int = 7) -> None:
        self.data: dict[str, dict[str, dict]] = {}
        self.page_size = page_size

    def upsert(self, vectors, namespace):
        ns = self.data.setdefault(namespace, {})
        for v in vectors:
            assert v["id"].isascii()
            ns[v["id"]] = v

    def query(self, vector, top_k, include_metadata, namespace):
        items = self.data.get(namespace, {}).values()
        scored = sorted(
            ((sum(a * b for a, b in zip(vector, v["values"], strict=True)), v) for v in items),
            key=lambda p: p[0],
            reverse=True,
        )[:top_k]
        return SimpleNamespace(
            matches=[
                SimpleNamespace(id=v["id"], score=s, metadata=v["metadata"]) for s, v in scored
            ]
        )

    def list(self, namespace):
        ids = list(self.data.get(namespace, {}))
        for i in range(0, len(ids), self.page_size):
            yield SimpleNamespace(
                vectors=[SimpleNamespace(id=x) for x in ids[i : i + self.page_size]]
            )

    def fetch(self, ids, namespace):
        ns = self.data.get(namespace, {})
        return SimpleNamespace(
            vectors={i: SimpleNamespace(metadata=ns[i]["metadata"]) for i in ids if i in ns}
        )


def _store(chunks):
    store = PineconeStore(HashingEmbedder(), "idx", namespace="t", index=FakeIndex())
    store.add(chunks)
    return store


def test_ids_ascii_e_estaveis():
    assert pinecone_id("regulamento_ação.txt:1:abc").isascii()
    assert pinecone_id("x") == pinecone_id("x")


def test_all_chunks_percorre_todas_as_paginas(chunks):
    recovered = _store(chunks).all_chunks()
    assert {c.id for c in recovered} == {c.id for c in chunks}
    original = {c.id: c for c in chunks}
    assert all(c == original[c.id] for c in recovered)


def test_busca_e_hibrido_sobre_pinecone(chunks):
    retriever = HybridRetriever(_store(chunks), candidates=10)
    results = retriever.retrieve("taxa de administração do Fundo Exemplo Alpha", k=3)
    assert any(r.chunk.article == "Art.4º" for r in results)
    assert len(retriever.sparse.chunks) == len(chunks)


def test_page_ids_aceita_formato_antigo_e_novo():
    assert _page_ids(["a", "b"]) == ["a", "b"]
    assert _page_ids(SimpleNamespace(vectors=[SimpleNamespace(id="c")])) == ["c"]
