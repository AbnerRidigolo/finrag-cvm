from finrag.retrieval.fusion import reciprocal_rank_fusion
from finrag.retrieval.sparse import SparseIndex
from finrag.retrieval.text import tokenize
from finrag.schemas import Chunk, RetrievedChunk


def _rc(cid: str, score: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(chunk=Chunk(id=cid, text=cid, source="s"), score=score, retriever="x")


def test_tokenize_remove_acentos_e_stopwords():
    assert tokenize("A Taxa de Administração é de 1,2%") == ["taxa", "administracao", "1", "2"]


def test_rrf_favorece_consenso_entre_rankings():
    fused = reciprocal_rank_fusion([[_rc("a"), _rc("b"), _rc("c")], [_rc("b"), _rc("c")]])
    assert [r.chunk.id for r in fused][:2] == ["b", "c"]
    assert all(r.retriever == "hybrid" for r in fused)


def test_bm25_acha_termo_exato(chunks):
    results = SparseIndex(chunks).search("índice de subordinação cotas subordinadas", k=3)
    assert results[0].chunk.article == "Art.6º"


def test_bm25_vazio():
    assert SparseIndex([]).search("qualquer coisa", k=3) == []


def test_hibrido_retorna_artigo_certo(retriever):
    results = retriever.retrieve("taxa de administração do Fundo Exemplo Alpha", k=3)
    assert any(
        r.chunk.source == "regulamento_fundo_exemplo_alpha.txt" and r.chunk.article == "Art.4º"
        for r in results
    )


def test_modos_de_busca(retriever):
    for mode in ("dense", "sparse", "hybrid"):
        results = retriever.retrieve("chamada de capital inadimplente", k=2, mode=mode)
        assert 0 < len(results) <= 2


def test_bm25_reconstruido_a_partir_do_chroma(retriever, chunks):
    assert len(retriever.sparse.chunks) == len(chunks)
