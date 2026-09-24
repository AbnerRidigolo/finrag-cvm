from finrag.ingestion.chunking import _split_long, chunk_document
from finrag.ingestion.loaders import RawDocument


def test_divide_por_artigo():
    doc = RawDocument("n.txt", "Título\n\nArt. 1º Primeira regra.\n\nArt. 2º Segunda regra.")
    chunks = chunk_document(doc)
    assert [c.article for c in chunks] == ["", "Art.1º", "Art.2º"]
    assert chunks[1].text.startswith("Art. 1º")


def test_ids_sao_estaveis_e_unicos():
    doc = RawDocument("n.txt", "Art. 1º A.\nArt. 2º B.")
    first, second = chunk_document(doc), chunk_document(doc)
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_artigo_longo_e_quebrado_com_sobreposicao():
    texto = "Art. 1º " + " ".join(f"Frase número {i}." for i in range(200))
    chunks = chunk_document(RawDocument("n.txt", texto), max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)
    assert all(c.article == "Art.1º" for c in chunks)


def test_split_long_termina_com_texto_curto():
    assert _split_long("curto", 100, 10) == ["curto"]


def test_corpus_de_exemplo(chunks):
    fontes = {c.source for c in chunks}
    assert "regulamento_fundo_exemplo_alpha.txt" in fontes
    assert any(c.article == "Art.4º" for c in chunks)


def test_titulo_do_documento_vai_para_o_texto_indexado():
    doc = RawDocument("n.txt", "REGULAMENTO DO FUNDO X\n\nArt. 4º A taxa de administração é 1%.")
    artigo = chunk_document(doc)[1]
    assert artigo.context == "REGULAMENTO DO FUNDO X"
    assert artigo.search_text.startswith("REGULAMENTO DO FUNDO X | Art.4º\n")
    assert "FUNDO X" not in artigo.text
