from finrag.ingestion.chunking import _split_long, chunk_document
from finrag.ingestion.loaders import RawDocument


def test_divide_por_artigo():
    doc = RawDocument("n.txt", "Título\n\nArt. 1º Primeira regra.\n\nArt. 2º Segunda regra.")
    chunks = chunk_document(doc)
    assert [c.article for c in chunks] == ["", "Art.1º", "Art.2º"]
    assert chunks[1].text.startswith("Art. 1º")


def test_regulamento_com_artigo_por_extenso():
    texto = (
        "REGULAMENTO DO FUNDO X\n"
        "Artigo 01. O Fundo é regido pela Resolução.\n"
        "Artigo 2º A Classe é única, conforme\n"
        "Artigo 12 deste Regulamento.\n"
        "Artigo 3º. O prazo é indeterminado."
    )
    chunks = chunk_document(RawDocument("r.pdf", texto))
    assert [c.article for c in chunks] == ["", "Art.1", "Art.2º", "Art.3º"]
    # Linha que começa com "Artigo 12 deste..." é continuação de frase, não um novo artigo.
    assert "Artigo 12 deste Regulamento" in chunks[2].text
    assert chunks[1].context == "REGULAMENTO DO FUNDO X"


def test_regulamento_com_clausulas():
    texto = "Cláusula 1 – Do Fundo. O Fundo é um FIDC.\nCláusula 2. Da Taxa. A taxa é 1%."
    chunks = chunk_document(RawDocument("r.pdf", texto))
    assert [c.article for c in chunks] == ["Cláusula 1", "Cláusula 2"]


def test_incisos_numerados_nao_viram_artigo():
    texto = "Artigo 2º Para os fins deste Regulamento:\n1. Administradora: a X.\n2. Gestora: a Y."
    chunks = chunk_document(RawDocument("r.pdf", texto))
    assert len(chunks) == 1 and chunks[0].article == "Art.2º"


def test_artigo_incluido_com_sufixo_nao_colide_com_o_original():
    # O PDF da CVM usa tanto hífen quanto travessão (U+2013) no sufixo.
    texto = "Art. 110. É vedado X.\nArt. 110-A. A classe pode Y.\nArt. 110–B. Z.\nArt. 111. W."
    chunks = chunk_document(RawDocument("n.pdf", texto))
    assert [c.article for c in chunks] == ["Art.110", "Art.110-A", "Art.110-B", "Art.111"]


def test_numeracao_que_recomeca_no_anexo_vira_outra_parte():
    texto = (
        "Artigo 1º O Fundo é um FIDC.\nArtigo 2º O Fundo tem prazo indeterminado.\n"
        "ANEXO I – DA CLASSE ÚNICA\n"
        "Artigo 1º A Classe é fechada.\nArtigo 2º A Classe tem taxa de 1%."
    )
    chunks = chunk_document(RawDocument("r.pdf", texto))
    assert [c.article for c in chunks] == [
        "Art.1º",
        "Art.2º",
        "Art.1º (parte 2)",
        "Art.2º (parte 2)",
    ]


REVOGADA = "Art. 134. Os FIDC devem adaptar-se até 31 de dezembro de 2023."
VIGENTE = "Art. 134. Os FIDC devem adaptar-se até 1º de abril de 2024."


def test_texto_consolidado_mantem_so_a_redacao_vigente(caplog):
    texto = f"Art. 133. X.\n{REVOGADA}\n{VIGENTE}\nArt. 135. Y."
    with caplog.at_level("INFO", logger="finrag"):
        chunks = chunk_document(RawDocument("resol175consolid_ParteGeral.pdf", texto))
    assert [c.article for c in chunks] == ["Art.133", "Art.134", "Art.135"]
    assert "1º de abril de 2024" in chunks[1].text
    assert "31 de dezembro de 2023" not in " ".join(c.text for c in chunks)
    assert "resol175consolid_ParteGeral.pdf Art.134" in caplog.text


def test_deduplicacao_so_vale_para_textos_consolidados():
    texto = f"{REVOGADA}\n{VIGENTE}"
    chunks = chunk_document(RawDocument("regulamento_fidc.pdf", texto))
    assert len(chunks) == 2


def test_deduplicacao_nao_descarta_pedacos_de_artigo_longo():
    longo = "Art. 12. " + " ".join(f"Frase número {i}." for i in range(200))
    texto = f"Art. 12. Redação antiga e curta.\n{longo}\nArt. 13. Fim."
    chunks = chunk_document(
        RawDocument("resol175consolid_Anexo01.pdf", texto), max_chars=300, overlap=50
    )
    pedacos = [c for c in chunks if c.article == "Art.12"]
    # Os vários pedaços consecutivos do Art. 12 vigente ficam; só a seção antiga sai.
    assert len(pedacos) > 3
    assert "Redação antiga" not in " ".join(c.text for c in chunks)
    assert "Frase número 199." in pedacos[-1].text


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
