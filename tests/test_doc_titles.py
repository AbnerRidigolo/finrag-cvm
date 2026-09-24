import json
from pathlib import Path

from finrag.agent import FinRAGAgent
from finrag.eval.build_dataset import build_dataset
from finrag.ingestion.chunking import chunk_documents
from finrag.ingestion.loaders import RawDocument, load_doc_titles
from finrag.llm import Completion, ExtractiveLLM

ROOT = Path(__file__).resolve().parents[1]
# Primeira linha típica de um PDF da CVM: o timbre, não o título.
PDF = RawDocument("anexo.pdf", "COMISSÃO DE VALORES MOBILIÁRIOS\nArt. 1º A classe é fechada.")


def test_titulo_do_mapa_substitui_a_primeira_linha():
    titles = {"anexo.pdf": "Resolução CVM nº 175 – Anexo Normativo II – FIDC"}
    chunk = chunk_documents([PDF], titles=titles)[-1]
    assert chunk.context == titles["anexo.pdf"]
    assert chunk.search_text.startswith("Resolução CVM nº 175 – Anexo Normativo II – FIDC | Art.1º")


def test_arquivo_fora_do_mapa_avisa_e_usa_a_primeira_linha(caplog):
    with caplog.at_level("WARNING", logger="finrag"):
        chunk = chunk_documents([PDF], titles={"outro.pdf": "X"})[-1]
    assert chunk.context == "COMISSÃO DE VALORES MOBILIÁRIOS"
    assert "anexo.pdf não está no mapa de títulos" in caplog.text


def test_sem_mapa_nao_avisa(caplog):
    with caplog.at_level("WARNING", logger="finrag"):
        chunk_documents([PDF])
    assert not caplog.text


def test_mapa_versionado_tem_title_e_subject_em_todas_as_entradas(tmp_path):
    titles = load_doc_titles(ROOT / "data" / "doc_titles.json")
    assert len(titles) == 16
    assert all(v["title"].strip() and v["subject"].strip() for v in titles.values())
    assert load_doc_titles(tmp_path / "nao-existe.json") == {}


class RecordingLLM:
    name = model = "recording"

    def __init__(self):
        self.prompts = []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        return Completion(json.dumps({"question": "Pergunta sobre o Croma FIDC?"}), "r", "r")


def test_gerador_recebe_o_assunto_e_nao_o_titulo():
    doc = RawDocument("croma.pdf", "REGULAMENTO DO\nArtigo 1º O prazo de duração é indeterminado.")
    chunks = chunk_documents([doc], titles={"croma.pdf": "Regulamento do Croma FIDC RL"})
    llm = RecordingLLM()
    rows = build_dataset(llm, chunks, n=1, min_chars=10, subjects={"croma.pdf": "Croma FIDC"})
    assert llm.prompts[0].startswith("Assunto: Croma FIDC\n")
    assert "Regulamento do Croma FIDC RL" not in llm.prompts[0]
    assert rows[0]["subject"] == "Croma FIDC"


def test_fontes_da_resposta_trazem_o_titulo(chunks, tmp_path):
    from finrag.retrieval.dense import DenseStore
    from finrag.retrieval.embeddings import HashingEmbedder
    from finrag.retrieval.hybrid import HybridRetriever

    titled = chunk_documents([PDF], titles={"anexo.pdf": "Anexo II – FIDC"})
    dense = DenseStore(HashingEmbedder(), path=str(tmp_path / "c"), collection="titulos")
    dense.add(titled)
    answer = FinRAGAgent(ExtractiveLLM(), HybridRetriever(dense, candidates=5)).ask(
        "A classe é fechada?"
    )
    assert answer.sources[0].title == "Anexo II – FIDC"
