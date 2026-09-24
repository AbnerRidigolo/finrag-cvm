from pathlib import Path

from finrag import pipeline
from finrag.config import get_settings
from finrag.eval.retrieval_eval import (
    bootstrap_ci,
    evaluate,
    first_relevant_rank,
    load_dataset,
    paired_difference_ci,
    per_question,
    report,
)
from finrag.schemas import Chunk, RetrievedChunk

DATASET = Path(__file__).resolve().parents[1] / "data" / "eval" / "questions.jsonl"


def test_first_relevant_rank():
    results = [
        RetrievedChunk(
            chunk=Chunk(id=str(i), text="", source="a.txt", article=f"Art.{i}º"),
            score=0,
            retriever="x",
        )
        for i in range(1, 4)
    ]
    assert first_relevant_rank(results, [{"source": "a.txt", "article": "Art.2º"}]) == 2
    assert first_relevant_rank(results, [{"source": "b.txt", "article": "Art.1º"}]) is None


def test_hibrido_supera_ou_empata_com_denso_no_corpus_de_exemplo(retriever):
    dataset = load_dataset(DATASET)
    dense = evaluate(retriever, dataset, k=5, mode="dense", rerank=False)
    hybrid = evaluate(retriever, dataset, k=5, mode="hybrid", rerank=False)
    assert hybrid["hit"] >= dense["hit"]
    assert hybrid["hit"] >= 0.8


def test_cli_index_e_ask(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    docs = Path(__file__).resolve().parents[1] / "data" / "sample" / "docs"
    pipeline.cmd_index(str(docs))
    pipeline.cmd_ask("O que é linha d'água?")
    out = capsys.readouterr().out
    assert "chunks indexados" in out and "[rota: normas]" in out
    get_settings.cache_clear()


def test_bootstrap_e_deterministico_e_contem_a_media():
    values = [1.0] * 30 + [0.0] * 20  # média 0,6
    low, high = bootstrap_ci(values)
    assert bootstrap_ci(values) == (low, high)  # semente fixa
    assert 0.4 < low < 0.6 < high < 0.8
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_diferenca_pareada():
    a, b = [1.0, 1.0, 0.0, 1.0] * 10, [1.0, 0.0, 0.0, 1.0] * 10
    mean, low, high = paired_difference_ci(a, b)
    assert mean == 0.25 and 0.0 < low <= mean <= high  # a nunca perde: IC acima de zero
    assert paired_difference_ci(a, a) == (0.0, 0.0, 0.0)


def test_per_question_a_partir_das_posicoes():
    values = per_question([1, 3, None])
    assert values["hit1"] == [1.0, 0.0, 0.0]
    assert values["hit"] == [1.0, 1.0, 0.0]
    assert values["mrr"] == [1.0, 1 / 3, 0.0]


def test_relatorio_mostra_intervalos_e_diferencas(retriever):
    table = report(retriever, load_dataset(DATASET), k=5, with_rerank=False)
    assert "Diferença pareada" in table and "Híbrido (RRF) − Denso" in table
    assert "[" in table.splitlines()[2]  # linha do Denso traz o IC
