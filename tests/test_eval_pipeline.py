from pathlib import Path

from finrag import pipeline
from finrag.config import get_settings
from finrag.eval.retrieval_eval import evaluate, first_relevant_rank, load_dataset
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
