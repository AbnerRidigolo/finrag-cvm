from pathlib import Path

import pytest

from finrag.config import Settings, get_settings
from finrag.ingestion.chunking import chunk_documents
from finrag.ingestion.loaders import load_directory
from finrag.retrieval.dense import DenseStore
from finrag.retrieval.embeddings import HashingEmbedder
from finrag.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "sample" / "docs"

OFFLINE_ENV = {
    "LLM_PROVIDER": "extractive",
    "JUDGE_PROVIDER": "",
    "EMBEDDING_PROVIDER": "hashing",
    "RERANKER": "none",
    "VECTOR_STORE": "chroma",
    "GRAPH_ENABLED": "false",
}


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    """Isola os testes do .env local: sem ele, nenhum teste chama API paga por acidente."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for key, value in OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def chunks():
    return chunk_documents(load_directory(SAMPLE_DOCS))


@pytest.fixture()
def retriever(chunks, tmp_path):
    dense = DenseStore(HashingEmbedder(), path=str(tmp_path / "chroma"), collection="test")
    dense.add(chunks)
    return HybridRetriever(dense, candidates=10)
