from pathlib import Path

import pytest

from finrag.ingestion.chunking import chunk_documents
from finrag.ingestion.loaders import load_directory
from finrag.retrieval.dense import DenseStore
from finrag.retrieval.embeddings import HashingEmbedder
from finrag.retrieval.hybrid import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "sample" / "docs"


@pytest.fixture(scope="session")
def chunks():
    return chunk_documents(load_directory(SAMPLE_DOCS))


@pytest.fixture()
def retriever(chunks, tmp_path):
    dense = DenseStore(HashingEmbedder(), path=str(tmp_path / "chroma"), collection="test")
    dense.add(chunks)
    return HybridRetriever(dense, candidates=10)
