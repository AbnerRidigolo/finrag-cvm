"""Configuração centralizada, lida de variáveis de ambiente ou do arquivo .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "extractive"] = "extractive"
    llm_model: str = "gpt-4o-mini"

    # Juiz da avaliação de respostas. Idealmente um modelo diferente (e mais forte)
    # do que o que gera as respostas, para reduzir o viés de autoavaliação.
    judge_provider: Literal["openai", "anthropic", ""] = ""
    judge_model: str = ""

    # Preço em USD por milhão de tokens: {"modelo": [entrada, saída]}.
    llm_prices: dict[str, list[float]] = {}

    embedding_provider: Literal["openai", "hashing"] = "hashing"
    embedding_model: str = "text-embedding-3-small"

    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    vector_store: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: str = ".chroma"
    collection_name: str = "cvm_docs"
    pinecone_index: str = "finrag-cvm"
    pinecone_namespace: str = "cvm-docs"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150
    candidates_per_retriever: int = 20
    top_k: int = 5

    graph_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "troque-esta-senha"


@lru_cache
def get_settings() -> Settings:
    return Settings()
