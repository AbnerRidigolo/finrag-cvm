"""Configuração centralizada, lida de variáveis de ambiente ou do arquivo .env."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "extractive"] = "extractive"
    llm_model: str = "gpt-4o-mini"

    embedding_provider: Literal["openai", "hashing"] = "hashing"
    embedding_model: str = "text-embedding-3-small"

    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    chroma_path: str = ".chroma"
    collection_name: str = "cvm_docs"

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
