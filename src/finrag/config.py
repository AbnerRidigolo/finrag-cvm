"""Configuração centralizada, lida de variáveis de ambiente ou do arquivo .env."""

import os
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

    # Títulos e assuntos dos documentos (ver ingestion/loaders.py:load_doc_titles).
    doc_titles_path: str = "data/doc_titles.json"

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


def load_api_keys(env_file: str = ".env") -> None:
    """Exporta as chaves de API (*_API_KEY) do .env para os.environ.

    O Settings lê o .env só para os próprios campos; os SDKs (OpenAI, Anthropic, Pinecone)
    procuram a chave em os.environ. Deve ser chamada só nos pontos de entrada (CLIs e a
    construção do agente real na API), nunca na importação de módulos, para que os testes
    não carreguem chaves reais. Variáveis já definidas no ambiente têm precedência.
    """
    from dotenv import dotenv_values

    for key, value in dotenv_values(env_file).items():
        if key.endswith("_API_KEY") and value and key not in os.environ:
            os.environ[key] = value
