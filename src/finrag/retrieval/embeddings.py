"""Provedores de embeddings.

`HashingEmbedder` existe para rodar a demo e o CI sem chave de API nem download de
modelos. Ele não entende semântica (é um saco de palavras projetado por hashing), então
os resultados de avaliação só são representativos com um embedder real.
"""

import hashlib
import math
from typing import Protocol

from finrag.metrics import EMBEDDING_TOKENS
from finrag.retrieval.text import tokenize


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class OpenAIEmbedder:
    def __init__(self, model: str, batch_size: int = 128) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            if response.usage:
                EMBEDDING_TOKENS.labels(self.model).inc(response.usage.total_tokens)
        return vectors


def build_embedder(provider: str, model: str) -> Embedder:
    if provider == "openai":
        return OpenAIEmbedder(model)
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Provedor de embeddings desconhecido: {provider}")
