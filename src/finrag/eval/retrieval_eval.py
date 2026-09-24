"""Avaliação da recuperação: compara denso, esparso, híbrido e híbrido + re-ranking.

Cada linha do dataset (JSONL) tem a pergunta e os trechos considerados relevantes,
identificados por arquivo e artigo:
    {"question": "...", "relevant": [{"source": "arquivo.txt", "article": "Art.5º"}]}

Métricas:
- Hit@k: fração das perguntas com pelo menos um trecho relevante no top-k.
- MRR@k: média do inverso da posição do primeiro trecho relevante.
"""

import argparse
import json
from pathlib import Path

from finrag.config import get_settings
from finrag.factory import build_dense
from finrag.retrieval.hybrid import HybridRetriever
from finrag.retrieval.rerank import build_reranker
from finrag.schemas import RetrievedChunk


def load_dataset(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def first_relevant_rank(results: list[RetrievedChunk], relevant: list[dict]) -> int | None:
    targets = {(r["source"], r.get("article", "")) for r in relevant}
    for rank, item in enumerate(results, start=1):
        key = (item.chunk.source, item.chunk.article)
        if key in targets or (item.chunk.source, "") in targets:
            return rank
    return None


def evaluate(retriever: HybridRetriever, dataset: list[dict], k: int, mode: str, rerank: bool):
    hits, rr = 0, 0.0
    for row in dataset:
        results = retriever.retrieve(row["question"], k=k, mode=mode, rerank=rerank)
        rank = first_relevant_rank(results, row["relevant"])
        if rank is not None:
            hits += 1
            rr += 1.0 / rank
    n = len(dataset) or 1
    return {"hit": hits / n, "mrr": rr / n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    s = get_settings()
    dataset = load_dataset(args.dataset)
    retriever = HybridRetriever(
        build_dense(s), build_reranker(s.reranker, s.reranker_model), s.candidates_per_retriever
    )
    configs = [
        ("denso", "dense", False),
        ("esparso (BM25)", "sparse", False),
        ("híbrido (RRF)", "hybrid", False),
    ]
    if s.reranker != "none":
        configs.append(("híbrido + re-ranking", "hybrid", True))

    print(f"Embeddings: {s.embedding_provider} | {len(dataset)} perguntas | k={args.k}\n")
    print(f"| Configuração | Hit@{args.k} | MRR@{args.k} |")
    print("| :-- | --: | --: |")
    for label, mode, rerank in configs:
        m = evaluate(retriever, dataset, args.k, mode, rerank)
        print(f"| {label} | {m['hit']:.2f} | {m['mrr']:.2f} |")


if __name__ == "__main__":
    main()
