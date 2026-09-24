"""Avaliação da recuperação: compara denso, esparso, híbrido e híbrido + re-ranking.

Cada linha do dataset (JSONL) tem a pergunta e os trechos considerados relevantes,
identificados por arquivo e artigo:
    {"question": "...", "relevant": [{"source": "arquivo.txt", "article": "Art.5º"}]}

Métricas:
- Hit@1 e Hit@k: fração das perguntas com um trecho relevante na 1ª posição / no top-k.
- MRR@k: média do inverso da posição do primeiro trecho relevante.
- Latência p50 e p95 da recuperação, em milissegundos.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from finrag.config import get_settings, load_api_keys
from finrag.factory import build_retriever
from finrag.retrieval.hybrid import HybridRetriever
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


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1]


def evaluate(retriever: HybridRetriever, dataset: list[dict], k: int, mode: str, rerank: bool):
    hits, hits_at_1, rr, latencies = 0, 0, 0.0, []
    for row in dataset:
        start = time.perf_counter()
        results = retriever.retrieve(row["question"], k=k, mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - start) * 1000)
        rank = first_relevant_rank(results, row["relevant"])
        if rank is not None:
            hits += 1
            hits_at_1 += rank == 1
            rr += 1.0 / rank
    n = len(dataset) or 1
    return {
        "hit": hits / n,
        "hit1": hits_at_1 / n,
        "mrr": rr / n,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


def report(retriever: HybridRetriever, dataset: list[dict], k: int, with_rerank: bool) -> str:
    configs = [
        ("Denso", "dense", False),
        ("Esparso (BM25)", "sparse", False),
        ("Híbrido (RRF)", "hybrid", False),
    ]
    if with_rerank:
        configs.append(("Híbrido + re-ranking", "hybrid", True))
    lines = [
        f"| Configuração | Hit@1 | Hit@{k} | MRR@{k} | p50 (ms) | p95 (ms) |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for label, mode, rerank in configs:
        m = evaluate(retriever, dataset, k, mode, rerank)
        lines.append(
            f"| {label} | {m['hit1']:.2f} | {m['hit']:.2f} | {m['mrr']:.2f} "
            f"| {m['p50_ms']:.0f} | {m['p95_ms']:.0f} |"
        )
    return "\n".join(lines)


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--output", help="Salva a tabela em Markdown neste arquivo")
    args = parser.parse_args()

    s = get_settings()
    dataset = load_dataset(args.dataset)
    retriever = build_retriever(s)
    header = (
        f"Banco vetorial: {s.vector_store} | embeddings: {s.embedding_provider} "
        f"({s.embedding_model}) | reranker: {s.reranker} | {len(dataset)} perguntas | k={args.k}"
    )
    table = report(retriever, dataset, args.k, with_rerank=s.reranker != "none")
    print(f"{header}\n\n{table}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(f"{header}\n\n{table}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
