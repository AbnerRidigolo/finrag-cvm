"""Avaliação da recuperação: compara denso, esparso, híbrido e híbrido + re-ranking.

Cada linha do dataset (JSONL) tem a pergunta e os trechos considerados relevantes,
identificados por arquivo e artigo:
    {"question": "...", "relevant": [{"source": "arquivo.txt", "article": "Art.5º"}]}

Métricas:
- Hit@1 e Hit@k: fração das perguntas com um trecho relevante na 1ª posição / no top-k.
- MRR@k: média do inverso da posição do primeiro trecho relevante.
- Latência p50 e p95 da recuperação, em milissegundos.

Com poucas dezenas de perguntas, uma diferença de 0,05 pode ser ruído. Cada métrica vem com
um intervalo de confiança de 95% por bootstrap (reamostragem das perguntas com reposição,
semente fixa), e as comparações entre configurações usam a diferença pareada: as mesmas
perguntas reamostradas nos dois lados, o que é mais sensível do que comparar dois ICs.
"""

import argparse
import json
import random
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


BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42
METRICS = ("hit1", "hit", "mrr")


def per_question(ranks: list[int | None]) -> dict[str, list[float]]:
    """Valor de cada métrica por pergunta, a partir da posição do primeiro trecho relevante."""
    return {
        "hit1": [float(r == 1) for r in ranks],
        "hit": [float(r is not None) for r in ranks],
        "mrr": [1.0 / r if r else 0.0 for r in ranks],
    }


def bootstrap_ci(
    values: list[float],
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    """IC percentil da média por bootstrap. Semente fixa: a mesma entrada dá o mesmo IC."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_samples))
    tail = (1 - level) / 2
    return means[int(tail * n_samples)], means[min(int((1 - tail) * n_samples), n_samples - 1)]


def paired_difference_ci(a: list[float], b: list[float], **kwargs) -> tuple[float, float, float]:
    """Diferença média (a - b) e seu IC, reamostrando as mesmas perguntas nos dois lados."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    low, high = bootstrap_ci(diffs, **kwargs)
    return (sum(diffs) / len(diffs) if diffs else 0.0), low, high


def evaluate(retriever: HybridRetriever, dataset: list[dict], k: int, mode: str, rerank: bool):
    ranks: list[int | None] = []
    latencies = []
    for row in dataset:
        start = time.perf_counter()
        results = retriever.retrieve(row["question"], k=k, mode=mode, rerank=rerank)
        latencies.append((time.perf_counter() - start) * 1000)
        ranks.append(first_relevant_rank(results, row["relevant"]))
    values = per_question(ranks)
    n = len(dataset) or 1
    return {
        **{m: sum(v) / n for m, v in values.items()},
        **{f"{m}_ci": bootstrap_ci(v) for m, v in values.items()},
        "per_question": values,
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
    }


def _fmt(value: float, ci: tuple[float, float]) -> str:
    return f"{value:.2f} [{ci[0]:.2f}–{ci[1]:.2f}]"


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
    results = {}
    for label, mode, rerank in configs:
        m = results[label] = evaluate(retriever, dataset, k, mode, rerank)
        lines.append(
            f"| {label} | {_fmt(m['hit1'], m['hit1_ci'])} | {_fmt(m['hit'], m['hit_ci'])} "
            f"| {_fmt(m['mrr'], m['mrr_ci'])} | {m['p50_ms']:.0f} | {m['p95_ms']:.0f} |"
        )
    comparisons = [("Híbrido (RRF)", "Denso"), ("Híbrido (RRF)", "Esparso (BM25)")]
    if with_rerank:
        comparisons.append(("Híbrido + re-ranking", "Híbrido (RRF)"))
    lines += [
        "",
        f"Diferença pareada (IC 95%, bootstrap com {BOOTSTRAP_SAMPLES} reamostragens):",
        "",
        f"| Comparação | Δ Hit@1 | Δ Hit@{k} | Δ MRR@{k} |",
        "| :-- | --: | --: | --: |",
    ]
    for a, b in comparisons:
        cells = []
        for metric in METRICS:
            mean, low, high = paired_difference_ci(
                results[a]["per_question"][metric], results[b]["per_question"][metric]
            )
            cells.append(f"{mean:+.2f} [{low:+.2f} a {high:+.2f}]")
        lines.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    lines += ["", "Entre colchetes: IC de 95% por bootstrap percentil (semente fixa)."]
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
