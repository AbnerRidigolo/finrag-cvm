"""Gera um dataset de avaliação a partir dos documentos reais já indexados.

Para cada chunk sorteado, o LLM escreve uma pergunta que só aquele trecho responde. O
trecho de origem vira o gabarito (fonte + artigo), e o dataset pode ser usado pela
avaliação de recuperação e pela de respostas.

Limitação conhecida: perguntas sintéticas tendem a repetir palavras do trecho, o que
favorece o BM25. Por isso o prompt pede paráfrase, e vale revisar à mão uma amostra
antes de publicar os números.

Uso:
    python -m finrag.eval.build_dataset data/eval/cvm_questions.jsonl -n 50
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from finrag.config import get_settings
from finrag.factory import build_vector_store
from finrag.llm import LLM, build_judge
from finrag.schemas import Chunk

GENERATOR_SYSTEM = """Você cria perguntas de avaliação para um sistema de busca sobre
normas e regulamentos de fundos de investimento brasileiros.
Dado um trecho, escreva UMA pergunta em português que:
- seja respondida por este trecho, e não dependa de outros;
- soe como a dúvida de um analista, parafraseando em vez de copiar frases do trecho;
- não mencione "o trecho", "o texto" ou "o artigo acima".
Se o trecho não tiver conteúdo substantivo (índice, cabeçalho, assinatura), responda SKIP.
Responda somente com JSON: {"question": "..."} ou {"question": "SKIP"}."""


def sample_chunks(chunks: list[Chunk], n: int, min_chars: int, seed: int) -> list[Chunk]:
    """Amostra estratificada por documento, para não concentrar tudo no maior arquivo."""
    eligible = [c for c in chunks if c.article and len(c.text) >= min_chars]
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for c in eligible:
        by_source[c.source].append(c)
    rng = random.Random(seed)
    for items in by_source.values():
        rng.shuffle(items)
    sampled: list[Chunk] = []
    while len(sampled) < n and any(by_source.values()):
        for source in sorted(by_source):
            if by_source[source] and len(sampled) < n:
                sampled.append(by_source[source].pop())
    return sampled


def generate_question(llm: LLM, chunk: Chunk) -> str | None:
    completion = llm.complete(GENERATOR_SYSTEM, f"Trecho ({chunk.context}):\n{chunk.text}")
    try:
        question = json.loads(completion.text)["question"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
    return None if not question or question.upper() == "SKIP" else question


def build_dataset(llm: LLM, chunks: list[Chunk], n: int, min_chars: int = 200, seed: int = 42):
    rows = []
    for chunk in sample_chunks(chunks, n, min_chars, seed):
        question = generate_question(llm, chunk)
        if question:
            rows.append(
                {
                    "question": question,
                    "relevant": [{"source": chunk.source, "article": chunk.article}],
                    "chunk_id": chunk.id,
                    "synthetic": True,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    s = get_settings()
    chunks = build_vector_store(s).all_chunks()
    rows = build_dataset(build_judge(s), chunks, args.n, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} perguntas geradas a partir de {len(chunks)} chunks -> {args.output}")


if __name__ == "__main__":
    main()
