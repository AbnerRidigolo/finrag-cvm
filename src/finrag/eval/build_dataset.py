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
from collections.abc import Callable, Collection
from pathlib import Path

from finrag.config import get_settings, load_api_keys
from finrag.factory import build_vector_store
from finrag.llm import LLM, build_judge, strip_code_fence
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
        question = json.loads(strip_code_fence(completion.text))["question"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
    return None if not question or question.upper() == "SKIP" else question


def build_dataset(
    llm: LLM,
    chunks: list[Chunk],
    n: int,
    min_chars: int = 200,
    seed: int = 42,
    skip_ids: Collection[str] = (),
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Gera as perguntas. `skip_ids` pula chunks já processados; `on_row` recebe cada linha
    assim que ela é gerada, para gravação incremental."""
    rows = []
    for chunk in sample_chunks(chunks, n, min_chars, seed):
        if chunk.id in skip_ids:
            continue
        question = generate_question(llm, chunk)
        if question:
            row = {
                "question": question,
                "relevant": [{"source": chunk.source, "article": chunk.article}],
                "chunk_id": chunk.id,
                "synthetic": True,
            }
            rows.append(row)
            if on_row:
                on_row(row)
    return rows


def existing_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f if line.strip()}


def build_dataset_file(
    llm: LLM,
    chunks: list[Chunk],
    n: int,
    output: str | Path,
    seed: int = 42,
    min_chars: int = 200,
) -> tuple[int, int]:
    """Grava cada pergunta assim que ela é gerada e retoma de onde parou.

    Uma falha no meio (saldo, rede, 429) não perde as perguntas já pagas: ao rodar de novo
    com a mesma semente, a amostra é a mesma e os chunks que já estão no arquivo são pulados.
    Devolve (novas, já existentes).
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_chunk_ids(path)
    with open(path, "a", encoding="utf-8") as f:

        def write(row: dict) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        rows = build_dataset(llm, chunks, n, min_chars, seed, skip_ids=done, on_row=write)
    return len(rows), len(done)


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    s = get_settings()
    chunks = build_vector_store(s).all_chunks()
    new, existing = build_dataset_file(build_judge(s), chunks, args.n, args.output, args.seed)
    print(
        f"{new} perguntas novas ({existing} já existiam) a partir de {len(chunks)} chunks "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
