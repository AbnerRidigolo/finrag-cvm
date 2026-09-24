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
import logging
import random
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping
from pathlib import Path

from finrag.config import get_settings, load_api_keys
from finrag.factory import build_vector_store
from finrag.ingestion.loaders import load_doc_titles
from finrag.llm import LLM, Completion, build_judge, strip_code_fence
from finrag.metrics import cost_usd
from finrag.schemas import Chunk

logger = logging.getLogger(__name__)

EXCLUDED_CHUNKS = Path("data/eval/excluded_chunks.txt")

GENERATOR_SYSTEM = """Você cria perguntas de avaliação para um sistema de busca sobre
normas e regulamentos de fundos de investimento brasileiros.
Você recebe o assunto do documento (o fundo ou o tipo de fundo de que ele trata) e um trecho
dele. Escreva UMA pergunta em português, como a dúvida real de um analista do mercado, que:
- seja respondida por este trecho, sem depender de outros trechos;
- seja autossuficiente: cite o assunto exatamente como informado, para que a pergunta
  deixe claro de qual fundo ou tipo de fundo está falando;
- não mencione a estrutura do documento: nada de número de anexo, artigo, inciso,
  parágrafo ou alínea, nem "este regulamento", "o trecho", "o texto" ou "acima";
- parafraseie com suas próprias palavras, em vez de copiar frases do trecho.
Nunca invente nomes de fundos, administradores ou gestores.
Se o trecho não tiver conteúdo substantivo (índice, cabeçalho, assinatura, endereço),
responda SKIP.
Responda somente com JSON: {"question": "..."} ou {"question": "SKIP"}."""

# Uma pergunta que repete 6 ou mais palavras seguidas do trecho favorece o BM25 e não
# representa como um usuário real pergunta; ela é gravada, mas marcada para revisão.
COPY_MIN_WORDS = 6


def _words(text: str) -> list[str]:
    """Palavras em minúsculas e sem acentos, para comparar pergunta e trecho."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return re.findall(r"\w+", "".join(ch for ch in decomposed if not unicodedata.combining(ch)))


def longest_shared_run(question: str, text: str) -> int:
    """Maior número de palavras consecutivas que a pergunta repete do trecho."""
    q, t = _words(question), _words(text)
    best, prev = 0, [0] * (len(t) + 1)
    for qw in q:
        cur = [0] * (len(t) + 1)
        for j, tw in enumerate(t, 1):
            if qw == tw:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


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


def generate_question(llm: LLM, chunk: Chunk, subject: str) -> str | None:
    # Só o assunto curto, não o título completo: o título tem "Anexo Normativo II", e a
    # pergunta não deve citar a estrutura do documento.
    prompt = f"Assunto: {subject}\n\nTrecho:\n{chunk.text}"
    completion = llm.complete(GENERATOR_SYSTEM, prompt)
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
    subjects: Mapping[str, str] | None = None,
) -> list[dict]:
    """Gera as perguntas. `skip_ids` pula chunks já processados; `on_row` recebe cada linha
    assim que ela é gerada, para gravação incremental; `subjects` mapeia o arquivo para o
    assunto que a pergunta deve citar (sem ele, usa o título do chunk)."""
    rows = []
    for chunk in sample_chunks(chunks, n, min_chars, seed):
        if chunk.id in skip_ids:
            continue
        subject = (subjects or {}).get(chunk.source) or chunk.context
        question = generate_question(llm, chunk, subject)
        if question:
            row = {
                "question": question,
                "relevant": [{"source": chunk.source, "article": chunk.article}],
                "chunk_id": chunk.id,
                "subject": subject,
                "synthetic": True,
                "copied": longest_shared_run(question, chunk.text) >= COPY_MIN_WORDS,
            }
            if row["copied"]:
                logger.warning("Pergunta copiada do trecho (%s): %s", chunk.id, question)
            rows.append(row)
            if on_row:
                on_row(row)
    return rows


def existing_chunk_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["chunk_id"] for line in f if line.strip()}


def excluded_chunk_ids(path: Path) -> set[str]:
    """Chunks cujas perguntas foram removidas na revisão manual: um chunk_id por linha,
    opcionalmente seguido do motivo em comentário ("id  # motivo"); linhas vazias e
    comentários são ignorados. Sem isso, rodar de novo recriaria a pergunta removida."""
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {s for line in f if (s := line.split("#", 1)[0].strip())}


class UsageMeter:
    """Envolve um LLM e soma os tokens de todas as chamadas, para informar o custo real."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.name, self.model = llm.name, llm.model
        self.calls = self.input_tokens = self.output_tokens = 0

    def complete(self, system: str, prompt: str) -> Completion:
        completion = self.llm.complete(system, prompt)
        self.calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        return completion


def build_dataset_file(
    llm: LLM,
    chunks: list[Chunk],
    n: int,
    output: str | Path,
    seed: int = 42,
    min_chars: int = 200,
    excluded: Collection[str] = (),
    subjects: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Grava cada pergunta assim que ela é gerada e retoma de onde parou.

    Uma falha no meio (saldo, rede, 429) não perde as perguntas já pagas: ao rodar de novo
    com a mesma semente, a amostra é a mesma e os chunks que já estão no arquivo, ou na
    lista de exclusão da revisão manual, são pulados. Devolve (novas, já existentes).
    """
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_chunk_ids(path)
    with open(path, "a", encoding="utf-8") as f:

        def write(row: dict) -> None:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

        rows = build_dataset(
            llm,
            chunks,
            n,
            min_chars,
            seed,
            skip_ids=done | set(excluded),
            on_row=write,
            subjects=subjects,
        )
    return len(rows), len(done)


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", default=str(EXCLUDED_CHUNKS))
    args = parser.parse_args()

    s = get_settings()
    chunks = build_vector_store(s).all_chunks()
    excluded = excluded_chunk_ids(Path(args.exclude))
    subjects = {k: v["subject"] for k, v in load_doc_titles(s.doc_titles_path).items()}
    missing = sorted({c.source for c in chunks} - subjects.keys())
    if missing:
        logger.warning("Sem assunto no mapa de títulos (usa o título do chunk): %s", missing)
    meter = UsageMeter(build_judge(s))
    new, existing = build_dataset_file(
        meter, chunks, args.n, args.output, args.seed, excluded=excluded, subjects=subjects
    )
    cost = cost_usd(meter.model, meter.input_tokens, meter.output_tokens, s.llm_prices)
    print(
        f"{new} perguntas novas ({existing} já existiam, {len(excluded)} chunks excluídos) "
        f"a partir de {len(chunks)} chunks -> {args.output}\n"
        f"{meter.calls} chamadas a {meter.name}/{meter.model}: "
        f"{meter.input_tokens} tokens de entrada + {meter.output_tokens} de saída = "
        + (f"US$ {cost:.4f}" if cost is not None else "custo desconhecido (modelo sem preço)")
    )


if __name__ == "__main__":
    main()
