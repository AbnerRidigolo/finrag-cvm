"""Controle negativo do juiz: respostas deliberadamente erradas devem receber nota baixa.

Se todas as respostas reais levam nota 5, falta saber se o juiz é capaz de dar nota baixa.
Este módulo pega respostas já avaliadas pelo answer_eval e aplica três perturbações
determinísticas, sem LLM, mantendo a pergunta e o contexto originais:

1. numeros: troca cada número da resposta (fora das citações [n]) pelo dobro. Só entram as
   respostas que têm números. Deve derrubar a fidelidade.
2. troca: a resposta de outra pergunta da amostra. Deve derrubar a relevância.
3. inventada: acrescenta uma frase plausível de uma lista fixa, conferindo antes que ela
   não aparece no contexto. Deve derrubar a fidelidade.

Taxa de detecção: fidelidade <= 3 nas perturbações 1 e 3; relevância <= 3 na 2.

Uso:
    python -m finrag.eval.judge_control data/eval/results/answer_eval.jsonl --max-cost 0.12
"""

import argparse
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from finrag.config import get_settings, load_api_keys
from finrag.eval.answer_eval import CostBudget, JudgeError, judge_answer
from finrag.llm import LLM, build_judge
from finrag.schemas import Answer

PERTURBATIONS = ("numeros", "troca", "inventada")
DETECTION = {"numeros": "fidelidade", "troca": "relevancia", "inventada": "fidelidade"}
DETECTION_MAX_SCORE = 3

# Frases plausíveis para fundos, com um marcador distintivo cada. Uma frase só é usada se
# nem ela nem o marcador aparecem no contexto daquela resposta.
INVENTED_CLAIMS = (
    ("Além disso, a taxa de administração não pode ultrapassar 2,5% ao ano.", "2,5%"),
    ("O prazo para essa providência é de 3 dias úteis, sob pena de multa diária.", "3 dias"),
    ("Essa exigência é dispensada para classes com patrimônio inferior a R$ 5 milhões.", "5 milh"),
    ("Essa regra deixou de valer para os fundos constituídos a partir de 2025.", "2025"),
)

NUMBER_RE = re.compile(r"(?<![\[\w])(\d+(?:[.,]\d+)*)(?![\]\w])")
CITATION_RE = re.compile(r"\[\d+\]")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def perturb_numbers(answer: str) -> str | None:
    """Dobra cada número da resposta, sem tocar nas citações [n]. None se não há número."""
    # Cada citação vira um marcador sem dígitos e volta no fim, intacta.
    citations = CITATION_RE.findall(answer)
    protected = CITATION_RE.sub("\x00", answer)
    if not NUMBER_RE.search(protected):
        return None

    def double(match: re.Match) -> str:
        # Formato brasileiro: ponto de milhar, vírgula decimal ("10.000,00", "1,2").
        raw = match.group(1)
        integer, _, decimals = raw.replace(".", "").partition(",")
        value = (int(integer) + (int(decimals) / 10 ** len(decimals) if decimals else 0)) * 2
        return f"{value:.{len(decimals)}f}".replace(".", ",") if decimals else str(int(value))

    changed = NUMBER_RE.sub(double, protected)
    for citation in citations:
        changed = changed.replace("\x00", citation, 1)
    return changed


def swap_answers(rows: list[dict]) -> list[str]:
    """A resposta de cada pergunta passa a ser a da pergunta seguinte (em círculo)."""
    return [rows[(i + 1) % len(rows)]["answer"] for i in range(len(rows))]


def add_invented_claim(answer: str, context: str, start: int = 0) -> tuple[str, str] | None:
    """Acrescenta a primeira frase da lista (a partir de `start`) ausente do contexto."""
    folded = _fold(context)
    for offset in range(len(INVENTED_CLAIMS)):
        claim, marker = INVENTED_CLAIMS[(start + offset) % len(INVENTED_CLAIMS)]
        if _fold(marker) not in folded and _fold(claim) not in folded:
            return f"{answer.rstrip()} {claim}", claim
    return None


def build_cases(sample: list[dict]) -> list[dict]:
    cases = []
    swapped = swap_answers(sample)
    for i, row in enumerate(sample):
        numbers = perturb_numbers(row["answer"])
        if numbers is not None:
            cases.append({"perturbation": "numeros", "source": row, "answer": numbers})
        cases.append({"perturbation": "troca", "source": row, "answer": swapped[i]})
        invented = add_invented_claim(row["answer"], row.get("context", ""), start=i)
        if invented is not None:
            cases.append(
                {
                    "perturbation": "inventada",
                    "source": row,
                    "answer": invented[0],
                    "claim": invented[1],
                }
            )
    return cases


def fit_to_budget(sample: list[dict], cost_per_call: float, limit: float) -> tuple[int, list[dict]]:
    """Maior prefixo da amostra cujos casos cabem no teto: reduz a amostra, nunca o teto.
    Conta os casos reais, porque só as respostas com números entram na perturbação 1."""
    for n in range(len(sample), 0, -1):
        cases = build_cases(sample[:n])
        if len(cases) * cost_per_call <= limit:
            return n, cases
    return 0, []


def run_control(
    judge: LLM,
    cases: list[dict],
    prices: dict[str, list[float]] | None = None,
    budget: CostBudget | None = None,
) -> list[dict]:
    budget = budget or CostBudget(None)
    results = []
    for case in cases:
        if not budget.can_afford_next():
            break
        src = case["source"]
        answer = Answer(
            question=src["question"],
            answer=case["answer"],
            route=src.get("route", ""),
            context=src.get("context", ""),
        )
        try:
            verdict = judge_answer(judge, answer, prices)
        except JudgeError as error:
            verdict = {
                "fidelidade": None,
                "relevancia": None,
                "custo_juiz_usd": 0.0,
                "justificativa": f"erro do juiz: {error}",
            }
        budget.add(verdict["custo_juiz_usd"])
        results.append(
            {
                "perturbation": case["perturbation"],
                "question": src["question"],
                "answer": case["answer"],
                "claim": case.get("claim"),
                "fidelidade": verdict["fidelidade"],
                "relevancia": verdict["relevancia"],
                "justificativa": verdict["justificativa"],
                "custo_juiz_usd": verdict["custo_juiz_usd"],
            }
        )
    return results


def summarize_control(results: list[dict]) -> dict:
    summary = {}
    for name in PERTURBATIONS:
        rows = [r for r in results if r["perturbation"] == name and r["fidelidade"] is not None]
        metric = DETECTION[name]
        summary[name] = {
            "n": len(rows),
            "fidelidade": dict(sorted(Counter(r["fidelidade"] for r in rows).items())),
            "relevancia": dict(sorted(Counter(r["relevancia"] for r in rows).items())),
            "criterio": f"{metric} <= {DETECTION_MAX_SCORE}",
            "deteccao": (
                sum(r[metric] <= DETECTION_MAX_SCORE for r in rows) / len(rows) if rows else None
            ),
        }
    summary["custo_usd"] = round(sum(r["custo_juiz_usd"] for r in results), 6)
    return summary


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Saída do answer_eval (JSONL)")
    parser.add_argument("-n", type=int, default=10, help="Respostas por perturbação (máximo)")
    parser.add_argument("--max-cost", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/eval/results/judge_control.jsonl")
    args = parser.parse_args()

    s = get_settings()
    lines = Path(args.results).read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    valid = [r for r in rows if r.get("fidelidade") is not None]
    # Custo médio real de um julgamento nesta avaliação, para dimensionar a amostra.
    cost_per_call = statistics.mean(r["custo_juiz_usd"] for r in valid)
    sample = random.Random(args.seed).sample(valid, min(args.n, len(valid)))
    n, cases = fit_to_budget(sample, cost_per_call, args.max_cost)
    counts = Counter(c["perturbation"] for c in cases)
    print(
        f"{n} respostas por perturbação (custo médio real por julgamento US$ {cost_per_call:.5f}, "
        f"teto US$ {args.max_cost:.2f}); casos: {dict(counts)}; estimativa "
        f"US$ {len(cases) * cost_per_call:.4f}"
    )

    results = run_control(build_judge(s), cases, s.llm_prices, CostBudget(args.max_cost))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8"
    )
    print(json.dumps(summarize_control(results), ensure_ascii=False, indent=2))
    print(f"\nDetalhes em {args.output}")


if __name__ == "__main__":
    main()
