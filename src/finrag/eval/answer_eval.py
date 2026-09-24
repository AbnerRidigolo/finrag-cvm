"""Avaliação da qualidade das respostas com LLM como juiz.

Para cada pergunta do dataset, o agente responde e um segundo LLM (o juiz) avalia:
- Fidelidade (1-5): cada afirmação da resposta é sustentada pelo contexto recuperado?
  Mede alucinação. Uma resposta que diz "o contexto não informa" é fiel.
- Relevância (1-5): a resposta atende ao que foi perguntado?

Boas práticas aplicadas: o juiz recebe critérios explícitos com âncoras por nota, justifica
antes de dar a nota, roda com temperatura zero e, de preferência, é um modelo diferente do
que gerou as respostas (JUDGE_PROVIDER / JUDGE_MODEL), para reduzir viés de autoavaliação.

Uso:
    python -m finrag.eval.answer_eval data/eval/cvm_questions.jsonl -n 30
"""

import argparse
import json
import statistics
from pathlib import Path

from finrag.agent import FinRAGAgent
from finrag.config import get_settings, load_api_keys
from finrag.eval.retrieval_eval import load_dataset
from finrag.factory import build_agent
from finrag.llm import LLM, build_judge, strip_code_fence
from finrag.metrics import cost_usd
from finrag.schemas import Answer

JUDGE_SYSTEM = """Você avalia respostas de um assistente sobre fundos de investimento.
Você recebe a PERGUNTA, o CONTEXTO que o assistente recebeu e a RESPOSTA.

Fidelidade (use somente o CONTEXTO como verdade, ignore o que você sabe):
5 = toda afirmação está sustentada pelo contexto, ou a resposta diz corretamente
    que o contexto não traz a informação
4 = sustentada, com um detalhe menor não verificável no contexto
3 = mistura afirmações sustentadas com ao menos uma não sustentada relevante
2 = a maior parte não está no contexto
1 = contradiz o contexto ou inventa a resposta

Relevância:
5 = responde diretamente e por completo ao que foi perguntado
3 = responde em parte ou com rodeios
1 = não responde à pergunta

Escreva a justificativa antes das notas. Responda somente com JSON:
{"justificativa": "...", "fidelidade": n, "relevancia": n}"""


def judge_answer(judge: LLM, answer: Answer, prices: dict[str, list[float]] | None = None) -> dict:
    """Notas do juiz para uma resposta, com os tokens e o custo da própria chamada ao juiz.

    O custo é contado mesmo quando a saída é inválida, porque a chamada foi paga.
    """
    prompt = (
        f"PERGUNTA:\n{answer.question}\n\n"
        f"CONTEXTO:\n{answer.context or '(vazio)'}\n\n"
        f"RESPOSTA:\n{answer.answer}"
    )
    completion = judge.complete(JUDGE_SYSTEM, prompt)
    usage = {
        "judge_input_tokens": completion.input_tokens,
        "judge_output_tokens": completion.output_tokens,
        "custo_juiz_usd": cost_usd(
            completion.model, completion.input_tokens, completion.output_tokens, prices or {}
        )
        or 0.0,
    }
    text = strip_code_fence(completion.text)
    try:
        parsed = json.loads(text)
        verdict = {
            "fidelidade": int(parsed["fidelidade"]),
            "relevancia": int(parsed["relevancia"]),
            "justificativa": str(parsed.get("justificativa", "")),
        }
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        verdict = {
            "fidelidade": None,
            "relevancia": None,
            "justificativa": f"inválido: {text[:200]}",
        }
    return {**verdict, **usage}


def run(
    agent: FinRAGAgent,
    judge: LLM,
    dataset: list[dict],
    prices: dict[str, list[float]] | None = None,
) -> list[dict]:
    results = []
    for row in dataset:
        answer = agent.ask(row["question"])
        verdict = judge_answer(judge, answer, prices)
        results.append(
            {
                "question": row["question"],
                "answer": answer.answer,
                "sources": [f"{s.source} {s.article}" for s in answer.sources],
                **verdict,
                "latency_ms": answer.usage.latency_ms,
                "custo_agente_usd": answer.usage.cost_usd,
            }
        )
    return results


def summarize(results: list[dict]) -> dict:
    """Médias de qualidade e latência, e custo total da rodada separado entre agente e juiz.

    O custo do agente é o que uma pergunta custa em produção; o do juiz é custo só da avaliação.
    """
    if not results:
        return {"n": 0, "validas": 0}
    custo_agente = sum(r["custo_agente_usd"] for r in results)
    custo_juiz = sum(r["custo_juiz_usd"] for r in results)
    costs = {
        "custo_agente_usd": round(custo_agente, 6),
        "custo_juiz_usd": round(custo_juiz, 6),
        "custo_total_usd": round(custo_agente + custo_juiz, 6),
        "custo_medio_agente_usd": round(custo_agente / len(results), 6),
    }
    valid = [r for r in results if r["fidelidade"] is not None]
    if not valid:
        return {"n": len(results), "validas": 0, **costs}
    return {
        "n": len(results),
        "validas": len(valid),
        "fidelidade_media": statistics.mean(r["fidelidade"] for r in valid),
        "relevancia_media": statistics.mean(r["relevancia"] for r in valid),
        "pct_fieis": sum(r["fidelidade"] >= 4 for r in valid) / len(valid),
        "latencia_media_ms": statistics.mean(r["latency_ms"] for r in results),
        **costs,
    }


def main() -> None:
    load_api_keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("-n", type=int, default=0, help="Limita o número de perguntas")
    parser.add_argument("--output", default="data/eval/results/answer_eval.jsonl")
    args = parser.parse_args()

    s = get_settings()
    dataset = load_dataset(args.dataset)
    if args.n:
        dataset = dataset[: args.n]
    judge_llm = build_judge(s)
    for model in (s.llm_model, judge_llm.model):
        if model not in s.llm_prices:
            print(f"Aviso: {model} sem preço em LLM_PRICES; o custo dele sairá como zero.")
    results = run(build_agent(s), judge_llm, dataset, s.llm_prices)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = summarize(results)
    judge = f"{s.judge_provider or s.llm_provider}/{s.judge_model or s.llm_model}"
    print(f"Gerador: {s.llm_provider}/{s.llm_model} | Juiz: {judge}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    worst = sorted(
        (r for r in results if r["fidelidade"] is not None), key=lambda r: r["fidelidade"]
    )
    if worst and worst[0]["fidelidade"] < 4:
        print("\nPiores casos (revise à mão):")
        for r in worst[:3]:
            print(f"- [{r['fidelidade']}] {r['question']}\n  {r['justificativa']}")
    print(f"\nDetalhes por pergunta em {args.output}")


if __name__ == "__main__":
    main()
