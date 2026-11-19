"""Exercita casos escritos à mão e mostra a cobertura das evidências e a resposta."""

import argparse

from finrag.config import load_api_keys
from finrag.eval.retrieval_eval import load_dataset
from finrag.factory import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    load_api_keys()
    agent = build_agent()
    covered = answerable = 0
    for case in cases:
        question = case["question"]
        answer = agent.ask(question)
        expected = {(item["source"], item["article"]) for item in case["relevant"]}
        found = {(source.source, source.article) for source in answer.sources}
        if expected:
            covered += expected <= found
            answerable += 1
            evidence = f"{len(expected & found)}/{len(expected)} artigos esperados"
            missing = ", ".join(
                f"{source} ({article})" for source, article in sorted(expected - found)
            )
            if missing:
                evidence += f" · faltaram: {missing}"
        else:
            evidence = "evidência insuficiente no corpus · revisar a abstenção na resposta"

        print(f"\n{case['id']} · {case['category']} [{evidence}]")
        print(f"  Pergunta: {question}")
        print(f"  Gabarito: {case['expected']}")
        print(f"  Agente: {answer.answer}")

    print(
        f"\nCobertura completa das evidências no contexto: {covered}/{answerable} "
        f"casos respondíveis (top-{agent.top_k}). "
        f"Casos sem evidência para revisar à parte: {len(cases) - answerable}."
    )


if __name__ == "__main__":
    main()
