"""Exercita casos escritos à mão e mostra a cobertura das evidências e a resposta."""

import argparse

from finrag.config import load_api_keys
from finrag.eval.retrieval_eval import load_dataset
from finrag.factory import build_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument(
        "--verify-abstention",
        action="store_true",
        help="Falha na ausência de abstenção esperada ou em abstenção indevida.",
    )
    args = parser.parse_args()

    cases = load_dataset(args.dataset)
    load_api_keys()
    agent = build_agent()
    covered = answerable = 0
    missed_abstentions = false_abstentions = unanswerable = 0
    for case in cases:
        question = case["question"]
        answer = agent.ask(question)
        expected = {(item["source"], item["article"]) for item in case["relevant"]}
        found = {(source.source, source.article) for source in answer.sources}
        if expected:
            covered += expected <= found
            answerable += 1
            false_abstentions += answer.abstained
            evidence = f"{len(expected & found)}/{len(expected)} artigos esperados"
            missing = ", ".join(
                f"{source} ({article})" for source, article in sorted(expected - found)
            )
            if missing:
                evidence += f" · faltaram: {missing}"
        else:
            unanswerable += 1
            missed_abstentions += not answer.abstained
            evidence = "evidência insuficiente no corpus"

        print(f"\n{case['id']} · {case['category']} [{evidence}]")
        print(f"  Pergunta: {question}")
        print(f"  Gabarito: {case['expected']}")
        print(f"  Agente: {answer.answer}")
        print(f"  Abstenção estruturada: {answer.abstention_reason or 'não'}")

    print(
        f"\nCobertura completa das evidências no contexto: {covered}/{answerable} "
        f"casos respondíveis (top-{agent.top_k})."
    )
    print(f"Abstenções corretas: {unanswerable - missed_abstentions}/{unanswerable}.")
    print(f"Abstenções indevidas: {false_abstentions}/{answerable}.")
    if args.verify_abstention and (
        not answerable or not unanswerable or missed_abstentions or false_abstentions
    ):
        raise SystemExit("A validação de abstenção falhou.")


if __name__ == "__main__":
    main()
