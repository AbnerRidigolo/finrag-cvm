import json

import pytest

from finrag.agent import FinRAGAgent
from finrag.eval.answer_eval import (
    RETRY_DELAY_S,
    CostBudget,
    load_results,
    run,
)
from finrag.llm import Completion, ExtractiveLLM

OK = '{"fidelidade": 5, "relevancia": 5}'
PERGUNTAS = [{"question": f"O que é linha d'água? ({i})"} for i in range(5)]


class FakeAPIError(Exception):
    """Imita um erro do SDK: tem status_code, como openai.BadRequestError."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


GENERICO = FakeAPIError(400, "Error code: 400 - something went wrong reading your request")
OUTRO_400 = FakeAPIError(400, "Error code: 400 - context_length_exceeded")


class Juiz:
    name = model = "juiz"

    def __init__(self, respostas):
        self.respostas = list(respostas)
        self.chamadas = 0

    def complete(self, system, prompt):
        self.chamadas += 1
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return Completion(resposta, "juiz", "juiz")


class Agente(FinRAGAgent):
    def __init__(self, retriever):
        super().__init__(ExtractiveLLM(), retriever)
        self.perguntas = 0

    def ask(self, question):
        self.perguntas += 1
        return super().ask(question)


def test_400_generico_tem_uma_nova_tentativa(retriever):
    esperas = []
    juiz = Juiz([GENERICO, OK])
    results = run(Agente(retriever), juiz, PERGUNTAS[:1], sleep=esperas.append)
    assert juiz.chamadas == 2 and esperas == [RETRY_DELAY_S]
    assert results[0]["fidelidade"] == 5 and not results[0].get("erro_juiz")


def test_400_generico_que_se_repete_vira_falha(retriever, tmp_path):
    juiz = Juiz([GENERICO, GENERICO])
    results = run(Agente(retriever), juiz, PERGUNTAS[:1], failed_dir=tmp_path, sleep=lambda s: None)
    assert juiz.chamadas == 2  # uma nova tentativa só
    assert results[0]["erro_juiz"] and results[0]["fidelidade"] is None


def test_outro_400_falha_direto_e_salva_o_prompt(retriever, tmp_path):
    esperas = []
    juiz = Juiz([OUTRO_400])
    results = run(Agente(retriever), juiz, PERGUNTAS[:1], failed_dir=tmp_path, sleep=esperas.append)
    assert juiz.chamadas == 1 and esperas == []  # sem nova tentativa
    assert results[0]["erro_juiz"] and results[0]["custo_juiz_usd"] == 0.0
    salvo = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert salvo["question"] == PERGUNTAS[0]["question"]
    assert "CONTEXTO:" in salvo["prompt"] and "context_length_exceeded" in salvo["error"]


def test_tres_falhas_seguidas_interrompem_a_rodada(retriever):
    agente = Agente(retriever)
    juiz = Juiz([OUTRO_400] * 5)
    budget = CostBudget(None)
    results = run(agente, juiz, PERGUNTAS, budget=budget)
    assert len(results) == 3 and agente.perguntas == 3  # não paga as 2 respostas restantes
    assert budget.stop_reason == "falhas_do_juiz"


def test_falha_isolada_nao_interrompe(retriever):
    juiz = Juiz([OUTRO_400, OUTRO_400, OK, OUTRO_400, OK])
    results = run(Agente(retriever), juiz, PERGUNTAS)
    assert len(results) == 5
    assert [bool(r.get("erro_juiz")) for r in results] == [True, True, False, True, False]


def test_erro_que_nao_vem_da_api_nao_e_engolido(retriever):
    with pytest.raises(ValueError):
        run(Agente(retriever), Juiz([ValueError("bug no código")]), PERGUNTAS[:1])


def test_gravacao_incremental_e_retomada_sem_as_falhas(retriever, tmp_path):
    output = tmp_path / "answer_eval.jsonl"
    with open(output, "w", encoding="utf-8") as f:
        run(
            Agente(retriever),
            Juiz([OK, OUTRO_400, OK]),
            PERGUNTAS[:3],
            on_result=lambda r: f.write(json.dumps(r, ensure_ascii=False) + "\n"),
        )
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3
    retomadas = load_results(output)
    # A pergunta em que o juiz falhou não conta como feita: será refeita na retomada.
    assert [r["question"] for r in retomadas] == [
        PERGUNTAS[0]["question"],
        PERGUNTAS[2]["question"],
    ]


def test_teto_considera_o_gasto_de_execucoes_anteriores():
    budget = CostBudget(0.01, spent=0.009, steps=3)  # média de 0,003 por pergunta
    assert not budget.can_afford_next() and budget.stop_reason == "custo"
