from finrag.eval.answer_eval import CostBudget
from finrag.eval.judge_control import (
    INVENTED_CLAIMS,
    add_invented_claim,
    build_cases,
    fit_to_budget,
    perturb_numbers,
    run_control,
    summarize_control,
    swap_answers,
)
from finrag.llm import Completion


def test_numeros_sao_dobrados_sem_tocar_nas_citacoes():
    resposta = "A taxa é de 1,2% ao ano e o prazo é de 180 dias [1]. Multa de R$ 10.000,00 [2]."
    assert perturb_numbers(resposta) == (
        "A taxa é de 2,4% ao ano e o prazo é de 360 dias [1]. Multa de R$ 20000,00 [2]."
    )


def test_resposta_sem_numeros_fica_de_fora():
    assert perturb_numbers("O administrador deve comunicar os cotistas [1][3].") is None


def test_troca_de_respostas_em_circulo():
    rows = [{"answer": "a"}, {"answer": "b"}, {"answer": "c"}]
    assert swap_answers(rows) == ["b", "c", "a"]


def test_frase_inventada_nunca_aparece_no_contexto():
    primeira, marcador = INVENTED_CLAIMS[0]
    # O contexto já fala em 2,5%: a primeira frase é pulada e a segunda é usada.
    contexto = f"A taxa máxima é de {marcador} ao ano."
    resposta, frase = add_invented_claim("Resposta real [1].", contexto)
    assert frase == INVENTED_CLAIMS[1][0] and resposta == f"Resposta real [1]. {frase}"
    todos = " ".join(m for _, m in INVENTED_CLAIMS)
    assert add_invented_claim("Resposta [1].", todos) is None


def test_amostra_reduzida_para_caber_no_teto():
    rows = [{"question": f"q{i}", "answer": f"Resposta {i} [1].", "context": ""} for i in range(10)]
    # Cada resposta gera 3 casos (tem número): 10 x 3 x 0,004 = 0,12 cabe; com teto 0,1, 8.
    assert fit_to_budget(rows, 0.004, 0.12)[0] == 10
    n, cases = fit_to_budget(rows, 0.004, 0.1)
    assert n == 8 and len(cases) * 0.004 <= 0.1
    assert fit_to_budget(rows, 0.004, 0.001) == (0, [])


def _rows():
    return [
        {
            "question": "Qual o prazo no FII?",
            "answer": "O prazo é de 180 dias [1].",
            "context": "c1",
        },
        {
            "question": "Quem convoca no FIDC?",
            "answer": "O administrador convoca [2].",
            "context": "c2",
        },
    ]


def test_casos_por_perturbacao():
    cases = build_cases(_rows())
    tipos = [c["perturbation"] for c in cases]
    assert tipos.count("numeros") == 1  # só a resposta com número
    assert tipos.count("troca") == 2 and tipos.count("inventada") == 2


class Juiz:
    name = model = "juiz"

    def complete(self, system, prompt):
        # Nota baixa quando a resposta tem a frase inventada ou números dobrados.
        ruim = "2,5%" in prompt or "3 dias" in prompt or "360 dias" in prompt
        nota = 2 if ruim else 5
        return Completion(f'{{"fidelidade": {nota}, "relevancia": 5}}', "juiz", "juiz", 1000, 100)


def test_deteccao_e_custo_por_perturbacao():
    results = run_control(Juiz(), build_cases(_rows()), {"juiz": [2.0, 10.0]})
    summary = summarize_control(results)
    assert summary["numeros"]["deteccao"] == 1.0
    assert summary["inventada"]["deteccao"] == 1.0
    assert summary["troca"]["deteccao"] == 0.0  # juiz de teste não olha relevância
    assert summary["custo_usd"] == round(5 * 0.003, 6)


def test_teto_interrompe_o_controle():
    budget = CostBudget(0.007)
    results = run_control(Juiz(), build_cases(_rows()), {"juiz": [2.0, 10.0]}, budget)
    assert len(results) == 2 and budget.stop_reason == "custo"
