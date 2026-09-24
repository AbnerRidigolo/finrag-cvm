import json

import pytest

from finrag.agent import FinRAGAgent
from finrag.eval.answer_eval import CostBudget, judge_answer, rejudge, run, summarize
from finrag.eval.build_dataset import (
    UsageMeter,
    build_dataset,
    build_dataset_file,
    excluded_chunk_ids,
    longest_shared_run,
    sample_chunks,
)
from finrag.llm import Completion, ExtractiveLLM, strip_code_fence
from finrag.metrics import cost_usd
from finrag.schemas import Answer


class ScriptedLLM:
    name = "scripted"
    model = "scripted"

    def __init__(self, replies, tokens=(0, 0)):
        self.replies = list(replies)
        self.tokens = tokens

    def complete(self, system, prompt):
        return Completion(self.replies.pop(0), self.name, self.model, *self.tokens)


def test_remove_cerca_de_codigo():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('  ```\n{"a": 1}\n```  ') == '{"a": 1}'
    assert strip_code_fence('```JSON{"a": 1}```') == '{"a": 1}'
    assert strip_code_fence(' {"a": 1} ') == '{"a": 1}'
    assert strip_code_fence("") == ""


def test_custo_por_milhao_de_tokens():
    prices = {"m": [0.5, 2.0]}
    assert cost_usd("m", 1_000_000, 500_000, prices) == 1.5
    assert cost_usd("sem-preco", 10, 10, prices) is None


def test_juiz_aceita_json_com_cerca_de_codigo():
    judge = ScriptedLLM(['```json\n{"justificativa": "ok", "fidelidade": 5, "relevancia": 4}\n```'])
    verdict = judge_answer(judge, Answer(question="q", answer="a", route="normas"))
    assert (verdict["fidelidade"], verdict["relevancia"]) == (5, 4)


def test_juiz_com_saida_invalida_nao_quebra():
    verdict = judge_answer(ScriptedLLM(["nota 5"]), Answer(question="q", answer="a", route="n"))
    assert verdict["fidelidade"] is None


def test_avaliacao_de_respostas_ponta_a_ponta(retriever):
    agent = FinRAGAgent(ExtractiveLLM(), retriever)
    judge = ScriptedLLM(
        [
            '{"justificativa": "sustentada", "fidelidade": 5, "relevancia": 5}',
            '{"justificativa": "inventou", "fidelidade": 2, "relevancia": 4}',
        ]
    )
    dataset = [{"question": "O que é linha d'água?"}, {"question": "Qual o prazo do Beta?"}]
    results = run(agent, judge, dataset)
    summary = summarize(results)
    assert summary["fidelidade_media"] == 3.5 and summary["pct_fieis"] == 0.5
    assert all(r["sources"] for r in results)


def test_custo_do_juiz_separado_do_agente(retriever):
    agent = FinRAGAgent(ExtractiveLLM(), retriever)
    judge = ScriptedLLM(
        ['{"fidelidade": 5, "relevancia": 5}', "saída inválida"], tokens=(1000, 100)
    )
    prices = {"scripted": [2.0, 10.0]}
    dataset = [{"question": "O que é linha d'água?"}, {"question": "Qual o prazo do Beta?"}]
    summary = summarize(run(agent, judge, dataset, prices))
    # 2 chamadas x (1000 x 2 + 100 x 10) / 1e6; a chamada com saída inválida também é paga.
    assert summary["custo_juiz_usd"] == 0.006
    assert summary["custo_agente_usd"] == 0.0  # modo extrativo não chama LLM
    assert summary["custo_total_usd"] == 0.006
    assert summary["validas"] == 1


def test_juiz_sem_preco_custa_zero():
    judge = ScriptedLLM(['{"fidelidade": 5, "relevancia": 5}'], tokens=(1000, 100))
    verdict = judge_answer(judge, Answer(question="q", answer="a", route="n"), prices={})
    assert verdict["custo_juiz_usd"] == 0.0 and verdict["judge_input_tokens"] == 1000


def test_juiz_recebe_o_contexto_recuperado(retriever):
    agent = FinRAGAgent(ExtractiveLLM(), retriever)
    answer = agent.ask("O que é linha d'água?")
    assert "Linha d'água" in answer.context


def test_amostragem_estratificada_por_documento(chunks):
    sampled = sample_chunks(chunks, n=6, min_chars=50, seed=1)
    assert len({c.source for c in sampled}) == 3
    assert all(c.article for c in sampled)
    assert sample_chunks(chunks, 6, 50, seed=1) == sampled


def test_gera_dataset_e_descarta_skip(chunks):
    llm = ScriptedLLM(
        ['```json\n{"question": "Qual o prazo?"}\n```', '{"question": "SKIP"}', "lixo"]
    )
    rows = build_dataset(llm, chunks, n=3, min_chars=50)
    assert len(rows) == 1 and rows[0]["synthetic"]
    assert rows[0]["relevant"][0]["article"]


def test_dataset_gravado_a_cada_pergunta_e_retomado_apos_falha(chunks, tmp_path):
    output = tmp_path / "perguntas.jsonl"
    # A terceira chamada falha (a lista de respostas acaba), como uma queda no meio da rodada.
    falha = ScriptedLLM(['{"question": "P1?"}', '{"question": "P2?"}'])
    with pytest.raises(IndexError):
        build_dataset_file(falha, chunks, n=4, output=output, seed=1, min_chars=50)
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2  # já pagas, não perdidas

    retomada = ScriptedLLM(['{"question": "P3?"}', '{"question": "P4?"}'])
    new, existing = build_dataset_file(retomada, chunks, n=4, output=output, seed=1, min_chars=50)
    assert (new, existing) == (2, 2) and retomada.replies == []  # só os chunks que faltavam
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in rows] == ["P1?", "P2?", "P3?", "P4?"]
    assert len({r["chunk_id"] for r in rows}) == 4


def test_chunks_excluidos_na_revisao_nao_sao_regerados(chunks, tmp_path):
    output = tmp_path / "perguntas.jsonl"
    primeira = sample_chunks(chunks, n=3, min_chars=50, seed=1)[0]
    lista = tmp_path / "excluidos.txt"
    # O motivo fica ao lado do id, como em data/eval/excluded_chunks.txt.
    lista.write_text(
        f"# removidos na revisão\n\n{primeira.id}  # pergunta copiou o trecho\n", encoding="utf-8"
    )

    excluded = excluded_chunk_ids(lista)
    llm = UsageMeter(ScriptedLLM(['{"question": "P2?"}', '{"question": "P3?"}'], tokens=(10, 2)))
    new, _ = build_dataset_file(
        llm, chunks, n=3, output=output, seed=1, min_chars=50, excluded=excluded
    )

    ids = {json.loads(line)["chunk_id"] for line in output.read_text(encoding="utf-8").splitlines()}
    assert excluded == {primeira.id} and primeira.id not in ids and new == 2
    assert (llm.calls, llm.input_tokens, llm.output_tokens) == (2, 20, 4)
    assert excluded_chunk_ids(tmp_path / "nao-existe.txt") == set()


def test_pergunta_que_copia_6_palavras_do_trecho_e_marcada(chunks, caplog):
    trecho = "A taxa de administração é de 1,2% ao ano sobre o patrimônio líquido."
    assert longest_shared_run("Qual a TAXA DE ADMINISTRACAO E DE 1,2% ao ano?", trecho) >= 6
    assert longest_shared_run("Quanto custa a taxa de administração do fundo?", trecho) == 4

    chunk = next(c for c in chunks if c.article and len(c.text) >= 50)
    copia = " ".join(chunk.text.split()[:8])
    llm = ScriptedLLM([json.dumps({"question": copia}), '{"question": "Outra pergunta?"}'])
    with caplog.at_level("WARNING", logger="finrag"):
        rows = build_dataset(
            llm, [chunk, *[c for c in chunks if c is not chunk]], n=2, min_chars=50
        )
    marcadas = [r for r in rows if r["copied"]]
    assert len(marcadas) == 1 and marcadas[0]["question"] == copia
    assert "copiada" in caplog.text


PERGUNTAS = [{"question": "O que é linha d'água?"}, {"question": "Qual o prazo do Beta?"}] * 2


def test_teto_de_custo_para_antes_de_ultrapassar(retriever, caplog):
    agent = FinRAGAgent(ExtractiveLLM(), retriever)  # agente offline: custo zero
    judge = ScriptedLLM(['{"fidelidade": 5, "relevancia": 5}'] * 4, tokens=(1000, 100))
    budget = CostBudget(0.007)  # cada julgamento custa 0,003 com os preços abaixo
    with caplog.at_level("WARNING", logger="finrag"):
        results = run(agent, judge, PERGUNTAS, {"scripted": [2.0, 10.0]}, budget)
    # Após 2 perguntas (0,006), a 3ª levaria a 0,009 > 0,007: para antes de chamar.
    assert len(results) == 2 and budget.stopped and budget.spent <= 0.007
    assert len(judge.replies) == 2  # as chamadas não feitas não foram pagas
    assert "Teto de custo" in caplog.text


def test_sem_teto_avalia_tudo(retriever):
    judge = ScriptedLLM(['{"fidelidade": 5, "relevancia": 5}'] * 4)
    results = run(FinRAGAgent(ExtractiveLLM(), retriever), judge, PERGUNTAS)
    assert len(results) == 4 and all(r["context"] for r in results)


def test_rejulgamento_mede_a_concordancia(retriever):
    primeira = ScriptedLLM(
        [
            '{"fidelidade": 5, "relevancia": 5}',
            '{"fidelidade": 4, "relevancia": 3}',
            '{"fidelidade": 2, "relevancia": 5}',
            '{"fidelidade": 5, "relevancia": 4}',
        ]
    )
    results = run(FinRAGAgent(ExtractiveLLM(), retriever), primeira, PERGUNTAS)
    # Segunda rodada: nota 5 em todas. Concorda em 2 de 4; diferenças 0, 1, 3 e 0.
    segunda = ScriptedLLM([], tokens=(1000, 100))
    segunda.complete = lambda system, prompt: Completion(
        json.dumps({"fidelidade": 5, "relevancia": 5}), "scripted", "scripted", 1000, 100
    )
    summary = rejudge(segunda, results, n=4, prices={"scripted": [2.0, 10.0]})
    esperadas = [r["fidelidade"] for r in results]  # [5, 4, 2, 5] contra 5 em todas
    assert summary["n"] == 4
    assert summary["concordancia_fidelidade"] == sum(f == 5 for f in esperadas) / 4
    assert summary["dif_media_abs_fidelidade"] == sum(abs(5 - f) for f in esperadas) / 4
    assert summary["custo_usd"] == 0.012


def test_rejulgamento_usa_o_mesmo_contexto(retriever):
    results = run(
        FinRAGAgent(ExtractiveLLM(), retriever),
        ScriptedLLM(['{"fidelidade": 5, "relevancia": 5}']),
        PERGUNTAS[:1],
    )
    vistos = []

    class Juiz:
        name = model = "j"

        def complete(self, system, prompt):
            vistos.append(prompt)
            return Completion('{"fidelidade": 5, "relevancia": 5}', "j", "j")

    rejudge(Juiz(), results, n=1)
    assert results[0]["context"] and results[0]["context"] in vistos[0]
