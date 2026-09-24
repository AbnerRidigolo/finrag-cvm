from fastapi.testclient import TestClient

from finrag.agent import FinRAGAgent, heuristic_route
from finrag.api import create_app
from finrag.llm import Completion, ExtractiveLLM


class FakeLLM:
    name = "fake"
    model = "fake-model"

    def __init__(self, route_json: str) -> None:
        self.route_json = route_json
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> Completion:
        if "classifica" in system:
            return Completion(self.route_json, self.name, self.model, 100, 20)
        self.prompts.append(prompt)
        return Completion("Resposta com citação [1].", self.name, self.model, 1000, 50)


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def query(self, intent: str, entity: str):
        self.calls.append((intent, entity))
        return [{"gestor": "GESTORA ALFA CAPITAL LTDA", "fundo": "FUNDO EXEMPLO ALPHA FIDC"}]


def test_roteador_por_regras():
    assert heuristic_route("Qual a taxa de performance?")["route"] == "normas"
    r = heuristic_route("Quais fundos da gestora Alfa Capital?")
    assert r == {"route": "fundos", "intent": "fundos_do_gestor", "entity": "Alfa Capital"}
    r = heuristic_route("Quem é o gestor do Fundo Exemplo Alpha?")
    assert r["intent"] == "gestor_do_fundo" and r["entity"] == "Fundo Exemplo Alpha"


def test_agente_offline_responde_com_fontes(retriever):
    agent = FinRAGAgent(ExtractiveLLM(), retriever, top_k=3)
    answer = agent.ask("Qual é a taxa de administração do Fundo Exemplo Alpha?")
    assert answer.route == "normas"
    assert answer.sources and "[1]" in answer.answer


def test_agente_usa_grafo_quando_roteado(retriever):
    graph = FakeGraph()
    llm = FakeLLM('{"route": "fundos", "intent": "fundos_do_gestor", "entity": "Alfa"}')
    answer = FinRAGAgent(llm, retriever, graph=graph).ask("Quais fundos a Alfa gere?")
    assert answer.route == "fundos"
    assert graph.calls == [("fundos_do_gestor", "Alfa")]
    assert "GESTORA ALFA CAPITAL" in llm.prompts[0]


def test_roteador_aceita_json_com_cerca_de_codigo(retriever):
    graph = FakeGraph()
    llm = FakeLLM(
        '```json\n{"route": "fundos", "intent": "fundos_do_gestor", "entity": "Alfa"}\n```'
    )
    answer = FinRAGAgent(llm, retriever, graph=graph).ask("Me fale da Alfa")
    assert answer.route == "fundos" and graph.calls == [("fundos_do_gestor", "Alfa")]


def test_sem_grafo_cai_para_documentos(retriever):
    llm = FakeLLM('{"route": "fundos", "intent": "fundos_do_gestor", "entity": "Alfa"}')
    assert FinRAGAgent(llm, retriever, graph=None).ask("Fundos da Alfa?").route == "normas"


def test_json_invalido_do_llm_usa_regras(retriever):
    llm = FakeLLM("isto não é json")
    assert FinRAGAgent(llm, retriever).ask("O que é linha d'água?").route == "normas"


def test_api(retriever):
    client = TestClient(create_app(FinRAGAgent(ExtractiveLLM(), retriever)))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/ask", json={"question": "O que é linha d'água?"})
    assert response.status_code == 200
    assert response.json()["sources"]
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_uso_latencia_e_custo_por_pergunta(retriever):
    llm = FakeLLM('{"route": "normas", "intent": "", "entity": ""}')
    agent = FinRAGAgent(llm, retriever, prices={"fake-model": [1.0, 10.0]})
    usage = agent.ask("Qual a taxa de performance do Alpha?").usage
    assert usage.llm_calls == 2
    assert (usage.input_tokens, usage.output_tokens) == (1100, 70)
    assert usage.cost_usd == round((1100 * 1.0 + 70 * 10.0) / 1e6, 6)
    assert {"route", "retrieve", "generate"} <= set(usage.stage_latency_ms)
    assert usage.latency_ms >= sum(usage.stage_latency_ms.values()) * 0.9


def test_endpoint_de_metricas(retriever):
    client = TestClient(create_app(FinRAGAgent(ExtractiveLLM(), retriever)))
    client.post("/ask", json={"question": "O que é linha d'água?"})
    body = client.get("/metrics/").text
    assert "finrag_requests_total" in body
    assert "finrag_stage_latency_seconds_bucket" in body


def test_resposta_da_api_nao_expoe_contexto_interno(retriever):
    client = TestClient(create_app(FinRAGAgent(ExtractiveLLM(), retriever)))
    data = client.post("/ask", json={"question": "O que é linha d'água?"}).json()
    assert "context" not in data and "text" not in data["sources"][0]
    assert "latency_ms" in data["usage"]
