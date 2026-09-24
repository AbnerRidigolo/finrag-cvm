from fastapi.testclient import TestClient

from finrag.agent import FinRAGAgent, heuristic_route
from finrag.api import create_app
from finrag.llm import ExtractiveLLM


class FakeLLM:
    name = "fake"

    def __init__(self, route_json: str) -> None:
        self.route_json = route_json
        self.prompts: list[str] = []

    def complete(self, system: str, prompt: str) -> str:
        if "classifica" in system:
            return self.route_json
        self.prompts.append(prompt)
        return "Resposta com citação [1]."


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
