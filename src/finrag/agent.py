"""Agente em LangGraph: roteia a pergunta, busca o contexto certo e responde com citações.

    pergunta ──> route ──┬─> retrieve_docs ──┐
                         └─> query_graph  ───┴─> generate ──> resposta

- "normas": perguntas sobre regras e regulamentos -> recuperação híbrida nos documentos.
- "fundos": perguntas sobre fundos, gestores e administradores -> consulta ao grafo Neo4j.
"""

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from finrag.graph.store import QUERIES, GraphStore
from finrag.llm import LLM
from finrag.retrieval.hybrid import HybridRetriever
from finrag.schemas import Answer, RetrievedChunk, Source

ROUTER_SYSTEM = f"""Você classifica perguntas sobre o mercado de fundos brasileiro.
Responda somente com JSON no formato {{"route": "...", "intent": "...", "entity": "..."}}.
- route "normas": dúvidas sobre regras, regulamentos, taxas, obrigações, prazos.
- route "fundos": perguntas sobre fundos específicos, gestores ou administradores.
Para "fundos", intent é um de {sorted(QUERIES)} e entity é o nome ou CNPJ citado.
Para "normas", use intent "" e entity ""."""

ANSWER_SYSTEM = """Você é um analista do mercado de capitais brasileiro.
Responda em português usando SOMENTE o contexto fornecido.
Cite as fontes com [n] logo após cada afirmação.
Se o contexto não responder à pergunta, diga isso claramente em vez de supor."""

ENTITY_RE = re.compile(r"(?:gestora?|administradora?|fundo)\s+(?:d[oae]s?\s+)?(.+)$", re.IGNORECASE)
FUNDOS_HINTS = re.compile(r"\b(gestor|gestora|administrador|administradora|cnpj|quais fundos)\b")


class AgentState(TypedDict, total=False):
    question: str
    route: str
    intent: str
    entity: str
    chunks: list[RetrievedChunk]
    rows: list[dict[str, Any]]
    answer: str


def heuristic_route(question: str) -> dict[str, str]:
    q = question.lower()
    if not FUNDOS_HINTS.search(q):
        return {"route": "normas", "intent": "", "entity": ""}
    if "maiores" in q or "ranking" in q:
        return {"route": "fundos", "intent": "maiores_gestores", "entity": ""}
    match = ENTITY_RE.search(question)
    entity = match.group(1).strip(" ?.!") if match else ""
    intent = "gestor_do_fundo" if re.search(r"quem (é|e) o gestor", q) else "fundos_do_gestor"
    return {"route": "fundos", "intent": intent, "entity": entity}


class FinRAGAgent:
    def __init__(
        self,
        llm: LLM,
        retriever: HybridRetriever,
        graph: GraphStore | None = None,
        top_k: int = 5,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.graph = graph
        self.top_k = top_k
        self.app = self._build()

    def _route(self, state: AgentState) -> AgentState:
        decision = heuristic_route(state["question"])
        if self.llm.name != "extractive":
            try:
                parsed = json.loads(self.llm.complete(ROUTER_SYSTEM, state["question"]))
                if parsed.get("route") in {"normas", "fundos"}:
                    decision = {k: str(parsed.get(k, "")) for k in ("route", "intent", "entity")}
            except (json.JSONDecodeError, AttributeError):
                pass  # Mantém o roteamento por regras se o LLM não devolver JSON válido.
        if decision["route"] == "fundos" and (
            self.graph is None or decision["intent"] not in QUERIES
        ):
            decision = {"route": "normas", "intent": "", "entity": ""}
        return decision

    def _retrieve_docs(self, state: AgentState) -> AgentState:
        return {"chunks": self.retriever.retrieve(state["question"], k=self.top_k)}

    def _query_graph(self, state: AgentState) -> AgentState:
        assert self.graph is not None
        return {"rows": self.graph.query(state["intent"], state.get("entity", ""))}

    def _generate(self, state: AgentState) -> AgentState:
        if state.get("route") == "fundos":
            rows = state.get("rows", [])
            context = "\n".join(
                f"[{i}] {json.dumps(r, ensure_ascii=False)}" for i, r in enumerate(rows, 1)
            )
        else:
            chunks = state.get("chunks", [])
            context = "\n\n".join(
                f"[{i}] ({c.chunk.source}{', ' + c.chunk.article if c.chunk.article else ''})\n"
                f"{c.chunk.text}"
                for i, c in enumerate(chunks, 1)
            )
        prompt = f"Contexto:\n{context or '(vazio)'}\n\nPergunta: {state['question']}"
        return {"answer": self.llm.complete(ANSWER_SYSTEM, prompt)}

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("retrieve_docs", self._retrieve_docs)
        graph.add_node("query_graph", self._query_graph)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda s: "query_graph" if s["route"] == "fundos" else "retrieve_docs",
        )
        graph.add_edge("retrieve_docs", "generate")
        graph.add_edge("query_graph", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def ask(self, question: str) -> Answer:
        state = self.app.invoke({"question": question})
        sources = [
            Source(
                id=c.chunk.id,
                source=c.chunk.source,
                article=c.chunk.article,
                excerpt=c.chunk.text[:300],
                score=round(c.score, 4),
            )
            for c in state.get("chunks", [])
        ]
        return Answer(
            question=question, answer=state["answer"], route=state["route"], sources=sources
        )
