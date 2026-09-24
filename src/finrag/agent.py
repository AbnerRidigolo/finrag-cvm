"""Agente em LangGraph: roteia a pergunta, busca o contexto certo e responde com citações.

    pergunta ──> route ──┬─> retrieve_docs ──┐
                         └─> query_graph  ───┴─> generate ──> resposta

- "normas": perguntas sobre regras e regulamentos -> recuperação híbrida nos documentos.
- "fundos": perguntas sobre fundos, gestores e administradores -> consulta ao grafo Neo4j.

Cada etapa é cronometrada e cada chamada ao LLM registra tokens. Os dois vão para o
Prometheus e para o campo `usage` da resposta, o que dá latência e custo por pergunta.
"""

import json
import operator
import re
import time
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from finrag import metrics
from finrag.graph.store import QUERIES, GraphStore
from finrag.llm import LLM, Completion
from finrag.retrieval.hybrid import HybridRetriever
from finrag.schemas import Answer, RetrievedChunk, Source, Usage

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
    context: str
    answer: str
    completions: Annotated[list[Completion], operator.add]
    timings: Annotated[list[tuple[str, float]], operator.add]


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


def _timed(stage: str, fn: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
    def wrapper(state: AgentState) -> dict:
        start = time.perf_counter()
        try:
            result = fn(state)
        except Exception:
            metrics.ERRORS.labels(stage).inc()
            raise
        elapsed = time.perf_counter() - start
        metrics.STAGE_LATENCY.labels(stage).observe(elapsed)
        return {**result, "timings": [(stage, elapsed)]}

    return wrapper


class FinRAGAgent:
    def __init__(
        self,
        llm: LLM,
        retriever: HybridRetriever,
        graph: GraphStore | None = None,
        top_k: int = 5,
        prices: dict[str, list[float]] | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.graph = graph
        self.top_k = top_k
        self.prices = prices or {}
        self.app = self._build()

    def _route(self, state: AgentState) -> dict:
        decision = heuristic_route(state["question"])
        completions: list[Completion] = []
        if self.llm.name != "extractive":
            completion = self.llm.complete(ROUTER_SYSTEM, state["question"])
            completions.append(completion)
            try:
                parsed = json.loads(completion.text)
                if parsed.get("route") in {"normas", "fundos"}:
                    decision = {k: str(parsed.get(k, "")) for k in ("route", "intent", "entity")}
            except (json.JSONDecodeError, AttributeError):
                pass  # Mantém o roteamento por regras se o LLM não devolver JSON válido.
        if decision["route"] == "fundos" and (
            self.graph is None or decision["intent"] not in QUERIES
        ):
            decision = {"route": "normas", "intent": "", "entity": ""}
        return {**decision, "completions": completions}

    def _retrieve_docs(self, state: AgentState) -> dict:
        return {"chunks": self.retriever.retrieve(state["question"], k=self.top_k)}

    def _query_graph(self, state: AgentState) -> dict:
        assert self.graph is not None
        return {"rows": self.graph.query(state["intent"], state.get("entity", ""))}

    def _generate(self, state: AgentState) -> dict:
        if state.get("route") == "fundos":
            context = "\n".join(
                f"[{i}] {json.dumps(r, ensure_ascii=False)}"
                for i, r in enumerate(state.get("rows", []), 1)
            )
        else:
            context = "\n\n".join(
                f"[{i}] ({c.chunk.source}{', ' + c.chunk.article if c.chunk.article else ''})\n"
                f"{c.chunk.text}"
                for i, c in enumerate(state.get("chunks", []), 1)
            )
        prompt = f"Contexto:\n{context or '(vazio)'}\n\nPergunta: {state['question']}"
        completion = self.llm.complete(ANSWER_SYSTEM, prompt)
        return {"answer": completion.text, "context": context, "completions": [completion]}

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", _timed("route", self._route))
        graph.add_node("retrieve_docs", _timed("retrieve", self._retrieve_docs))
        graph.add_node("query_graph", _timed("graph", self._query_graph))
        graph.add_node("generate", _timed("generate", self._generate))
        graph.set_entry_point("route")
        graph.add_conditional_edges(
            "route",
            lambda s: "query_graph" if s["route"] == "fundos" else "retrieve_docs",
        )
        graph.add_edge("retrieve_docs", "generate")
        graph.add_edge("query_graph", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    def _usage(self, state: AgentState, elapsed: float) -> Usage:
        usage = Usage(latency_ms=round(elapsed * 1000, 1))
        for stage, seconds in state.get("timings", []):
            usage.stage_latency_ms[stage] = round(seconds * 1000, 1)
        for c in state.get("completions", []):
            usage.input_tokens += c.input_tokens
            usage.output_tokens += c.output_tokens
            usage.llm_calls += 1
            usage.cost_usd += metrics.record_llm_call(
                c.provider, c.model, c.input_tokens, c.output_tokens, self.prices
            )
        usage.cost_usd = round(usage.cost_usd, 6)
        return usage

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        state = self.app.invoke({"question": question})
        elapsed = time.perf_counter() - start

        metrics.REQUESTS.labels(state["route"]).inc()
        metrics.REQUEST_LATENCY.labels(state["route"]).observe(elapsed)

        sources = [
            Source(
                id=c.chunk.id,
                source=c.chunk.source,
                article=c.chunk.article,
                excerpt=c.chunk.text[:300],
                score=round(c.score, 4),
                text=c.chunk.text,
            )
            for c in state.get("chunks", [])
        ]
        return Answer(
            question=question,
            answer=state["answer"],
            route=state["route"],
            sources=sources,
            context=state.get("context", ""),
            usage=self._usage(state, elapsed),
        )
