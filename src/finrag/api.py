"""API HTTP do FinRAG."""

from fastapi import FastAPI
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from finrag.agent import FinRAGAgent
from finrag.schemas import Answer


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(agent: FinRAGAgent | None = None) -> FastAPI:
    app = FastAPI(title="FinRAG CVM", version="0.2.0")
    app.mount("/metrics", make_asgi_app())
    state: dict[str, FinRAGAgent] = {}

    def get_agent() -> FinRAGAgent:
        if "agent" not in state:
            from finrag.factory import build_agent

            state["agent"] = agent or build_agent()
        return state["agent"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=Answer)
    def ask(payload: Question) -> Answer:
        return get_agent().ask(payload.question)

    return app


app = create_app()
