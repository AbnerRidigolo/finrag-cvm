"""API HTTP do FinRAG."""

from importlib.resources import files

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from finrag import __version__
from finrag.agent import FinRAGAgent
from finrag.schemas import Answer


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(agent: FinRAGAgent | None = None) -> FastAPI:
    app = FastAPI(title="FinRAG CVM", version=__version__)
    app.mount("/metrics", make_asgi_app())
    state: dict[str, FinRAGAgent] = {}

    def get_agent() -> FinRAGAgent:
        if "agent" not in state:
            if agent is not None:
                state["agent"] = agent
            else:
                # Só o caminho real carrega as chaves: testes injetam o agente e não passam aqui.
                from finrag.config import load_api_keys
                from finrag.factory import build_agent

                load_api_keys()
                state["agent"] = build_agent()
        return state["agent"]

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        # Lido do pacote, e não de um caminho relativo, para funcionar também quando o
        # pacote é instalado sem o modo editável (como no Docker).
        return files("finrag").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=Answer)
    def ask(payload: Question) -> Answer:
        return get_agent().ask(payload.question)

    return app


app = create_app()
