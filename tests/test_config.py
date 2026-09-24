import importlib
import os

from fastapi.testclient import TestClient

from finrag.agent import FinRAGAgent
from finrag.config import load_api_keys
from finrag.llm import ExtractiveLLM


def _unset(monkeypatch, *names):
    # setenv antes de delenv faz o monkeypatch registrar e restaurar o estado original.
    for name in names:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


def test_load_api_keys_exporta_so_chaves_e_nao_sobrescreve(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "OPENAI_API_KEY=do-arquivo\nANTHROPIC_API_KEY=do-arquivo\nLLM_PROVIDER=openai\n",
        encoding="utf-8",
    )
    _unset(monkeypatch, "OPENAI_API_KEY", "LLM_PROVIDER")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "do-ambiente")

    load_api_keys(str(env))

    assert os.environ["OPENAI_API_KEY"] == "do-arquivo"
    assert os.environ["ANTHROPIC_API_KEY"] == "do-ambiente"  # ambiente tem precedência
    assert "LLM_PROVIDER" not in os.environ  # só *_API_KEY é exportado


def test_api_com_agente_injetado_nao_carrega_o_env(tmp_path, monkeypatch, retriever):
    # Um .env com chave falsa no diretório atual: se algo o carregasse, os.environ mudaria.
    (tmp_path / ".env").write_text("OPENAI_API_KEY=nao-deveria-carregar\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _unset(monkeypatch, "OPENAI_API_KEY")
    antes = dict(os.environ)

    import finrag.api

    api = importlib.reload(finrag.api)
    client = TestClient(api.create_app(FinRAGAgent(ExtractiveLLM(), retriever)))
    assert client.post("/ask", json={"question": "O que é linha d'água?"}).status_code == 200

    assert dict(os.environ) == antes


def test_versao_igual_no_pacote_na_api_e_no_pyproject(retriever):
    import tomllib
    from pathlib import Path

    from finrag import __version__
    from finrag.api import create_app

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]
    app = create_app(FinRAGAgent(ExtractiveLLM(), retriever))
    assert __version__ == declared == app.version == "0.3.0"
