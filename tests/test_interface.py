import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from finrag.agent import FinRAGAgent
from finrag.api import create_app
from finrag.llm import ExtractiveLLM

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(retriever):
    return TestClient(create_app(FinRAGAgent(ExtractiveLLM(), retriever)))


def test_pagina_inicial_serve_o_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "<title>FinRAG CVM</title>" in html
    # A página chama o /ask relativo e não depende de nada externo (funciona offline).
    assert 'fetch("ask"' in html
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html


def test_pagina_fora_do_schema_da_api(client):
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_fluxo_da_pagina_com_agente_offline(client):
    """O que a página faz ao enviar uma pergunta, com o LLM extrativo (sem chave)."""
    assert client.get("/").status_code == 200
    data = client.post("/ask", json={"question": "Qual é a taxa de administração do Alpha?"})
    data = data.json()
    assert data["route"] == "normas"
    assert data["answer"].startswith("Trecho mais relevante encontrado [1]")
    # Campos que a interface lê para as fontes e a linha de métricas.
    for source in data["sources"]:
        assert {"id", "source", "title", "article", "excerpt", "score", "text"} <= set(source)
        assert source["text"]
    usage = data["usage"]
    assert usage["cost_usd"] == 0 and usage["input_tokens"] == 0
    assert {"route", "retrieve", "generate"} <= set(usage["stage_latency_ms"])


def test_wheel_inclui_a_pagina(tmp_path):
    """No Docker o pacote é instalado sem o modo editável: o index.html precisa ir junto."""
    pytest.importorskip("setuptools")
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(ROOT / name, tmp_path / name)
    shutil.copytree(
        ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    dist = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from setuptools import build_meta; build_meta.build_wheel({str(dist)!r})",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (wheel,) = dist.glob("*.whl")
    with zipfile.ZipFile(wheel) as zf:
        assert "finrag/static/index.html" in zf.namelist()
