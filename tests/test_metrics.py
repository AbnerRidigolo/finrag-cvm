import json
import re
from pathlib import Path

from prometheus_client import generate_latest

from finrag import metrics

ROOT = Path(__file__).resolve().parents[1]


def test_series_conhecidas_existem_antes_do_primeiro_evento():
    """Sem série em zero, o primeiro erro ou pergunta nasce valendo 1 e increase() o ignora."""
    body = generate_latest().decode()
    for stage in metrics.STAGES:
        assert re.search(rf'finrag_errors_total\{{stage="{stage}"\}} ', body)
        assert re.search(rf'finrag_stage_latency_seconds_count\{{stage="{stage}"\}} ', body)
    for route in metrics.ROUTES:
        assert re.search(rf'finrag_requests_total\{{route="{route}"\}} ', body)
        assert re.search(rf'finrag_request_latency_seconds_count\{{route="{route}"\}} ', body)


def test_faixas_de_latencia_finas_onde_ficam_as_respostas():
    buckets = metrics.LATENCY_BUCKETS
    assert list(buckets) == sorted(buckets)
    # Nenhuma faixa com mais de 3 s de largura entre 2 e 15 s.
    faixa = [b for b in buckets if 2 <= b <= 15]
    assert max(b - a for a, b in zip(faixa, faixa[1:], strict=False)) <= 3


def test_painel_de_erros_mostra_zero_sem_eventos():
    dashboard = json.loads((ROOT / "monitoring/grafana/dashboards/finrag.json").read_text("utf-8"))
    panel = next(p for p in dashboard["panels"] if p["title"].startswith("Erros"))
    assert all(t["expr"].endswith("or vector(0)") for t in panel["targets"])
