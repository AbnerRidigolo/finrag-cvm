"""Métricas Prometheus e cálculo de custo por pergunta.

Os preços por milhão de tokens vêm da configuração (LLM_PRICES), e não do código,
porque mudam com frequência. Um modelo sem preço configurado gera custo zero e
aparece na métrica `finrag_llm_unpriced_calls_total`, para não passar despercebido.
"""

from prometheus_client import Counter, Histogram

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32)

REQUESTS = Counter("finrag_requests_total", "Perguntas respondidas", ["route"])
ERRORS = Counter("finrag_errors_total", "Falhas ao responder", ["stage"])
REQUEST_LATENCY = Histogram(
    "finrag_request_latency_seconds",
    "Latência total por pergunta",
    ["route"],
    buckets=LATENCY_BUCKETS,
)
STAGE_LATENCY = Histogram(
    "finrag_stage_latency_seconds",
    "Latência por etapa do agente",
    ["stage"],
    buckets=LATENCY_BUCKETS,
)
LLM_TOKENS = Counter(
    "finrag_llm_tokens_total", "Tokens consumidos no LLM", ["provider", "model", "kind"]
)
LLM_COST = Counter("finrag_llm_cost_usd_total", "Custo estimado em USD", ["provider", "model"])
UNPRICED = Counter(
    "finrag_llm_unpriced_calls_total", "Chamadas a modelos sem preço configurado", ["model"]
)
EMBEDDING_TOKENS = Counter("finrag_embedding_tokens_total", "Tokens de embeddings", ["model"])


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, prices: dict[str, list[float]]
) -> float | None:
    """Custo em USD dado o preço [entrada, saída] por milhão de tokens. None se sem preço."""
    price = prices.get(model)
    if not price:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


def record_llm_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    prices: dict[str, list[float]],
) -> float:
    LLM_TOKENS.labels(provider, model, "input").inc(input_tokens)
    LLM_TOKENS.labels(provider, model, "output").inc(output_tokens)
    cost = cost_usd(model, input_tokens, output_tokens, prices)
    if cost is None:
        if provider != "extractive":
            UNPRICED.labels(model).inc()
        return 0.0
    LLM_COST.labels(provider, model).inc(cost)
    return cost
