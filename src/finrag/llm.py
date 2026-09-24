"""Camada de LLM com provedores intercambiáveis (OpenAI, Anthropic e modo offline).

Todo provedor devolve um `Completion` com o texto e os tokens consumidos, o que permite
medir custo por pergunta sem depender de callbacks específicos de cada SDK.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from finrag.config import Settings

logger = logging.getLogger(__name__)

CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*(.*?)\s*```$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a cerca de código (```json ... ```) que alguns modelos colocam em volta do JSON."""
    text = text.strip()
    match = CODE_FENCE_RE.match(text)
    return match.group(1) if match else text


@dataclass(frozen=True)
class Completion:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM(Protocol):
    name: str
    model: str

    def complete(self, system: str, prompt: str) -> Completion: ...


# Nenhum dos provedores aceita mais temperature=0 nos modelos atuais (gpt-6-luna só aceita o
# valor padrão; claude-sonnet-5 removeu os parâmetros de amostragem). As respostas, e as notas
# do juiz, podem variar entre execuções.


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model

    def complete(self, system: str, prompt: str) -> Completion:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        usage = response.usage
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            logger.warning("Recusa do modelo %s: %s", self.model, message.refusal)
        return Completion(
            text=message.content or "",
            provider=self.name,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str, client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def complete(self, system: str, prompt: str) -> Completion:
        # Thinking desativado: as tarefas aqui (roteamento, perguntas, notas do juiz) devolvem
        # JSON curto, o max_tokens pequeno basta e o custo por chamada fica previsível.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            # Volta com HTTP 200 e sem texto; quem chama trata o texto vazio como inválido.
            details = getattr(response, "stop_details", None)
            logger.warning(
                "Recusa do modelo %s (categoria: %s)",
                self.model,
                getattr(details, "category", None),
            )
        return Completion(
            text="".join(block.text for block in response.content if block.type == "text"),
            provider=self.name,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


class ExtractiveLLM:
    """Modo offline: não gera texto, devolve o trecho mais relevante com a citação.

    Serve para rodar o projeto sem chave de API. O agente detecta este provedor e usa o
    roteador por regras no lugar do roteador por LLM.
    """

    name = "extractive"
    model = "extractive"

    def complete(self, system: str, prompt: str) -> Completion:
        marker = "[1]"
        start = prompt.find(marker)
        if start == -1:
            text = "Não encontrei trechos relevantes nos documentos indexados."
        else:
            end = prompt.find("\n[2]", start)
            excerpt = prompt[start + len(marker) : end if end != -1 else None].strip()
            text = f"Trecho mais relevante encontrado [1]:\n{excerpt[:800]}"
        return Completion(text=text, provider=self.name, model=self.model)


def make_llm(provider: str, model: str) -> LLM:
    if provider == "openai":
        return OpenAILLM(model)
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "extractive":
        return ExtractiveLLM()
    raise ValueError(f"Provedor de LLM desconhecido: {provider}")


def build_llm(settings: Settings) -> LLM:
    return make_llm(settings.llm_provider, settings.llm_model)


def build_judge(settings: Settings) -> LLM:
    provider = settings.judge_provider or settings.llm_provider
    if provider == "extractive":
        raise ValueError(
            "O juiz precisa de um LLM real. Defina JUDGE_PROVIDER=openai ou anthropic no .env."
        )
    return make_llm(provider, settings.judge_model or settings.llm_model)
