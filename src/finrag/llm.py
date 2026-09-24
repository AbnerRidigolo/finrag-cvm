"""Camada de LLM com provedores intercambiáveis (OpenAI, Anthropic e modo offline)."""

from typing import Protocol

from finrag.config import Settings


class LLM(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> str: ...


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class ExtractiveLLM:
    """Modo offline: não gera texto, devolve o trecho mais relevante com a citação.

    Serve para rodar o projeto sem chave de API. O agente detecta este provedor e usa o
    roteador por regras no lugar do roteador por LLM.
    """

    name = "extractive"

    def complete(self, system: str, prompt: str) -> str:
        marker = "[1]"
        start = prompt.find(marker)
        if start == -1:
            return "Não encontrei trechos relevantes nos documentos indexados."
        end = prompt.find("\n[2]", start)
        excerpt = prompt[start + len(marker) : end if end != -1 else None].strip()
        return f"Trecho mais relevante encontrado [1]:\n{excerpt[:800]}"


def build_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAILLM(settings.llm_model)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model)
    return ExtractiveLLM()
