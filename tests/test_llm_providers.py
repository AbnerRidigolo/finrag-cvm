from types import SimpleNamespace as NS

from finrag.llm import AnthropicLLM, OpenAILLM


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn", text='{"ok": 1}'):
        self.kwargs = {}
        content = [NS(type="text", text=text)] if text else []
        self.response = NS(
            content=content,
            stop_reason=stop_reason,
            stop_details=NS(category="cyber") if stop_reason == "refusal" else None,
            usage=NS(input_tokens=120, output_tokens=8),
        )
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeOpenAI:
    def __init__(self, content='{"ok": 1}', refusal=None):
        self.kwargs = {}
        message = NS(content=content, refusal=refusal)
        self.response = NS(
            choices=[NS(message=message)], usage=NS(prompt_tokens=90, completion_tokens=5)
        )
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_anthropic_sem_temperature_e_com_thinking_desativado():
    client = FakeAnthropic()
    completion = AnthropicLLM("claude-sonnet-5", client=client).complete("sys", "p")
    assert "temperature" not in client.kwargs
    assert client.kwargs["thinking"] == {"type": "disabled"}
    assert completion.text == '{"ok": 1}'
    assert (completion.input_tokens, completion.output_tokens) == (120, 8)


def test_anthropic_recusa_devolve_texto_vazio_e_registra(caplog):
    client = FakeAnthropic(stop_reason="refusal", text="")
    with caplog.at_level("WARNING", logger="finrag"):
        completion = AnthropicLLM("claude-sonnet-5", client=client).complete("sys", "p")
    assert completion.text == ""
    assert "Recusa" in caplog.text and "cyber" in caplog.text


def test_openai_sem_temperature():
    client = FakeOpenAI()
    completion = OpenAILLM("gpt-6-luna", client=client).complete("sys", "p")
    assert "temperature" not in client.kwargs
    assert completion.text == '{"ok": 1}' and completion.output_tokens == 5


def test_openai_recusa_registra(caplog):
    client = FakeOpenAI(content=None, refusal="não posso ajudar")
    with caplog.at_level("WARNING", logger="finrag"):
        completion = OpenAILLM("gpt-6-luna", client=client).complete("sys", "p")
    assert completion.text == "" and "Recusa" in caplog.text
