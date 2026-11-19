import pytest

from finrag.agent import FinRAGAgent
from finrag.eval.retrieval_eval import load_dataset
from finrag.llm import ExtractiveLLM
from finrag.schemas import Chunk, RetrievedChunk

CASES = load_dataset("data/eval/challenge_questions.jsonl")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_offline_abstention_on_challenge_questions(retriever, case):
    answer = FinRAGAgent(ExtractiveLLM(), retriever).ask(case["question"])
    if case["relevant"]:
        assert not answer.abstained
        assert "[1]" in answer.answer
    else:
        assert answer.abstained
        assert "evidência suficiente" in answer.answer
        assert "[1]" not in answer.answer


class StaticRetriever:
    def __init__(self, fragments):
        self.fragments = fragments

    def retrieve(self, question, k):
        return [
            RetrievedChunk(
                chunk=Chunk(id=str(i), text=text, source="ata.txt"),
                score=1,
                retriever="test",
            )
            for i, text in enumerate(self.fragments[:k])
        ]


def test_no_context_is_a_structured_abstention():
    answer = FinRAGAgent(ExtractiveLLM(), StaticRetriever([])).ask("Qual é a taxa?")
    assert answer.abstained
    assert answer.abstention_reason == "no_context"
    assert answer.usage.llm_calls == 0


def test_documented_date_remains_answerable():
    retriever = StaticRetriever(["A assembleia aprovou a prorrogação em 18 de maio de 2025."])
    answer = FinRAGAgent(ExtractiveLLM(), retriever).ask(
        "Em que data a assembleia aprovou a prorrogação em 2025?"
    )
    assert not answer.abstained
    assert "18 de maio de 2025" in answer.answer


def test_other_year_cannot_count_as_evidence():
    retriever = StaticRetriever(["A assembleia aprovou a prorrogação em 18/05/2024."])
    answer = FinRAGAgent(ExtractiveLLM(), retriever).ask(
        "Em que data a assembleia aprovou a prorrogação em 2025?"
    )
    assert answer.abstained
    assert answer.abstention_reason == "missing_temporal_evidence"
