"""Perguntas da interface selecionadas pelos arquivos presentes no índice."""

from typing import Literal

from pydantic import BaseModel

SAMPLE_EXAMPLES = (
    (
        "regulamento_fundo_exemplo_alpha.txt",
        "Exemplo fictício",
        "Alpha FIDC",
        "Qual é a taxa de administração do Fundo Exemplo Alpha?",
    ),
    (
        "regulamento_fundo_exemplo_beta.txt",
        "Exemplo fictício",
        "Beta FIP",
        "Qual é o período de investimento do Fundo Exemplo Beta?",
    ),
    (
        "regulamento_fundo_exemplo_beta.txt",
        "Exemplo fictício",
        "Beta FIP",
        "Como é calculada a taxa de performance do FIP Beta?",
    ),
    (
        "glossario_fundos.txt",
        "Exemplo fictício",
        "Glossário",
        "O que é linha d'água?",
    ),
)

# Perguntas do benchmark real com gabarito recuperado e nota 5 nos dois critérios do juiz.
CVM_EXAMPLES = (
    (
        "resol175consolid_Anexo10.pdf",
        "Resolução 175",
        "Anexo X · Ficart",
        "Que tipo de investimento caracteriza os fundos chamados Ficart?",
    ),
    (
        "regulamento_fidc_croma_2025-09.pdf",
        "Regulamento",
        "Croma FIDC",
        "No Croma FIDC, a taxa de administração pode ser reduzida sem aprovação dos cotistas? "
        "E o que é necessário para aumentá-la?",
    ),
    (
        "resol175consolid_Anexo09.pdf",
        "Resolução 175",
        "Anexo IX · FMAI",
        "Em que condições a classe de cotas do FMAI pode passar do regime fechado para o aberto?",
    ),
    (
        "resol175consolid_Anexo05.pdf",
        "Resolução 175",
        "Anexo V · Fundo de Índice",
        "Com que frequência devem ser atualizadas as informações sobre a composição da carteira "
        "do Fundo de Índice?",
    ),
)


class SuggestedQuestion(BaseModel):
    document: str
    topic: str
    question: str


class Examples(BaseModel):
    corpus: Literal["empty", "sample", "cvm", "mixed", "other"]
    questions: list[SuggestedQuestion]


def select_examples(indexed: set[str]) -> Examples:
    sample = [
        SuggestedQuestion(document=document, topic=topic, question=question)
        for source, document, topic, question in SAMPLE_EXAMPLES
        if source in indexed
    ]
    cvm = [
        SuggestedQuestion(document=document, topic=topic, question=question)
        for source, document, topic, question in CVM_EXAMPLES
        if source in indexed
    ]
    if sample and cvm:
        return Examples(corpus="mixed", questions=sample + cvm)
    if sample:
        return Examples(corpus="sample", questions=sample)
    if cvm:
        return Examples(corpus="cvm", questions=cvm)
    return Examples(corpus="other" if indexed else "empty", questions=[])
