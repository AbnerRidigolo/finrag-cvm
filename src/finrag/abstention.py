"""Limites explícitos da resposta extrativa quando a pergunta exige evidência temporal."""

import re

YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_QUESTION = re.compile(r"\b(?:em que data|qual (?:foi )?a data|quando)\b", re.IGNORECASE)
PAST_EVENT = re.compile(
    r"\b(?:aprovou|aprovad[ao]s?|ocorreu|aconteceu|realizou|publicou|deliberou|foi|foram)\b",
    re.IGNORECASE,
)
PERFORMANCE = re.compile(r"\b(?:retorno|rentabilidade|rendimento|desempenho)\b", re.IGNORECASE)
PAST_RESULT = re.compile(r"\b(?:foi|teve|rendeu|registrou|observad[ao]s?)\b", re.IGNORECASE)
CALENDAR_DATE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}|"
    r"\d{1,2}\s+de\s+(?:janeiro|fevereiro|março|abril|maio|junho|julho|agosto|"
    r"setembro|outubro|novembro|dezembro)\s+de\s+(?:19|20)\d{2})\b",
    re.IGNORECASE,
)


def has_required_temporal_evidence(question: str, excerpt: str) -> bool:
    """Test only the excerpt the offline provider will actually show."""
    years = set(YEAR.findall(question))
    if DATE_QUESTION.search(question) and PAST_EVENT.search(question):
        dated_years = {
            year for date in CALENDAR_DATE.findall(excerpt) for year in YEAR.findall(date)
        }
        return bool(dated_years) and years <= dated_years
    if years and PERFORMANCE.search(question) and PAST_RESULT.search(question):
        return years <= set(YEAR.findall(excerpt))
    return True
