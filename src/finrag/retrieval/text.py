"""Normalização e tokenização de português usadas pela busca esparsa."""

import re
import unicodedata


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


_RAW_STOPWORDS = """
    a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos
    pelas para com sem sob sobre e ou que se ao aos à às é ser são foi como qual
    quais quem onde quando mais menos muito seu sua seus suas este esta estes estas
    esse essa isso isto ele ela eles elas lhe já não sim deve devem pode podem será
    """.split()  # noqa: SIM905
# Os tokens perdem acentos na normalização, então as stopwords também precisam perder.
STOPWORDS = frozenset(strip_accents(w) for w in _RAW_STOPWORDS)
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(strip_accents(text.lower()))
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]
