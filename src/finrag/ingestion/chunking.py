"""Chunking orientado à estrutura de textos normativos.

Normas e regulamentos são organizados em artigos ("Art. 1º", "Art. 2º"...). Cortar no
meio de um artigo separa a regra das suas exceções, então o chunker primeiro divide por
artigo e só quebra um artigo em pedaços menores quando ele excede o limite de tamanho.
"""

import hashlib
import logging
import re

from finrag.ingestion.loaders import RawDocument
from finrag.schemas import Chunk

logger = logging.getLogger(__name__)

# Normas da CVM usam "Art. 1º". Regulamentos de fundos costumam escrever "Artigo 1º" ou
# "Cláusula 1"; nesses, o número precisa vir seguido de uma maiúscula, para não cortar
# uma frase que só por acaso quebrou a linha antes de "Artigo 12 deste Regulamento".
# Numerações como "1." e "1.1" ficam de fora: nos regulamentos são incisos e subitens
# dentro de um artigo, e no texto extraído do PDF colidem com CNPJs e valores.
_HEADING = (
    r"Art\.\s*\d+"
    r"|(?:Artigo|Cl[áa]usula)\s+\d+\s*[ºo°]?\s*[.:\-–]?\s+[A-ZÀ-Ý]"
)
ARTICLE_RE = re.compile(rf"(?=^\s*(?:{_HEADING}))", re.MULTILINE)
ARTICLE_LABEL_RE = re.compile(
    r"^\s*(?:Art\.\s*(?P<art>\d+)|(?P<kind>Artigo|Cl[áa]usula)\s+(?P<num>\d+)\s*)"
    r"(?P<ord>[ºo°]?)(?:[-‑–](?P<suffix>[A-Z])(?![a-z]))?"
)
# Primeiro artigo de uma numeração: se reaparece, a numeração recomeçou (ex.: o anexo da
# classe de cotas de um regulamento volta ao "Artigo 1").
FIRST_ARTICLE_RE = re.compile(r"(?:Art\.|Cláusula )1[ºo°]?")
# Textos consolidados da CVM (ex.: resol175consolid_ParteGeral.pdf) mostram a redação
# revogada tachada logo antes da vigente. O tachado se perde na extração do PDF, então o
# mesmo artigo aparece duas ou mais vezes seguidas, e só a última versão está em vigor.
CONSOLIDATED_SOURCE_RE = re.compile(r"^resol\d+consolid")


def _article_label(section: str) -> str:
    """Rótulo normalizado: "Art. 1º", "Artigo 01." e "Art. 110-A" viram "Art.1º", "Art.1"
    e "Art.110-A" (o sufixo distingue artigos incluídos depois, como o 110-A do 110)."""
    if not re.match(rf"^\s*(?:{_HEADING})", section):
        return ""
    match = ARTICLE_LABEL_RE.match(section)
    if match is None:
        return ""
    suffix = f"-{match['suffix']}" if match["suffix"] else ""
    if match["art"]:
        return f"Art.{match['art']}{match['ord']}{suffix}"
    number = int(match["num"])
    if match["kind"] == "Artigo":
        return f"Art.{number}{match['ord']}{suffix}"
    return f"Cláusula {number}{suffix}"


def drop_superseded(sections: list[str]) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Remove as redações revogadas: de seções seguidas do mesmo artigo, fica a última.

    Atua sobre artigos inteiros, antes da quebra em pedaços, para nunca descartar a
    continuação de um artigo longo. Devolve as seções mantidas e (artigo, descartada,
    mantida) de cada descarte, para registro.
    """
    labels = [_article_label(s) for s in sections]
    kept: list[str] = []
    dropped: list[tuple[str, str, str]] = []
    for i, section in enumerate(sections):
        if labels[i] and i + 1 < len(sections) and labels[i + 1] == labels[i]:
            dropped.append((labels[i], section, sections[i + 1]))
            continue
        kept.append(section)
    return kept, dropped


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    pieces, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefere cortar em fim de frase ou de parágrafo dentro da janela.
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in pieces if p]


def _chunk_id(source: str, position: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{position}|{text}".encode()).hexdigest()[:12]
    return f"{source}:{position}:{digest}"


def _title(text: str) -> str:
    first_line = text.split("\n", 1)[0].strip()
    return "" if _article_label(first_line) else first_line[:200]


def chunk_document(doc: RawDocument, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    text = _normalize(doc.text)
    title = _title(text)
    sections = [s.strip() for s in ARTICLE_RE.split(text) if s.strip()]
    if CONSOLIDATED_SOURCE_RE.match(doc.source):
        sections, dropped = drop_superseded(sections)
        for article, old, _ in dropped:
            logger.info(
                "Redação revogada descartada: %s %s: %s",
                doc.source,
                article,
                " ".join(old.split())[:100],
            )
    chunks: list[Chunk] = []
    seen: set[str] = set()
    part = 1
    for section in sections:
        article = _article_label(section)
        if article in seen and FIRST_ARTICLE_RE.fullmatch(article):
            # Regulamentos com parte geral e anexo da classe recomeçam no Artigo 1. Sem
            # distinguir as partes, "Art.5" apontaria para dois trechos e o gabarito da
            # avaliação ficaria ambíguo.
            part += 1
            seen = set()
        if article:
            seen.add(article)
            if part > 1:
                article = f"{article} (parte {part})"
        for piece in _split_long(section, max_chars, overlap):
            position = len(chunks)
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.source, position, piece),
                    text=piece,
                    source=doc.source,
                    article=article,
                    position=position,
                    context=title,
                )
            )
    return chunks


def chunk_documents(
    docs: list[RawDocument], max_chars: int = 1200, overlap: int = 150
) -> list[Chunk]:
    return [c for doc in docs for c in chunk_document(doc, max_chars, overlap)]
