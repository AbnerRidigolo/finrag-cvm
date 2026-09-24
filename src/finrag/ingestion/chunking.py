"""Chunking orientado à estrutura de textos normativos.

Normas e regulamentos são organizados em artigos ("Art. 1º", "Art. 2º"...). Cortar no
meio de um artigo separa a regra das suas exceções, então o chunker primeiro divide por
artigo e só quebra um artigo em pedaços menores quando ele excede o limite de tamanho.
"""

import hashlib
import re

from finrag.ingestion.loaders import RawDocument
from finrag.schemas import Chunk

ARTICLE_RE = re.compile(r"(?=^\s*Art\.\s*\d+)", re.MULTILINE)
ARTICLE_LABEL_RE = re.compile(r"^\s*(Art\.\s*\d+[ºo°]?)")


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
    return "" if ARTICLE_LABEL_RE.match(first_line) else first_line[:200]


def chunk_document(doc: RawDocument, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    text = _normalize(doc.text)
    title = _title(text)
    sections = [s.strip() for s in ARTICLE_RE.split(text) if s.strip()]
    chunks: list[Chunk] = []
    for section in sections:
        match = ARTICLE_LABEL_RE.match(section)
        article = match.group(1).replace(" ", "") if match else ""
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
