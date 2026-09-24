"""Leitura de documentos brutos (PDF e texto) do diretório de entrada."""

import json
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class RawDocument:
    source: str
    text: str


def load_file(path: Path) -> RawDocument:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8")
    return RawDocument(source=path.name, text=text)


def load_directory(directory: str | Path) -> list[RawDocument]:
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Diretório não encontrado: {root}")
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        raise ValueError(f"Nenhum arquivo {sorted(SUPPORTED)} em {root}")
    return [load_file(p) for p in files]


def load_doc_titles(path: str | Path) -> dict[str, dict[str, str]]:
    """Mapa arquivo -> {"title", "subject"}. O título extraído da primeira linha de um PDF
    costuma ser timbre ou número de página, então os títulos vêm deste arquivo explícito.
    "title" vai no cabeçalho indexado e nas fontes; "subject" é o que a pergunta sintética
    deve citar (o fundo ou o tipo de fundo). Devolve {} se o arquivo não existir."""
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))
