"""Leitura do cadastro de fundos da CVM (arquivo cad_fi.csv do Portal de Dados Abertos).

O arquivo usa ';' como separador e codificação latin-1. A CVM altera a estrutura dos
seus conjuntos de dados de tempos em tempos (a Resolução CVM 175 criou novos arquivos de
classes e subclasses), por isso o mapeamento de colunas fica explícito aqui e a leitura
falha com uma mensagem clara se alguma coluna esperada sumir.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

COLUMNS = {
    "cnpj": "CNPJ_FUNDO",
    "nome": "DENOM_SOCIAL",
    "situacao": "SIT",
    "classe": "CLASSE",
    "admin_cnpj": "CNPJ_ADMIN",
    "admin_nome": "ADMIN",
    "gestor_doc": "CPF_CNPJ_GESTOR",
    "gestor_nome": "GESTOR",
    "patrimonio": "VL_PATRIM_LIQ",
}


@dataclass(frozen=True)
class FundoRecord:
    cnpj: str
    nome: str
    situacao: str
    classe: str
    admin_cnpj: str
    admin_nome: str
    gestor_doc: str
    gestor_nome: str
    patrimonio: float | None


def only_digits(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def read_cadastro(path: str | Path, only_active: bool = True) -> list[FundoRecord]:
    with open(path, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing = set(COLUMNS.values()) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Colunas ausentes em {path}: {sorted(missing)}. "
                f"Colunas disponíveis: {reader.fieldnames}"
            )
        records = []
        for row in reader:
            if (
                only_active
                and row[COLUMNS["situacao"]].strip().upper() != "EM FUNCIONAMENTO NORMAL"
            ):
                continue
            cnpj = only_digits(row[COLUMNS["cnpj"]])
            if not cnpj:
                continue
            records.append(
                FundoRecord(
                    cnpj=cnpj,
                    nome=row[COLUMNS["nome"]].strip(),
                    situacao=row[COLUMNS["situacao"]].strip(),
                    classe=row[COLUMNS["classe"]].strip(),
                    admin_cnpj=only_digits(row[COLUMNS["admin_cnpj"]]),
                    admin_nome=row[COLUMNS["admin_nome"]].strip(),
                    gestor_doc=only_digits(row[COLUMNS["gestor_doc"]]),
                    gestor_nome=row[COLUMNS["gestor_nome"]].strip(),
                    patrimonio=parse_float(row[COLUMNS["patrimonio"]]),
                )
            )
        return records
