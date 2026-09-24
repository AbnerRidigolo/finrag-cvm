from pathlib import Path

import pytest

from finrag.graph.cadastro import only_digits, parse_float, read_cadastro

CSV = Path(__file__).resolve().parents[1] / "data" / "sample" / "cad_fi_exemplo.csv"


def test_le_apenas_fundos_ativos_e_normaliza():
    records = read_cadastro(CSV)
    assert len(records) == 3
    alpha = records[0]
    assert alpha.cnpj == "11111111000111"
    assert alpha.gestor_nome == "GESTORA ALFA CAPITAL LTDA"
    assert alpha.patrimonio == pytest.approx(125000000.50)


def test_inclui_inativos_quando_pedido():
    assert len(read_cadastro(CSV, only_active=False)) == 4


def test_coluna_ausente_gera_erro_claro(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("CNPJ_FUNDO;DENOM_SOCIAL\n1;X\n", encoding="latin-1")
    with pytest.raises(ValueError, match="Colunas ausentes"):
        read_cadastro(bad)


def test_helpers():
    assert only_digits("11.111.111/0001-11") == "11111111000111"
    assert parse_float("") is None
    assert parse_float("1,5") == 1.5
    assert parse_float("abc") is None
