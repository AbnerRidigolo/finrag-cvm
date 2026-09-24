"""Grafo de fundos no Neo4j.

Modelo: (Fundo)-[:GERIDO_POR]->(Gestor) e (Fundo)-[:ADMINISTRADO_POR]->(Administrador).

As consultas são templates Cypher parametrizados, escolhidos pelo agente a partir da
intenção detectada. Gerar Cypher livre com o LLM (text-to-Cypher) seria mais flexível,
mas abre espaço para consultas erradas ou caras; templates são previsíveis e seguros.
"""

from dataclasses import asdict
from typing import Any, Protocol

from finrag.graph.cadastro import FundoRecord

LOAD_QUERY = """
UNWIND $rows AS row
MERGE (f:Fundo {cnpj: row.cnpj})
  SET f.nome = row.nome, f.situacao = row.situacao, f.classe = row.classe,
      f.patrimonio = row.patrimonio
FOREACH (_ IN CASE WHEN row.gestor_doc <> '' THEN [1] ELSE [] END |
  MERGE (g:Gestor {doc: row.gestor_doc}) SET g.nome = row.gestor_nome
  MERGE (f)-[:GERIDO_POR]->(g))
FOREACH (_ IN CASE WHEN row.admin_cnpj <> '' THEN [1] ELSE [] END |
  MERGE (a:Administrador {cnpj: row.admin_cnpj}) SET a.nome = row.admin_nome
  MERGE (f)-[:ADMINISTRADO_POR]->(a))
"""

CONSTRAINTS = [
    "CREATE CONSTRAINT fundo_cnpj IF NOT EXISTS FOR (f:Fundo) REQUIRE f.cnpj IS UNIQUE",
    "CREATE CONSTRAINT gestor_doc IF NOT EXISTS FOR (g:Gestor) REQUIRE g.doc IS UNIQUE",
    "CREATE CONSTRAINT admin_cnpj IF NOT EXISTS FOR (a:Administrador) REQUIRE a.cnpj IS UNIQUE",
]

QUERIES: dict[str, str] = {
    "fundos_do_gestor": """
        MATCH (f:Fundo)-[:GERIDO_POR]->(g:Gestor)
        WHERE toLower(g.nome) CONTAINS toLower($entity)
        RETURN g.nome AS gestor, f.nome AS fundo, f.classe AS classe, f.patrimonio AS patrimonio
        ORDER BY f.patrimonio DESC LIMIT 20""",
    "gestor_do_fundo": """
        MATCH (f:Fundo)-[:GERIDO_POR]->(g:Gestor)
        WHERE toLower(f.nome) CONTAINS toLower($entity) OR f.cnpj = $entity
        OPTIONAL MATCH (f)-[:ADMINISTRADO_POR]->(a:Administrador)
        RETURN f.nome AS fundo, g.nome AS gestor, a.nome AS administrador LIMIT 20""",
    "maiores_gestores": """
        MATCH (f:Fundo)-[:GERIDO_POR]->(g:Gestor)
        RETURN g.nome AS gestor, count(f) AS fundos, sum(f.patrimonio) AS patrimonio_total
        ORDER BY patrimonio_total DESC LIMIT 10""",
}


class GraphStore(Protocol):
    def query(self, intent: str, entity: str) -> list[dict[str, Any]]: ...


class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self.driver.close()

    def load(self, records: list[FundoRecord], batch_size: int = 1000) -> int:
        with self.driver.session() as session:
            for statement in CONSTRAINTS:
                session.run(statement)
            for i in range(0, len(records), batch_size):
                rows = [asdict(r) for r in records[i : i + batch_size]]
                session.run(LOAD_QUERY, rows=rows)
        return len(records)

    def query(self, intent: str, entity: str) -> list[dict[str, Any]]:
        if intent not in QUERIES:
            raise ValueError(f"Intenção sem consulta: {intent}")
        with self.driver.session() as session:
            return [r.data() for r in session.run(QUERIES[intent], entity=entity)]
