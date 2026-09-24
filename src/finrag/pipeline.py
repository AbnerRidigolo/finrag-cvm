"""CLI: indexar documentos, fazer perguntas e carregar o grafo de fundos."""

import argparse
import time

from finrag.config import get_settings
from finrag.factory import build_agent, build_dense
from finrag.ingestion.chunking import chunk_documents
from finrag.ingestion.loaders import load_directory


def cmd_index(directory: str) -> None:
    s = get_settings()
    start = time.perf_counter()
    docs = load_directory(directory)
    chunks = chunk_documents(docs, s.chunk_max_chars, s.chunk_overlap_chars)
    build_dense(s).add(chunks)
    elapsed = time.perf_counter() - start
    print(f"{len(docs)} documentos -> {len(chunks)} chunks indexados em {elapsed:.1f}s")


def cmd_ask(question: str) -> None:
    answer = build_agent().ask(question)
    print(f"[rota: {answer.route}]\n\n{answer.answer}\n")
    for i, src in enumerate(answer.sources, 1):
        print(f"[{i}] {src.source} {src.article} (score {src.score})")


def cmd_graph_load(csv_path: str) -> None:
    from finrag.factory import build_graph
    from finrag.graph.cadastro import read_cadastro

    s = get_settings()
    s.graph_enabled = True
    graph = build_graph(s)
    records = read_cadastro(csv_path)
    print(f"{graph.load(records)} fundos carregados no Neo4j")
    graph.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("directory")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("graph-load").add_argument("csv_path")
    args = parser.parse_args()
    {
        "index": lambda: cmd_index(args.directory),
        "ask": lambda: cmd_ask(args.question),
        "graph-load": lambda: cmd_graph_load(args.csv_path),
    }[args.cmd]()


if __name__ == "__main__":
    main()
