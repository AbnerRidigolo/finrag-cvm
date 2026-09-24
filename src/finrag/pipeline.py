"""CLI: baixar dados, indexar documentos, fazer perguntas e carregar o grafo de fundos."""

import argparse
import logging
import time
import urllib.request
from pathlib import Path

from finrag.config import get_settings
from finrag.factory import build_agent, build_vector_store
from finrag.ingestion.chunking import chunk_documents
from finrag.ingestion.loaders import load_directory

CAD_FI_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"


def cmd_index(directory: str) -> None:
    s = get_settings()
    start = time.perf_counter()
    docs = load_directory(directory)
    chunks = chunk_documents(docs, s.chunk_max_chars, s.chunk_overlap_chars)
    build_vector_store(s).add(chunks)
    elapsed = time.perf_counter() - start
    print(
        f"{len(docs)} documentos -> {len(chunks)} chunks indexados no {s.vector_store} "
        f"em {elapsed:.1f}s"
    )


def cmd_ask(question: str) -> None:
    answer = build_agent().ask(question)
    print(f"[rota: {answer.route}]\n\n{answer.answer}\n")
    for i, src in enumerate(answer.sources, 1):
        print(f"[{i}] {src.source} {src.article} (score {src.score})")
    u = answer.usage
    stages = ", ".join(f"{k} {v:.0f} ms" for k, v in u.stage_latency_ms.items())
    print(
        f"\n{u.latency_ms:.0f} ms ({stages}) | {u.input_tokens}+{u.output_tokens} tokens "
        f"| US$ {u.cost_usd:.5f}"
    )


def cmd_download_cadastro(output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {CAD_FI_URL} ...")
    urllib.request.urlretrieve(CAD_FI_URL, path)
    print(f"Salvo em {path} ({path.stat().st_size / 1e6:.1f} MB)")


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
    # Logs do finrag (ex.: seções descartadas na indexação) em INFO; bibliotecas só em WARNING.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("finrag").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(prog="finrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("directory")
    sub.add_parser("ask").add_argument("question")
    sub.add_parser("graph-load").add_argument("csv_path")
    dl = sub.add_parser("download-cadastro")
    dl.add_argument("output", nargs="?", default="data/raw/cad_fi.csv")
    args = parser.parse_args()
    commands = {
        "index": lambda: cmd_index(args.directory),
        "ask": lambda: cmd_ask(args.question),
        "graph-load": lambda: cmd_graph_load(args.csv_path),
        "download-cadastro": lambda: cmd_download_cadastro(args.output),
    }
    commands[args.cmd]()


if __name__ == "__main__":
    main()
