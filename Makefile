.PHONY: install dev lint test index demo api eval up down graph-load download index-real dataset eval-real eval-answers judge-control

install:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest --cov=finrag --cov-report=term-missing

index:
	python -m finrag.pipeline index data/sample/docs

demo: index
	python -m finrag.pipeline ask "Qual é a taxa de administração do Fundo Exemplo Alpha?"

eval: index
	python -m finrag.eval.retrieval_eval data/eval/questions.jsonl

api:
	uvicorn finrag.api:app --reload

up:
	docker compose up -d --build

down:
	docker compose down

graph-load:
	python -m finrag.pipeline graph-load data/sample/cad_fi_exemplo.csv

download:
	python -m finrag.pipeline download-cadastro data/raw/cad_fi.csv

index-real:
	python -m finrag.pipeline index data/raw/docs

dataset:
	python -m finrag.eval.build_dataset data/eval/cvm_questions.jsonl -n 50

eval-real:
	python -m finrag.eval.retrieval_eval data/eval/cvm_questions.jsonl --output data/eval/results/retrieval.md

eval-answers:
	python -m finrag.eval.answer_eval data/eval/cvm_questions.jsonl -n 30 --rejudge 10 --max-cost 0.35

judge-control:
	python -m finrag.eval.judge_control data/eval/results/answer_eval.jsonl --max-cost 0.12
