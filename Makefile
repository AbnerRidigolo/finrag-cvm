.PHONY: install dev lint test index demo api eval up down graph-load

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
