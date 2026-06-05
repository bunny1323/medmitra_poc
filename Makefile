.PHONY: install lint format test test-cov run docker-up docker-down rebuild-index check-index generate-keys

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	ruff check .
	mypy app/

format:
	ruff format .

test:
	pytest tests/

test-cov:
	coverage run -m pytest tests/
	coverage report -m

run:
	uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

docker-up:
	docker compose up -d

docker-down:
	docker compose down

rebuild-index:
	python -m scripts.validate_pdfs
	python -m scripts.ingest_books

check-index:
	python -m scripts.check_index

generate-keys:
	python -m scripts.generate_api_key
