.PHONY: setup test run clean format lint download-policies

setup:
	python3 -m venv venv
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install -r requirements-dev.txt

test:
	pytest -v

run:
	python -m src.cli

format:
	ruff check --fix .
	ruff format .

lint:
	ruff check .
	ruff format --check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

download-policies:
	python scripts/download_policies.py

download-policies-test:
	python scripts/download_policies.py --max-files 5