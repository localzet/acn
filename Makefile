.PHONY: install lint format type-check test check compose-up compose-down api worker web train-fashion-mnist

PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev,ml]"
	cd apps/web && npm install

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check . --fix

type-check:
	$(PYTHON) -m mypy
	cd apps/web && npm run typecheck

test:
	$(PYTHON) -m pytest

check: lint type-check test

compose-up:
	docker compose up --build

compose-down:
	docker compose down

api:
	$(PYTHON) -m uvicorn acn_api.main:create_app --factory --host 0.0.0.0 --port 8000 --reload

worker:
	$(PYTHON) -m acn_worker.main

web:
	cd apps/web && npm run dev

train-fashion-mnist:
	$(PYTHON) scripts/train_fashion_mnist.py
