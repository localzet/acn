.PHONY: install lint format type-check test check migrate compose-up compose-down api worker web train-fashion-mnist simulate-controller simulate-citadel demo-continual-fashion demo-orchestration e2e-experiment research-benchmark demo-assets demo-mode

PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev,ml]" --break-system-packages
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

migrate:
	$(PYTHON) -m alembic upgrade head

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

simulate-controller:
	$(PYTHON) scripts/controller/simulate_rule_based_controller.py

simulate-citadel:
	$(PYTHON) scripts/citadel/simulate_citadel.py

demo-continual-fashion:
	$(PYTHON) scripts/continual/fashion_mnist_demo_scenario.py

demo-orchestration:
	$(PYTHON) scripts/orchestration/example_experiment_lifecycle.py

e2e-experiment:
	$(PYTHON) scripts/experiments/run_e2e_experiment.py

research-benchmark:
	$(PYTHON) scripts/experiments/run_research_benchmark.py

demo-assets:
	$(PYTHON) scripts/demo/generate_demo_assets.py

demo-mode: demo-assets
	cd apps/web && VITE_DEMO_MODE=true npm run dev
