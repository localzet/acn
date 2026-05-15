# Adaptive Core Network (ACN)

Adaptive Core Network is a modular monolith for versioned, reversible and traceable neural training workflows.

The initial scaffold provides the runtime boundaries only:

- FastAPI gateway
- Python worker process
- shared typed Python package
- React + TypeScript frontend
- Docker Compose stack for PostgreSQL, Redis, MLflow and MinIO

Business logic is intentionally not implemented yet.

## Repository Layout

```text
.
├── apps/
│   ├── api/
│   ├── web/
│   └── worker/
├── packages/
│   └── acn/
├── infra/
│   └── docker/
├── docs/
├── experiments/
├── checkpoints/
├── tests/
├── .env.example
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## Architecture

ACN starts as a modular monolith with explicit application boundaries:

- `apps/api` owns HTTP concerns and depends on shared application services through `packages/acn`.
- `apps/worker` owns asynchronous execution concerns and consumes the same shared package.
- `packages/acn` is the internal Python package for configuration, domain contracts, services and infrastructure adapters.
- `apps/web` is isolated as a TypeScript frontend and talks to the API over HTTP.
- `infra/docker` contains runtime images while `docker-compose.yml` wires local infrastructure.

This keeps the system deployable as a small local stack while preserving clean seams for trainer, version store, controller and experiment tracking modules.

## Trainer Core

The trainer core lives in `packages/acn/src/acn/training` and is independent from API, worker and UI runtimes.

It provides:

- typed trainer configuration;
- configurable optimizers and schedulers;
- mixed precision on CUDA devices;
- checkpoint save/load;
- layer freeze/unfreeze helpers;
- train and validation loops.

Run the lightweight Fashion-MNIST example:

```bash
make train-fashion-mnist
```

The example stores downloaded datasets under `data/` and checkpoints under `checkpoints/fashion-mnist/`.

## Requirements

- Python 3.12
- Node.js 20+
- Docker Compose v2

On Debian or Ubuntu, install `python3.12-venv` if `python3.12 -m venv` is unavailable.

## Setup

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
make install
make check
```

Run the full local stack:

```bash
make compose-up
```

Local endpoints:

- API: <http://localhost:8000>
- API health: <http://localhost:8000/health>
- Frontend: <http://localhost:5173>
- MLflow: <http://localhost:5000>
- MinIO console: <http://localhost:9001>

Run services without Docker:

```bash
make api
make worker
make web
```

## Development

Quality gates:

```bash
make lint
make type-check
make test
```

Python code is checked with Ruff, Black and MyPy in strict mode. Frontend code is checked with TypeScript.
