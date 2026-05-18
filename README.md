# Adaptive Core Network (ACN)

Adaptive Core Network is a modular monolith for versioned, reversible and traceable neural training workflows.

The initial scaffold provides the runtime boundaries only:

- FastAPI gateway
- Python worker process
- shared typed Python package
- React + TypeScript frontend
- Docker Compose stack for PostgreSQL, Redis, MLflow and MinIO

Business logic is intentionally not implemented yet.

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

## Training Version Store

Git-like training evolution state lives in `packages/acn/src/acn/versioning`.

It provides:

- stable immutable checkpoint metadata;
- commits with parent-child relationships;
- named branches with independent heads;
- rollback by moving a branch head to a reachable ancestor commit;
- commit graph nodes and edges for future dashboard visualization;
- SQLAlchemy repository implementation backed by PostgreSQL.

Apply database migrations:

```bash
make migrate
```

## Adaptive Controller

The rule-based adaptive controller lives in `packages/acn/src/acn/controller`.

It analyzes training metrics and returns explainable decisions without mutating trainer, versioning or model state directly.

Supported actions:

- rollback;
- decrease or increase learning rate;
- freeze or unfreeze layers;
- create an experimental branch;
- continue training.

Run the controller simulation:

```bash
make simulate-controller
```

The neural adaptive controller adds a PyTorch policy network with offline training,
decision explainability and fallback to the rule-based controller.

Evaluate and compare controllers:

```bash
python scripts/controller/evaluate_neural_policy.py
python scripts/controller/compare_controllers.py
```

## Citadel Safety Layer

The Citadel safety layer lives in `packages/acn/src/acn/citadel`.

It validates critical actions before execution and records audit logs for allowed, denied and override-approved decisions.

It enforces:

- rollback target reachability;
- stable checkpoint immutability;
- unsafe overwrite prevention;
- learning-rate and layer-action policy validation;
- explicit human override approval for supported risky actions.

Run the Citadel simulation:

```bash
make simulate-citadel
```

## Continual Learning Pipeline

The continual learning pipeline lives in `packages/acn/src/acn/continual`.

It provides:

- reusable `IDataSource` abstractions;
- image dataset sources with class filtering;
- future-ready video and camera stream sources;
- asynchronous frame ingestion with configurable sampling;
- synthetic domain shift sources;
- configurable dataset stages;
- incremental class introduction tracking;
- replay buffer support;
- forgetting, retention and adaptation metrics;
- model evaluation utilities that remain decoupled from the trainer.

Example configs live in `configs/continual`.

Run the Fashion-MNIST scenario demo:

```bash
make demo-continual-fashion
```

Typical experiment flow:

1. Build a `ContinualLearningScenario` from stage configs.
2. For each stage, build a stage dataset and optionally mix replay samples.
3. Train with the existing trainer using normal `DataLoader` objects.
4. Evaluate predictions through `ContinualEvaluationPipeline`.
5. Feed metrics into the adaptive controller and route critical actions through Citadel.

Stream ingestion is intentionally lightweight. `VideoFileSource` and `CameraStreamSource`
accept an injected frame reader, fill a `TemporalBuffer` asynchronously, and expose a
snapshot as a regular PyTorch dataset for the existing trainer.

## Experiment Orchestration

Experiment orchestration lives in `packages/acn/src/acn/orchestration`.

It coordinates experiment lifecycle, stage transitions, checkpoint commits, branch creation, rollback and controller decisions while keeping trainer, evaluator and controller decoupled.

Run the orchestration lifecycle example:

```bash
make demo-orchestration
```

Detailed diagrams and event flow are documented in `docs/13_orchestration.md`.

## End-to-End Experiment Pipeline

Run a reproducible continual-learning integration experiment:

```bash
make e2e-experiment
```

The default config is `configs/experiments/acn_e2e_reproducible.json`. It produces metrics,
commit and branch graphs, forgetting/adaptation plots, rollback events, a Markdown report and
an SVG dashboard screenshot under `experiments/acn-e2e-fashion-cifar10c/`.

Run the research benchmark comparison:

```bash
make research-benchmark
```

The benchmark compares standard training, manual tuning, rule-based ACN and neural-controller
ACN, then exports CSV summaries, statistical comparisons, plots and a reproducible report.

## Demo Mode

Run a polished reproducible presentation workflow:

```bash
make demo-mode
```

Generate demo screenshots/assets without starting the web server:

```bash
make demo-assets
```

## Dashboard Frontend

The dashboard frontend lives in `apps/web` and uses React, TypeScript, Tailwind, React Flow and Recharts.

It consumes FastAPI REST endpoints for snapshots and override submissions, then listens for live updates through SSE with WebSocket fallback.

Run the dashboard:

```bash
make web
```

The frontend integration contract is documented in `docs/14_dashboard_frontend.md`.

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
make coverage
```

Python code is checked with Ruff, Black and MyPy in strict mode. Frontend code is checked with TypeScript.
Coverage reports are written under `reports/coverage/`.
