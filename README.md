# Adaptive Core Network (ACN)

Adaptive Core Network is a modular monolith for versioned, reversible and traceable neural
training workflows. The current platform is a local research system targeted at a single
workstation with Python 3.12, PyTorch, FastAPI, PostgreSQL-compatible repositories, Docker
Compose services and a React dashboard.

ACN currently supports:

- PyTorch trainer core with checkpointing, mixed precision, optimizers and schedulers;
- local artifact lifecycle with checksum-validated checkpoint save/load;
- Git-like training versioning with commits, branches, rollback and commit graph records;
- Citadel safety checks for critical actions;
- rule-based adaptive controller as the stable local decision path;
- experimental neural controller with rule-based fallback;
- continual-learning abstractions, replay buffer and forgetting/retention metrics;
- synchronous orchestration for Stage 1 local execution;
- real Fashion-MNIST vertical slice with checkpoint restoration and dashboard telemetry;
- React dashboard with REST snapshot plus SSE/WebSocket update contracts.

## Requirements

- Python 3.12
- Node.js 20+
- Docker Compose v2
- Optional CUDA-capable PyTorch environment for GPU runs

On Debian or Ubuntu, install `python3.12-venv` if `python3.12 -m venv` is unavailable.

## Setup

```bash
cp .env.example .env
python3.12 -m venv .venv
source .venv/bin/activate
make install
```

Run quality gates:

```bash
make lint
make type-check
make test
```

## Real Adaptive Vertical Slice

Run the first real end-to-end adaptive continual-learning milestone:

```bash
make real-vertical-slice
```

This runs:

1. Fashion-MNIST baseline training.
2. Checkpoint artifact save.
3. Version commit on `main`.
4. Intentional degraded stage with learning-rate spike and corrupted samples.
5. Real metric evaluation.
6. Rule-based degradation detection.
7. Citadel-protected rollback.
8. Real checkpoint restoration.
9. Continued recovery training.
10. Dashboard telemetry/report generation.

Default outputs are written to:

```text
experiments/acn-real-fashion-mnist-rollback/
```

Important generated files:

- `dashboard_snapshot.json`
- `metrics.json`
- `rollback_events.json`
- `report.md`
- `rollback_report.md`
- `validation_plot.svg`
- `forgetting_plot.svg`
- `adaptation_plot.svg`
- `experiment.db`
- `artifacts/checkpoints/...`

## Dashboard With Real Telemetry

After running `make real-vertical-slice`, expose that telemetry through the API:

```bash
export ACN_DASHBOARD_TELEMETRY_PATH=experiments/acn-real-fashion-mnist-rollback/dashboard_snapshot.json
make api
```

In another terminal:

```bash
make web
```

Open:

- Frontend: <http://localhost:5173>
- API health: <http://localhost:8000/health>
- Dashboard snapshot: <http://localhost:8000/api/v1/dashboard/snapshot>
- SSE stream: <http://localhost:8000/api/v1/dashboard/events>

If the frontend cannot reach the API, set:

```bash
export VITE_API_BASE_URL=http://localhost:8000
```

from inside `apps/web` before running `npm run dev`, or keep the default local setup.

## Docker Compose Stack

Start local infrastructure and app services:

```bash
make compose-up
```

Endpoints:

- API: <http://localhost:8000>
- Frontend: <http://localhost:5173>
- MLflow: <http://localhost:5000>
- MinIO console: <http://localhost:9001>

Stop:

```bash
make compose-down
```

## Other Workflows

Trainer example:

```bash
make train-fashion-mnist
```

Rule-based controller simulation:

```bash
make simulate-controller
```

Citadel simulation:

```bash
make simulate-citadel
```

Continual-learning scenario demo:

```bash
make demo-continual-fashion
```

Synchronous orchestration lifecycle example:

```bash
make demo-orchestration
```

Synthetic deterministic E2E utility:

```bash
make e2e-experiment
```

Synthetic research benchmark utility:

```bash
make research-benchmark
```

Presentation demo mode:

```bash
make demo-mode
```

## Architecture Boundaries

- `acn.training` must not import API/UI/orchestration managers.
- `acn.orchestration` coordinates domains and must not own PyTorch training implementation.
- `acn.versioning` owns commits, branches and checkpoint metadata only.
- `acn.artifacts` owns checkpoint artifact persistence.
- `acn.experiments.real_vertical` is a milestone script that composes core modules.
- `acn.experiments.e2e` and `acn.experiments.research` are synthetic utilities, not empirical ML benchmarks.
- `acn.controller.neural` is experimental until trained and calibrated on real policy data.

Architecture guardrails live in `tests/architecture/test_architecture_guardrails.py`.

## Development

Quality gates:

```bash
make lint
make type-check
make test
make coverage
```

Python uses Ruff, Black and strict MyPy. Frontend uses TypeScript, Tailwind, React Flow and Recharts.
Generated datasets, checkpoints, experiment outputs, coverage reports and local databases are ignored
by git.
