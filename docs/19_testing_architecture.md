# Testing Architecture

ACN uses pytest as the primary test runner. Tests are isolated, deterministic and organized by
subsystem.

## Test Layers

- `tests/unit`: configuration and small pure units.
- `tests/training`: trainer, checkpointing, optimizer and freezing behavior.
- `tests/versioning`: commit graph, branches, rollback and checkpoint immutability.
- `tests/citadel`: safety validation and audit logging.
- `tests/controller`: rule-based and neural-controller decisions.
- `tests/continual`: dataset sources, replay buffer, stream ingestion and forgetting metrics.
- `tests/orchestration`: experiment lifecycle, decision execution and rollback coordination.
- `tests/api`: FastAPI health and dashboard integration contract.
- `tests/worker`: worker startup and logging behavior.
- `tests/experiments`: E2E and research benchmark artifact generation.
- `tests/demo`: demo asset and screenshot export generation.
- `tests/integration`: cross-module branch and rollback consistency.

## Commands

Run all tests:

```bash
make test
```

Run coverage reports:

```bash
make coverage
```

Coverage outputs:
- terminal missing-line report;
- `reports/coverage/html`;
- `reports/coverage/coverage.xml`.

## Reproducibility

- Database tests use in-memory SQLite sessions.
- Experiment and benchmark tests use deterministic synthetic presets.
- Replay buffer tests pin RNG seeds.
- API and worker tests run in-process without external services.
