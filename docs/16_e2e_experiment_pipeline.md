# End-to-End Experiment Pipeline

The ACN E2E pipeline runs a reproducible continual-learning integration experiment without
introducing new architecture. It coordinates the existing continual learning, controller,
Citadel and versioning modules and writes portable artifacts for inspection.

## Command

```bash
make e2e-experiment
```

Equivalent direct CLI:

```bash
python scripts/experiments/run_e2e_experiment.py \
  --config configs/experiments/acn_e2e_reproducible.json
```

Use `--output-dir` to redirect generated artifacts.

## Covered Flow

1. Dataset preparation for synthetic Fashion-MNIST and CIFAR-10-C-style stages.
2. Baseline Fashion-MNIST stage.
3. Incremental class introduction.
4. Domain shift introduction.
5. Forgetting and adaptation evaluation.
6. Adaptive controller decisions.
7. Rollback demonstration.
8. Experimental branch creation.
9. Experiment artifact logging.
10. Final report generation.

The default backend is deterministic and synthetic so CI and local development do not depend on
dataset downloads. The dataset names and stage structure mirror the target Fashion-MNIST and
CIFAR-10-C experiment contract.

## Outputs

The default output directory is `experiments/acn-e2e-fashion-cifar10c`.

Generated files:
- `metrics.json`
- `commit_graph.json`
- `branch_graph.json`
- `forgetting_plot.svg`
- `adaptation_plot.svg`
- `rollback_events.json`
- `experiment_summary.json`
- `report.md`
- `dashboard_screenshot.svg`

## Architecture Notes

- The runner lives in `acn.experiments.e2e`.
- Versioning uses the existing SQLAlchemy repository against an in-memory SQLite database.
- Rollback and branch creation route through existing `DecisionExecutor`, `RollbackCoordinator`
  and `CitadelSafetyLayer`.
- Plots and screenshot export are SVG files generated without extra dependencies.
