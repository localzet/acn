# Research Evaluation Utilities

ACN research evaluation compares adaptive strategies against non-adaptive baselines using the
existing reproducible E2E experiment scenario.

## Compared Strategies

- Standard training
- Manual tuning
- Rule-based ACN
- Neural-controller ACN

## Metrics

- validation accuracy
- forgetting score
- adaptation latency
- rollback recovery rate
- branch success rate
- training stability

## Command

```bash
make research-benchmark
```

Equivalent direct CLI:

```bash
python scripts/experiments/run_research_benchmark.py \
  --config configs/experiments/research_benchmark.json
```

## Outputs

The default output directory is `experiments/acn-research-baseline-comparison`.

Generated files:
- `strategy_runs.csv`
- `strategy_summary.csv`
- `statistical_comparisons.json`
- `research_report.md`
- `validation_accuracy_comparison.svg`
- `forgetting_score_comparison.svg`
- `training_stability_comparison.svg`

## Design Notes

- The benchmark runner lives in `acn.experiments.research`.
- Statistical utilities operate on typed run records and aggregates.
- Plots are dependency-free SVG exports.
- The default benchmark uses deterministic strategy profiles over the E2E experiment baseline,
  so research reports are reproducible in CI and on local machines.
