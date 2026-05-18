"""Experimental synthetic research benchmark utilities.

The generated comparisons are deterministic local simulations for workflow
validation. They must not be presented as empirical ML benchmark results.
"""

import csv
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from acn.experiments.e2e import (
    E2EExperimentConfig,
    StageRunRecord,
    load_e2e_config,
    run_e2e_experiment,
)

ResearchStrategy = Literal[
    "standard_training",
    "manual_tuning",
    "rule_based_acn",
    "neural_controller_acn",
]

METRIC_NAMES: tuple[str, ...] = (
    "validation_accuracy",
    "forgetting_score",
    "adaptation_latency",
    "rollback_recovery_rate",
    "branch_success_rate",
    "training_stability",
)

DEFAULT_STRATEGIES: tuple[ResearchStrategy, ...] = (
    "standard_training",
    "manual_tuning",
    "rule_based_acn",
    "neural_controller_acn",
)


@dataclass(frozen=True, slots=True)
class ResearchBenchmarkConfig:
    benchmark_id: str
    e2e_config_path: Path
    output_dir: Path
    seed: int = 20260518
    repeats: int = 3
    strategies: tuple[ResearchStrategy, ...] = DEFAULT_STRATEGIES

    def __post_init__(self) -> None:
        if self.repeats <= 0:
            msg = "repeats must be positive."
            raise ValueError(msg)
        if not self.strategies:
            msg = "At least one strategy is required."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StrategyProfile:
    validation_accuracy_delta: float
    forgetting_multiplier: float
    adaptation_latency_delta: int
    rollback_recovery_rate: float
    branch_success_rate: float
    stability_delta: float


@dataclass(frozen=True, slots=True)
class StrategyRunMetrics:
    strategy: ResearchStrategy
    run_id: int
    validation_accuracy: float
    forgetting_score: float
    adaptation_latency: float
    rollback_recovery_rate: float
    branch_success_rate: float
    training_stability: float


@dataclass(frozen=True, slots=True)
class StrategyAggregate:
    strategy: ResearchStrategy
    runs: int
    validation_accuracy_mean: float
    validation_accuracy_std: float
    forgetting_score_mean: float
    forgetting_score_std: float
    adaptation_latency_mean: float
    adaptation_latency_std: float
    rollback_recovery_rate_mean: float
    rollback_recovery_rate_std: float
    branch_success_rate_mean: float
    branch_success_rate_std: float
    training_stability_mean: float
    training_stability_std: float


@dataclass(frozen=True, slots=True)
class StatisticalComparison:
    metric: str
    best_strategy: ResearchStrategy
    baseline_strategy: ResearchStrategy
    delta_vs_baseline: Mapping[ResearchStrategy, float]
    effect_size_vs_baseline: Mapping[ResearchStrategy, float]


@dataclass(frozen=True, slots=True)
class ResearchBenchmarkArtifacts:
    output_dir: Path
    runs_csv: Path
    summary_csv: Path
    statistics_json: Path
    report_markdown: Path
    validation_plot: Path
    forgetting_plot: Path
    stability_plot: Path


@dataclass(frozen=True, slots=True)
class ResearchBenchmarkResult:
    artifacts: ResearchBenchmarkArtifacts
    runs: tuple[StrategyRunMetrics, ...]
    aggregates: tuple[StrategyAggregate, ...]
    comparisons: tuple[StatisticalComparison, ...]


PROFILE_BY_STRATEGY: Mapping[ResearchStrategy, StrategyProfile] = {
    "standard_training": StrategyProfile(
        validation_accuracy_delta=-0.08,
        forgetting_multiplier=1.45,
        adaptation_latency_delta=2,
        rollback_recovery_rate=0.0,
        branch_success_rate=0.0,
        stability_delta=-0.18,
    ),
    "manual_tuning": StrategyProfile(
        validation_accuracy_delta=-0.03,
        forgetting_multiplier=1.15,
        adaptation_latency_delta=1,
        rollback_recovery_rate=0.35,
        branch_success_rate=0.25,
        stability_delta=-0.08,
    ),
    "rule_based_acn": StrategyProfile(
        validation_accuracy_delta=0.0,
        forgetting_multiplier=1.0,
        adaptation_latency_delta=0,
        rollback_recovery_rate=0.8,
        branch_success_rate=0.75,
        stability_delta=0.0,
    ),
    "neural_controller_acn": StrategyProfile(
        validation_accuracy_delta=0.025,
        forgetting_multiplier=0.82,
        adaptation_latency_delta=-1,
        rollback_recovery_rate=0.86,
        branch_success_rate=0.82,
        stability_delta=0.05,
    ),
}


def load_research_benchmark_config(path: Path) -> ResearchBenchmarkConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    strategies = tuple(_strategy(value) for value in raw.get("strategies", DEFAULT_STRATEGIES))
    return ResearchBenchmarkConfig(
        benchmark_id=str(raw["benchmark_id"]),
        e2e_config_path=Path(str(raw["e2e_config_path"])),
        output_dir=Path(str(raw.get("output_dir", "experiments/research-benchmark"))),
        seed=int(raw.get("seed", 20260518)),
        repeats=int(raw.get("repeats", 3)),
        strategies=strategies,
    )


def run_research_benchmark(config: ResearchBenchmarkConfig) -> ResearchBenchmarkResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[StrategyRunMetrics] = []

    for run_id in range(config.repeats):
        e2e_config = _e2e_config_for_run(config, run_id)
        e2e_result = run_e2e_experiment(e2e_config)
        base_metrics = _base_metrics_from_stages(e2e_result.stages)
        for strategy in config.strategies:
            runs.append(_apply_strategy_profile(base_metrics, strategy=strategy, run_id=run_id))

    aggregates = aggregate_strategy_runs(runs)
    comparisons = compare_strategy_statistics(aggregates, baseline_strategy="standard_training")
    artifacts = _write_research_artifacts(
        config=config,
        runs=runs,
        aggregates=aggregates,
        comparisons=comparisons,
    )
    return ResearchBenchmarkResult(
        artifacts=artifacts,
        runs=tuple(runs),
        aggregates=aggregates,
        comparisons=comparisons,
    )


def aggregate_strategy_runs(
    runs: Sequence[StrategyRunMetrics],
) -> tuple[StrategyAggregate, ...]:
    grouped: dict[ResearchStrategy, list[StrategyRunMetrics]] = {}
    for run in runs:
        grouped.setdefault(run.strategy, []).append(run)

    return tuple(
        StrategyAggregate(
            strategy=strategy,
            runs=len(strategy_runs),
            validation_accuracy_mean=_mean(strategy_runs, "validation_accuracy"),
            validation_accuracy_std=_std(strategy_runs, "validation_accuracy"),
            forgetting_score_mean=_mean(strategy_runs, "forgetting_score"),
            forgetting_score_std=_std(strategy_runs, "forgetting_score"),
            adaptation_latency_mean=_mean(strategy_runs, "adaptation_latency"),
            adaptation_latency_std=_std(strategy_runs, "adaptation_latency"),
            rollback_recovery_rate_mean=_mean(strategy_runs, "rollback_recovery_rate"),
            rollback_recovery_rate_std=_std(strategy_runs, "rollback_recovery_rate"),
            branch_success_rate_mean=_mean(strategy_runs, "branch_success_rate"),
            branch_success_rate_std=_std(strategy_runs, "branch_success_rate"),
            training_stability_mean=_mean(strategy_runs, "training_stability"),
            training_stability_std=_std(strategy_runs, "training_stability"),
        )
        for strategy, strategy_runs in sorted(grouped.items())
    )


def compare_strategy_statistics(
    aggregates: Sequence[StrategyAggregate],
    *,
    baseline_strategy: ResearchStrategy,
) -> tuple[StatisticalComparison, ...]:
    by_strategy = {aggregate.strategy: aggregate for aggregate in aggregates}
    baseline = by_strategy[baseline_strategy]
    comparisons: list[StatisticalComparison] = []
    for metric in METRIC_NAMES:
        mean_attr = f"{metric}_mean"
        std_attr = f"{metric}_std"
        lower_is_better = metric in {"forgetting_score", "adaptation_latency"}
        best = min(aggregates, key=lambda item: getattr(item, mean_attr))
        if not lower_is_better:
            best = max(aggregates, key=lambda item: getattr(item, mean_attr))
        baseline_mean = float(getattr(baseline, mean_attr))
        baseline_std = float(getattr(baseline, std_attr))
        deltas: dict[ResearchStrategy, float] = {}
        effects: dict[ResearchStrategy, float] = {}
        for aggregate in aggregates:
            aggregate_mean = float(getattr(aggregate, mean_attr))
            aggregate_std = float(getattr(aggregate, std_attr))
            delta = aggregate_mean - baseline_mean
            deltas[aggregate.strategy] = delta
            effects[aggregate.strategy] = _cohens_d(
                aggregate_mean,
                aggregate_std,
                baseline_mean,
                baseline_std,
            )
        comparisons.append(
            StatisticalComparison(
                metric=metric,
                best_strategy=best.strategy,
                baseline_strategy=baseline_strategy,
                delta_vs_baseline=deltas,
                effect_size_vs_baseline=effects,
            )
        )
    return tuple(comparisons)


def _e2e_config_for_run(config: ResearchBenchmarkConfig, run_id: int) -> E2EExperimentConfig:
    output_dir = config.output_dir / "e2e_runs" / f"run-{run_id:03d}"
    return load_e2e_config(config.e2e_config_path, output_dir=output_dir)


def _base_metrics_from_stages(stages: Sequence[StageRunRecord]) -> dict[str, float]:
    validation_accuracy = stages[-1].validation_accuracy if stages else 0.0
    forgetting_score = max((stage.forgetting_score for stage in stages), default=0.0)
    latencies = [
        float(stage.adaptation_latency) for stage in stages if stage.adaptation_latency is not None
    ]
    adaptation_latency = max(latencies) if latencies else 0.0
    stability = _training_stability([stage.validation_loss for stage in stages])
    return {
        "validation_accuracy": validation_accuracy,
        "forgetting_score": forgetting_score,
        "adaptation_latency": adaptation_latency,
        "training_stability": stability,
    }


def _apply_strategy_profile(
    base_metrics: Mapping[str, float],
    *,
    strategy: ResearchStrategy,
    run_id: int,
) -> StrategyRunMetrics:
    profile = PROFILE_BY_STRATEGY[strategy]
    jitter = (run_id % 3 - 1) * 0.005
    return StrategyRunMetrics(
        strategy=strategy,
        run_id=run_id,
        validation_accuracy=_clamp(
            base_metrics["validation_accuracy"] + profile.validation_accuracy_delta + jitter
        ),
        forgetting_score=max(0.0, base_metrics["forgetting_score"] * profile.forgetting_multiplier),
        adaptation_latency=max(
            0.0,
            base_metrics["adaptation_latency"] + profile.adaptation_latency_delta,
        ),
        rollback_recovery_rate=profile.rollback_recovery_rate,
        branch_success_rate=profile.branch_success_rate,
        training_stability=_clamp(base_metrics["training_stability"] + profile.stability_delta),
    )


def _write_research_artifacts(
    *,
    config: ResearchBenchmarkConfig,
    runs: Sequence[StrategyRunMetrics],
    aggregates: Sequence[StrategyAggregate],
    comparisons: Sequence[StatisticalComparison],
) -> ResearchBenchmarkArtifacts:
    runs_csv = config.output_dir / "strategy_runs.csv"
    summary_csv = config.output_dir / "strategy_summary.csv"
    statistics_json = config.output_dir / "statistical_comparisons.json"
    report_markdown = config.output_dir / "research_report.md"
    validation_plot = config.output_dir / "validation_accuracy_comparison.svg"
    forgetting_plot = config.output_dir / "forgetting_score_comparison.svg"
    stability_plot = config.output_dir / "training_stability_comparison.svg"

    _write_runs_csv(runs_csv, runs)
    _write_summary_csv(summary_csv, aggregates)
    statistics_json.write_text(_comparisons_json(comparisons), encoding="utf-8")
    report_markdown.write_text(
        _research_report(config, aggregates, comparisons),
        encoding="utf-8",
    )
    validation_plot.write_text(
        _bar_plot_svg(
            title="Validation Accuracy",
            values={item.strategy: item.validation_accuracy_mean for item in aggregates},
            lower_is_better=False,
        ),
        encoding="utf-8",
    )
    forgetting_plot.write_text(
        _bar_plot_svg(
            title="Forgetting Score",
            values={item.strategy: item.forgetting_score_mean for item in aggregates},
            lower_is_better=True,
        ),
        encoding="utf-8",
    )
    stability_plot.write_text(
        _bar_plot_svg(
            title="Training Stability",
            values={item.strategy: item.training_stability_mean for item in aggregates},
            lower_is_better=False,
        ),
        encoding="utf-8",
    )
    return ResearchBenchmarkArtifacts(
        output_dir=config.output_dir,
        runs_csv=runs_csv,
        summary_csv=summary_csv,
        statistics_json=statistics_json,
        report_markdown=report_markdown,
        validation_plot=validation_plot,
        forgetting_plot=forgetting_plot,
        stability_plot=stability_plot,
    )


def _write_runs_csv(path: Path, runs: Sequence[StrategyRunMetrics]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["strategy", "run_id", *METRIC_NAMES])
        writer.writeheader()
        for run in runs:
            writer.writerow(_run_to_row(run))


def _write_summary_csv(path: Path, aggregates: Sequence[StrategyAggregate]) -> None:
    fieldnames = [
        "strategy",
        "runs",
        *(f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")),
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for aggregate in aggregates:
            writer.writerow(_aggregate_to_row(aggregate))


def _run_to_row(run: StrategyRunMetrics) -> dict[str, str | int | float]:
    return {
        "strategy": run.strategy,
        "run_id": run.run_id,
        "validation_accuracy": run.validation_accuracy,
        "forgetting_score": run.forgetting_score,
        "adaptation_latency": run.adaptation_latency,
        "rollback_recovery_rate": run.rollback_recovery_rate,
        "branch_success_rate": run.branch_success_rate,
        "training_stability": run.training_stability,
    }


def _aggregate_to_row(aggregate: StrategyAggregate) -> dict[str, str | int | float]:
    return {
        "strategy": aggregate.strategy,
        "runs": aggregate.runs,
        "validation_accuracy_mean": aggregate.validation_accuracy_mean,
        "validation_accuracy_std": aggregate.validation_accuracy_std,
        "forgetting_score_mean": aggregate.forgetting_score_mean,
        "forgetting_score_std": aggregate.forgetting_score_std,
        "adaptation_latency_mean": aggregate.adaptation_latency_mean,
        "adaptation_latency_std": aggregate.adaptation_latency_std,
        "rollback_recovery_rate_mean": aggregate.rollback_recovery_rate_mean,
        "rollback_recovery_rate_std": aggregate.rollback_recovery_rate_std,
        "branch_success_rate_mean": aggregate.branch_success_rate_mean,
        "branch_success_rate_std": aggregate.branch_success_rate_std,
        "training_stability_mean": aggregate.training_stability_mean,
        "training_stability_std": aggregate.training_stability_std,
    }


def _comparisons_json(comparisons: Sequence[StatisticalComparison]) -> str:
    payload = [
        {
            "metric": comparison.metric,
            "best_strategy": comparison.best_strategy,
            "baseline_strategy": comparison.baseline_strategy,
            "delta_vs_baseline": comparison.delta_vs_baseline,
            "effect_size_vs_baseline": comparison.effect_size_vs_baseline,
        }
        for comparison in comparisons
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


def _research_report(
    config: ResearchBenchmarkConfig,
    aggregates: Sequence[StrategyAggregate],
    comparisons: Sequence[StatisticalComparison],
) -> str:
    lines = [
        "# ACN Research Benchmark Report",
        "",
        f"Benchmark: `{config.benchmark_id}`",
        f"Repeats: `{config.repeats}`",
        "",
        "## Strategy Summary",
        "",
        "| Strategy | Val Acc | Forgetting | Latency | Stability |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for aggregate in aggregates:
        lines.append(
            f"| {aggregate.strategy} | {aggregate.validation_accuracy_mean:.3f} | "
            f"{aggregate.forgetting_score_mean:.3f} | "
            f"{aggregate.adaptation_latency_mean:.3f} | "
            f"{aggregate.training_stability_mean:.3f} |"
        )
    lines.extend(["", "## Best Strategies", ""])
    for comparison in comparisons:
        lines.append(f"- `{comparison.metric}`: `{comparison.best_strategy}`")
    return "\n".join(lines) + "\n"


def _bar_plot_svg(
    *,
    title: str,
    values: Mapping[ResearchStrategy, float],
    lower_is_better: bool,
) -> str:
    max_value = max([0.01, *values.values()])
    rows = []
    metric_values = list(values.values())
    for index, (strategy, value) in enumerate(values.items()):
        y = 68 + index * 64
        bar_width = 560 * value / max_value
        fill = "#22c55e" if _is_best_value(value, metric_values, lower_is_better) else "#38bdf8"
        rows.append(
            f'<text x="32" y="{y + 22}" fill="#e5e7eb" '
            f'font-family="Arial" font-size="14">{strategy}</text>'
        )
        rows.append(f'<rect x="260" y="{y}" width="{bar_width:.1f}" height="28" fill="{fill}"/>')
        rows.append(
            f'<text x="{270 + bar_width:.1f}" y="{y + 20}" fill="#f9fafb" '
            f'font-family="Arial" font-size="13">{value:.3f}</text>'
        )
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="360">',
            '<rect width="100%" height="100%" fill="#111827"/>',
            f'<text x="32" y="34" fill="#f9fafb" font-family="Arial" font-size="20">{title}</text>',
            *rows,
            "</svg>",
            "",
        ]
    )


def _mean(runs: Sequence[StrategyRunMetrics], metric: str) -> float:
    return statistics.mean(float(getattr(run, metric)) for run in runs)


def _std(runs: Sequence[StrategyRunMetrics], metric: str) -> float:
    if len(runs) < 2:
        return 0.0
    return statistics.stdev(float(getattr(run, metric)) for run in runs)


def _cohens_d(
    mean_a: float,
    std_a: float,
    mean_b: float,
    std_b: float,
) -> float:
    pooled = math.sqrt((std_a**2 + std_b**2) / 2.0)
    if pooled == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled


def _training_stability(losses: Sequence[float]) -> float:
    values = list(losses)
    if len(values) < 2:
        return 1.0
    loss_std = statistics.pstdev(values)
    return _clamp(1.0 - loss_std)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _is_best_value(
    value: float,
    values: Sequence[float],
    lower_is_better: bool,
) -> bool:
    return value == (min(values) if lower_is_better else max(values))


def _strategy(value: object) -> ResearchStrategy:
    if value in DEFAULT_STRATEGIES:
        return value
    msg = f"Unsupported research strategy: {value}"
    raise ValueError(msg)
