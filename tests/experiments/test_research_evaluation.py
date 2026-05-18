import csv
import json
from pathlib import Path

from acn.experiments.research import (
    ResearchBenchmarkConfig,
    StrategyRunMetrics,
    aggregate_strategy_runs,
    compare_strategy_statistics,
    run_research_benchmark,
)


def test_statistical_comparison_selects_expected_best_strategy() -> None:
    runs = (
        StrategyRunMetrics(
            strategy="standard_training",
            run_id=0,
            validation_accuracy=0.70,
            forgetting_score=0.20,
            adaptation_latency=3.0,
            rollback_recovery_rate=0.0,
            branch_success_rate=0.0,
            training_stability=0.60,
        ),
        StrategyRunMetrics(
            strategy="neural_controller_acn",
            run_id=0,
            validation_accuracy=0.82,
            forgetting_score=0.10,
            adaptation_latency=1.0,
            rollback_recovery_rate=0.8,
            branch_success_rate=0.8,
            training_stability=0.80,
        ),
    )

    aggregates = aggregate_strategy_runs(runs)
    comparisons = compare_strategy_statistics(aggregates, baseline_strategy="standard_training")
    by_metric = {comparison.metric: comparison for comparison in comparisons}

    assert by_metric["validation_accuracy"].best_strategy == "neural_controller_acn"
    assert by_metric["forgetting_score"].best_strategy == "neural_controller_acn"
    assert by_metric["validation_accuracy"].delta_vs_baseline["neural_controller_acn"] > 0.0


def test_research_benchmark_generates_exports(tmp_path: Path) -> None:
    config = ResearchBenchmarkConfig(
        benchmark_id="test-research-benchmark",
        e2e_config_path=Path("configs/experiments/acn_e2e_reproducible.json"),
        output_dir=tmp_path,
        repeats=2,
    )

    result = run_research_benchmark(config)

    assert len(result.runs) == 8
    assert len(result.aggregates) == 4
    assert len(result.comparisons) == 6

    for path in (
        result.artifacts.runs_csv,
        result.artifacts.summary_csv,
        result.artifacts.statistics_json,
        result.artifacts.report_markdown,
        result.artifacts.validation_plot,
        result.artifacts.forgetting_plot,
        result.artifacts.stability_plot,
    ):
        assert path.exists()

    with result.artifacts.runs_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    statistics_payload = json.loads(result.artifacts.statistics_json.read_text(encoding="utf-8"))

    assert len(rows) == 8
    assert {row["strategy"] for row in rows} == {
        "standard_training",
        "manual_tuning",
        "rule_based_acn",
        "neural_controller_acn",
    }
    assert statistics_payload[0]["metric"] == "validation_accuracy"
    assert "ACN Research Benchmark Report" in result.artifacts.report_markdown.read_text(
        encoding="utf-8"
    )
