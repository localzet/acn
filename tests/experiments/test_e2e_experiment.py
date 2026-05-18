import json
from pathlib import Path

from acn.experiments.e2e import load_e2e_config, run_e2e_experiment


def test_e2e_experiment_generates_reproducible_artifacts(tmp_path: Path) -> None:
    config = load_e2e_config(
        Path("configs/experiments/acn_e2e_reproducible.json"),
        output_dir=tmp_path,
    )

    result = run_e2e_experiment(config)

    assert len(result.stages) == 4
    assert len(result.rollback_events) == 1
    assert result.branch_events
    assert len(result.commit_graph.nodes) == 4

    expected_files = (
        result.artifacts.metrics,
        result.artifacts.commit_graph,
        result.artifacts.branch_graph,
        result.artifacts.rollback_events,
        result.artifacts.summary_json,
        result.artifacts.report_markdown,
        result.artifacts.forgetting_plot,
        result.artifacts.adaptation_plot,
        result.artifacts.screenshot_svg,
    )
    for path in expected_files:
        assert path is not None
        assert path.exists()

    summary = json.loads(result.artifacts.summary_json.read_text(encoding="utf-8"))
    metrics = json.loads(result.artifacts.metrics.read_text(encoding="utf-8"))
    rollback_events = json.loads(result.artifacts.rollback_events.read_text(encoding="utf-8"))

    assert summary["experiment_id"] == "acn-e2e-fashion-cifar10c"
    assert summary["stage_count"] == 4
    assert summary["rollback_count"] == 1
    assert {metric["dataset"] for metric in metrics} == {"fashion-mnist", "cifar-10-c"}
    assert rollback_events[0]["action"] == "rollback"
    assert "<svg" in result.artifacts.forgetting_plot.read_text(encoding="utf-8")
    assert "ACN E2E Experiment Report" in result.artifacts.report_markdown.read_text(
        encoding="utf-8"
    )
