import json
from pathlib import Path
from typing import Any

import torch
from fastapi.testclient import TestClient
from torch import Tensor
from torch.utils.data import Dataset

from acn.config.settings import Settings
from acn.experiments.real_vertical import RealVerticalSliceConfig, run_real_vertical_slice
from acn_api.main import create_app


class _TinyFashionDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, *, samples: int) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return self._samples

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        label = index % 10
        image = torch.full((1, 28, 28), float(label) / 9.0)
        image[:, label : label + 3, label : label + 3] = 1.0
        return image, label


def test_real_vertical_slice_runs_rollback_and_writes_dashboard_snapshot(tmp_path: Path) -> None:
    config = RealVerticalSliceConfig(
        output_dir=tmp_path,
        max_train_samples=80,
        max_validation_samples=40,
        batch_size=20,
        spike_learning_rate=0.75,
        device="cpu",
    )

    result = run_real_vertical_slice(config, dataset_factory=_dataset_factory)

    assert result.rollback_restored is True
    assert result.baseline_commit_id != result.degraded_commit_id
    assert result.recovery_commit_id != result.baseline_commit_id
    assert result.artifacts.metrics_json.exists()
    assert result.artifacts.dashboard_snapshot_json.exists()
    assert result.artifacts.rollback_events_json.exists()
    assert result.artifacts.validation_plot_svg.exists()
    assert result.artifacts.forgetting_plot_svg.exists()
    assert result.artifacts.adaptation_plot_svg.exists()
    assert result.artifacts.rollback_report_markdown.exists()

    snapshot = json.loads(result.artifacts.dashboard_snapshot_json.read_text(encoding="utf-8"))
    assert len(snapshot["commitGraph"]["nodes"]) == 3
    assert snapshot["branchGraph"]["nodes"][0]["headCommitId"] == result.recovery_commit_id
    assert snapshot["controllerDecisions"][0]["action"] == "rollback"
    assert snapshot["rollbackHistory"][0]["toCommitId"] == result.baseline_commit_id
    assert len(snapshot["metricsTimeline"]) == 3


def test_dashboard_snapshot_reads_real_vertical_slice_telemetry(tmp_path: Path) -> None:
    config = RealVerticalSliceConfig(
        output_dir=tmp_path,
        max_train_samples=40,
        max_validation_samples=20,
        batch_size=20,
        device="cpu",
    )
    result = run_real_vertical_slice(config, dataset_factory=_dataset_factory)
    client = TestClient(
        create_app(
            Settings(
                env="test",
                dashboard_telemetry_path=result.artifacts.dashboard_snapshot_json,
            )
        )
    )

    response = client.get("/api/v1/dashboard/snapshot")

    assert response.status_code == 200
    payload = response.json()
    assert payload["experiments"][0]["id"] == config.experiment_id
    assert payload["rollbackHistory"][0]["fromCommitId"] == result.degraded_commit_id
    assert payload["controllerDecisions"][0]["action"] == "rollback"


def _dataset_factory(config: RealVerticalSliceConfig) -> tuple[Dataset[Any], Dataset[Any]]:
    return (
        _TinyFashionDataset(samples=config.max_train_samples),
        _TinyFashionDataset(samples=config.max_validation_samples),
    )
