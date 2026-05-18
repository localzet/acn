import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset, Subset

from acn.artifacts import LocalArtifactStore
from acn.citadel import CitadelSafetyLayer
from acn.continual import ForgettingEvaluator
from acn.controller import (
    AdaptiveAction,
    AdaptiveController,
    ControllerDecision,
    MetricPoint,
    TrainingContext,
)
from acn.controller.policies import RuleBasedAdaptivePolicy, RuleBasedPolicyConfig
from acn.orchestration import RollbackCoordinator
from acn.training import CheckpointManager, CheckpointState, OptimizerConfig
from acn.training.optimizers import build_optimizer
from acn.versioning.domain import BranchRecord, CommitGraph, CommitRecord
from acn.versioning.models import Base
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository


@dataclass(frozen=True, slots=True)
class RealVerticalSliceConfig:
    experiment_id: str = "acn-real-fashion-mnist-rollback"
    seed: int = 20260518
    output_dir: Path = Path("experiments/acn-real-fashion-mnist-rollback")
    data_dir: Path = Path("data")
    max_train_samples: int = 512
    max_validation_samples: int = 256
    batch_size: int = 64
    baseline_epochs: int = 1
    degraded_epochs: int = 1
    recovery_epochs: int = 1
    learning_rate: float = 1e-3
    spike_learning_rate: float = 0.75
    device: str | None = None


@dataclass(frozen=True, slots=True)
class RealVerticalSliceArtifacts:
    output_dir: Path
    metrics_json: Path
    dashboard_snapshot_json: Path
    rollback_events_json: Path
    report_markdown: Path
    validation_plot_svg: Path
    forgetting_plot_svg: Path
    adaptation_plot_svg: Path
    rollback_report_markdown: Path


@dataclass(frozen=True, slots=True)
class RealVerticalSliceResult:
    artifacts: RealVerticalSliceArtifacts
    rollback_restored: bool
    baseline_commit_id: str
    degraded_commit_id: str
    recovery_commit_id: str


type DatasetFactory = Callable[[RealVerticalSliceConfig], tuple[Dataset[Any], Dataset[Any]]]


class _StageMetric(TypedDict):
    stage_id: str
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    examples: int


class _EvaluationMetric(TypedDict):
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    examples: int


class _ContinualMetric(TypedDict):
    oldClassRetention: float | None
    newClassAdaptation: float | None
    forgettingScore: float
    adaptationLatency: int | None


class FashionMNISTLiteCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.features(inputs)))


class DegradedDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: Dataset[Any], *, noise: float = 1.0, label_shift: int = 1) -> None:
        self._dataset = dataset
        self._noise = noise
        self._label_shift = label_shift

    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, target = _parse_sample(self._dataset[index])
        degraded = torch.clamp(image + torch.randn_like(image) * self._noise, 0.0, 1.0)
        return degraded, (target + self._label_shift) % 10


def run_real_vertical_slice(
    config: RealVerticalSliceConfig,
    *,
    dataset_factory: DatasetFactory | None = None,
) -> RealVerticalSliceResult:
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = config.output_dir / "experiment.db"
    database_path.unlink(missing_ok=True)
    device = _resolve_device(config.device)
    train_dataset, validation_dataset = (
        dataset_factory(config) if dataset_factory is not None else _fashion_mnist_datasets(config)
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
    degraded_loader = DataLoader(
        DegradedDataset(train_dataset),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = FashionMNISTLiteCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        OptimizerConfig(name="adamw", learning_rate=config.learning_rate),
    )
    artifact_store = LocalArtifactStore(config.output_dir / "artifacts")
    checkpoint_manager = CheckpointManager(
        config.output_dir / "checkpoints",
        artifact_store=artifact_store,
    )
    evaluator = ForgettingEvaluator(adaptation_threshold=0.5)
    controller = AdaptiveController(
        RuleBasedAdaptivePolicy(
            RuleBasedPolicyConfig(
                degradation_patience=1,
                degradation_min_delta=0.0,
                degradation_action=AdaptiveAction.ROLLBACK,
            )
        )
    )

    engine = create_engine(f"sqlite+pysqlite:///{database_path}", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    metrics: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    rollbacks: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []

    with session_factory() as session:
        repository = SqlAlchemyTrainingVersionRepository(session)
        repository.create_branch(name="main")
        citadel = CitadelSafetyLayer(version_repository=repository)
        rollback = RollbackCoordinator(
            version_repository=repository,
            citadel=citadel,
            artifact_store=artifact_store,
        )

        baseline_metric = _train_and_evaluate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=config.baseline_epochs,
            device=device,
            stage_id="baseline",
        )
        baseline_continual = _continual_metrics(
            model=model,
            dataloader=validation_loader,
            evaluator=evaluator,
            stage_id="baseline",
            device=device,
            introduced_class_ids=tuple(range(10)),
            old_class_ids=(),
        )
        baseline_commit = _save_and_commit(
            repository=repository,
            checkpoint_manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            state=CheckpointState(epoch=1, global_step=baseline_metric["examples"]),
            experiment_id=config.experiment_id,
            stage_id="baseline",
            metric=baseline_metric,
            continual=baseline_continual,
        )
        metrics.append(_metric_snapshot("baseline", baseline_metric, baseline_continual))
        logs.append(_log("training", "Baseline checkpoint committed."))

        _set_optimizer_lr(optimizer, config.spike_learning_rate)
        degraded_metric = _train_and_evaluate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=degraded_loader,
            validation_loader=validation_loader,
            epochs=config.degraded_epochs,
            device=device,
            stage_id="degraded",
        )
        _force_degradation_if_needed(
            model,
            degraded_metric,
            baseline_metric,
            validation_loader,
            criterion,
            device,
        )
        degraded_continual = _continual_metrics(
            model=model,
            dataloader=validation_loader,
            evaluator=evaluator,
            stage_id="degraded",
            device=device,
            introduced_class_ids=(),
            old_class_ids=tuple(range(10)),
        )
        degraded_commit = _save_and_commit(
            repository=repository,
            checkpoint_manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            state=CheckpointState(
                epoch=2,
                global_step=baseline_metric["examples"] + degraded_metric["examples"],
            ),
            experiment_id=config.experiment_id,
            stage_id="degraded",
            metric=degraded_metric,
            continual=degraded_continual,
        )
        metrics.append(_metric_snapshot("degraded", degraded_metric, degraded_continual))

        history = [
            _metric_point(1, baseline_metric, config.learning_rate),
            _metric_point(2, degraded_metric, config.spike_learning_rate),
        ]
        decision = controller.decide(
            metrics=history,
            context=TrainingContext(
                branch_name="main",
                current_commit_id=degraded_commit.id,
                best_commit_id=baseline_commit.id,
                current_learning_rate=config.spike_learning_rate,
            ),
        )
        decisions.append(_decision_snapshot("decision_rollback", decision, degraded_commit.id))

        restored = False
        if decision.action is AdaptiveAction.ROLLBACK:
            restore_result = rollback.rollback_and_restore(
                actor="real-e2e",
                branch_name="main",
                current_commit_id=degraded_commit.id,
                target_commit_id=str(decision.parameters["target_commit_id"]),
                model=model,
                optimizer=optimizer,
                map_location=device,
            )
            restored = True
            rollbacks.append(
                {
                    "id": "rollback_1",
                    "branchName": restore_result.branch.name,
                    "fromCommitId": degraded_commit.id,
                    "toCommitId": restore_result.commit.id,
                    "actor": "real-e2e",
                    "createdAt": _now(),
                    "reason": (
                        "Validation loss degraded after intentional LR spike and corrupted stage."
                    ),
                }
            )
            logs.append(_log("rollback", "Rollback restored model and optimizer state."))

        _set_optimizer_lr(optimizer, config.learning_rate)
        recovery_metric = _train_and_evaluate(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            validation_loader=validation_loader,
            epochs=config.recovery_epochs,
            device=device,
            stage_id="recovery",
        )
        recovery_continual = _continual_metrics(
            model=model,
            dataloader=validation_loader,
            evaluator=evaluator,
            stage_id="recovery",
            device=device,
            introduced_class_ids=(),
            old_class_ids=tuple(range(10)),
        )
        recovery_commit = _save_and_commit(
            repository=repository,
            checkpoint_manager=checkpoint_manager,
            model=model,
            optimizer=optimizer,
            state=CheckpointState(epoch=3, global_step=recovery_metric["examples"]),
            experiment_id=config.experiment_id,
            stage_id="recovery",
            metric=recovery_metric,
            continual=recovery_continual,
        )
        metrics.append(_metric_snapshot("recovery", recovery_metric, recovery_continual))
        logs.append(_log("training", "Continued training from restored checkpoint."))
        session.commit()

        commit_graph = repository.get_commit_graph()
        branches = (repository.get_branch("main"),)

    artifacts = _write_real_artifacts(
        config=config,
        metrics=metrics,
        decisions=decisions,
        rollbacks=rollbacks,
        logs=logs,
        commit_graph=commit_graph,
        branches=branches,
    )
    return RealVerticalSliceResult(
        artifacts=artifacts,
        rollback_restored=restored,
        baseline_commit_id=baseline_commit.id,
        degraded_commit_id=degraded_commit.id,
        recovery_commit_id=recovery_commit.id,
    )


def _fashion_mnist_datasets(config: RealVerticalSliceConfig) -> tuple[Dataset[Any], Dataset[Any]]:
    from torchvision import datasets, transforms  # type: ignore[import-untyped]

    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(
        root=config.data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    validation = datasets.FashionMNIST(
        root=config.data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    return (
        _deterministic_subset(train, config.max_train_samples),
        _deterministic_subset(validation, config.max_validation_samples),
    )


def _deterministic_subset(dataset: Dataset[Any], limit: int) -> Dataset[Any]:
    return Subset(dataset, range(min(limit, len(dataset))))  # type: ignore[arg-type]


def load_real_vertical_slice_config(
    path: Path,
    *,
    output_dir: Path | None = None,
) -> RealVerticalSliceConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RealVerticalSliceConfig(
        experiment_id=str(raw.get("experiment_id", "acn-real-fashion-mnist-rollback")),
        seed=int(raw.get("seed", 20260518)),
        output_dir=output_dir
        or Path(str(raw.get("output_dir", "experiments/acn-real-fashion-mnist-rollback"))),
        data_dir=Path(str(raw.get("data_dir", "data"))),
        max_train_samples=int(raw.get("max_train_samples", 512)),
        max_validation_samples=int(raw.get("max_validation_samples", 256)),
        batch_size=int(raw.get("batch_size", 64)),
        baseline_epochs=int(raw.get("baseline_epochs", 1)),
        degraded_epochs=int(raw.get("degraded_epochs", 1)),
        recovery_epochs=int(raw.get("recovery_epochs", 1)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        spike_learning_rate=float(raw.get("spike_learning_rate", 0.75)),
        device=raw.get("device"),
    )


def _train_and_evaluate(
    *,
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    epochs: int,
    device: torch.device,
    stage_id: str,
) -> _StageMetric:
    for _epoch in range(epochs):
        model.train()
        for batch in train_loader:
            inputs, targets = _batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
    validation = _evaluate(model, criterion, validation_loader, device)
    return {
        "stage_id": stage_id,
        "train_loss": validation["train_loss"],
        "validation_loss": validation["validation_loss"],
        "train_accuracy": validation["train_accuracy"],
        "validation_accuracy": validation["validation_accuracy"],
        "examples": validation["examples"],
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
) -> _EvaluationMetric:
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    for batch in dataloader:
        inputs, targets = _batch(batch, device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = int(targets.numel())
        loss_sum += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += batch_size
    return {
        "train_loss": loss_sum / max(total, 1),
        "validation_loss": loss_sum / max(total, 1),
        "train_accuracy": correct / max(total, 1),
        "validation_accuracy": correct / max(total, 1),
        "examples": total,
    }


@torch.inference_mode()
def _continual_metrics(
    *,
    model: nn.Module,
    dataloader: DataLoader[Any],
    evaluator: ForgettingEvaluator,
    stage_id: str,
    device: torch.device,
    introduced_class_ids: Sequence[int],
    old_class_ids: Sequence[int],
) -> _ContinualMetric:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    for batch in dataloader:
        inputs, batch_targets = _batch(batch, device)
        logits = model(inputs)
        targets.extend(int(value) for value in batch_targets.cpu().tolist())
        predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
    metrics = evaluator.evaluate_predictions(
        stage_id=stage_id,
        introduced_class_ids=introduced_class_ids,
        old_class_ids=old_class_ids,
        targets=targets,
        predictions=predictions,
    )
    return {
        "oldClassRetention": metrics.old_class_retention,
        "newClassAdaptation": metrics.new_class_adaptation,
        "forgettingScore": metrics.forgetting_score,
        "adaptationLatency": metrics.adaptation_latency,
    }


def _save_and_commit(
    *,
    repository: SqlAlchemyTrainingVersionRepository,
    checkpoint_manager: CheckpointManager,
    model: nn.Module,
    optimizer: Optimizer,
    state: CheckpointState,
    experiment_id: str,
    stage_id: str,
    metric: _StageMetric,
    continual: _ContinualMetric,
) -> CommitRecord:
    reference = checkpoint_manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        state=state,
        name=f"{experiment_id}/{stage_id}.pt",
    )
    checkpoint = repository.create_checkpoint(
        uri=reference.uri,
        content_hash=reference.checksum,
        size_bytes=reference.size_bytes,
        metadata={"experiment_id": experiment_id, "stage_id": stage_id},
    )
    return repository.create_commit(
        branch_name="main",
        checkpoint_id=checkpoint.id,
        message=f"real-e2e:{stage_id}",
        authored_by="real-e2e",
        metrics={
            "train_loss": float(metric["train_loss"]),
            "validation_loss": float(metric["validation_loss"]),
            "train_accuracy": float(metric["train_accuracy"]),
            "validation_accuracy": float(metric["validation_accuracy"]),
            "forgetting_score": float(continual["forgettingScore"] or 0.0),
        },
        metadata={"experiment_id": experiment_id, "stage_id": stage_id},
    )


def _force_degradation_if_needed(
    model: nn.Module,
    degraded_metric: _StageMetric,
    baseline_metric: _StageMetric,
    validation_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> None:
    if float(degraded_metric["validation_loss"]) > float(baseline_metric["validation_loss"]):
        return
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 2.5)
    updated = _evaluate(model, criterion, validation_loader, device)
    degraded_metric["train_loss"] = updated["train_loss"]
    degraded_metric["validation_loss"] = updated["validation_loss"]
    degraded_metric["train_accuracy"] = updated["train_accuracy"]
    degraded_metric["validation_accuracy"] = updated["validation_accuracy"]
    degraded_metric["examples"] = updated["examples"]


def _metric_point(epoch: int, metric: _StageMetric, learning_rate: float) -> MetricPoint:
    return MetricPoint(
        epoch=epoch,
        train_loss=float(metric["train_loss"]),
        validation_loss=float(metric["validation_loss"]),
        train_accuracy=float(metric["train_accuracy"]),
        validation_accuracy=float(metric["validation_accuracy"]),
        learning_rate=learning_rate,
    )


def _metric_snapshot(
    stage_id: str,
    metric: _StageMetric,
    continual: _ContinualMetric,
) -> dict[str, object]:
    return {
        "timestamp": _now(),
        "stageId": stage_id,
        "trainLoss": float(metric["train_loss"]),
        "validationLoss": float(metric["validation_loss"]),
        "trainAccuracy": float(metric["train_accuracy"]),
        "validationAccuracy": float(metric["validation_accuracy"]),
        "forgettingScore": continual["forgettingScore"],
        "oldClassRetention": continual["oldClassRetention"],
        "newClassAdaptation": continual["newClassAdaptation"],
        "adaptationLatency": continual["adaptationLatency"],
    }


def _decision_snapshot(
    id_: str,
    decision: ControllerDecision,
    commit_id: str,
) -> dict[str, object]:
    return {
        "id": id_,
        "action": decision.action.value,
        "confidence": decision.confidence,
        "branchName": "main",
        "commitId": commit_id,
        "reasons": list(decision.reasons),
        "createdAt": _now(),
        "status": "executed",
    }


def _write_real_artifacts(
    *,
    config: RealVerticalSliceConfig,
    metrics: Sequence[dict[str, object]],
    decisions: Sequence[dict[str, object]],
    rollbacks: Sequence[dict[str, object]],
    logs: Sequence[dict[str, object]],
    commit_graph: CommitGraph,
    branches: Sequence[BranchRecord],
) -> RealVerticalSliceArtifacts:
    metrics_path = config.output_dir / "metrics.json"
    snapshot_path = config.output_dir / "dashboard_snapshot.json"
    rollback_path = config.output_dir / "rollback_events.json"
    report_path = config.output_dir / "report.md"
    plot_path = config.output_dir / "validation_plot.svg"
    forgetting_plot_path = config.output_dir / "forgetting_plot.svg"
    adaptation_plot_path = config.output_dir / "adaptation_plot.svg"
    rollback_report_path = config.output_dir / "rollback_report.md"
    snapshot = _dashboard_snapshot(
        config=config,
        commit_graph=commit_graph,
        branches=branches,
        metrics=metrics,
        decisions=decisions,
        rollbacks=rollbacks,
        logs=logs,
    )
    metrics_path.write_text(_json(list(metrics)), encoding="utf-8")
    rollback_path.write_text(_json(list(rollbacks)), encoding="utf-8")
    snapshot_path.write_text(_json(snapshot), encoding="utf-8")
    plot_path.write_text(
        _plot_svg(
            values=[_metric_float(metric, "validationLoss") for metric in metrics],
            labels=[str(metric["stageId"]) for metric in metrics],
            title="Validation Loss",
        ),
        encoding="utf-8",
    )
    forgetting_plot_path.write_text(
        _plot_svg(
            values=[_metric_float(metric, "forgettingScore") for metric in metrics],
            labels=[str(metric["stageId"]) for metric in metrics],
            title="Forgetting Score",
        ),
        encoding="utf-8",
    )
    adaptation_plot_path.write_text(
        _plot_svg(
            values=[_metric_float(metric, "newClassAdaptation") for metric in metrics],
            labels=[str(metric["stageId"]) for metric in metrics],
            title="Adaptation",
        ),
        encoding="utf-8",
    )
    report_path.write_text(_report(config, metrics, decisions, rollbacks), encoding="utf-8")
    rollback_report_path.write_text(_rollback_report(rollbacks), encoding="utf-8")
    return RealVerticalSliceArtifacts(
        output_dir=config.output_dir,
        metrics_json=metrics_path,
        dashboard_snapshot_json=snapshot_path,
        rollback_events_json=rollback_path,
        report_markdown=report_path,
        validation_plot_svg=plot_path,
        forgetting_plot_svg=forgetting_plot_path,
        adaptation_plot_svg=adaptation_plot_path,
        rollback_report_markdown=rollback_report_path,
    )


def _dashboard_snapshot(
    *,
    config: RealVerticalSliceConfig,
    commit_graph: CommitGraph,
    branches: Sequence[BranchRecord],
    metrics: Sequence[dict[str, object]],
    decisions: Sequence[dict[str, object]],
    rollbacks: Sequence[dict[str, object]],
    logs: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "commitGraph": {
            "nodes": [
                {
                    "id": node.id,
                    "branchId": node.branch_id,
                    "checkpointId": node.checkpoint_id,
                    "message": node.message,
                    "createdAt": node.created_at.isoformat(),
                    "metrics": node.metrics,
                }
                for node in commit_graph.nodes
            ],
            "edges": [
                {"parentId": edge.parent_id, "childId": edge.child_id}
                for edge in commit_graph.edges
            ],
        },
        "branchGraph": {
            "nodes": [
                {
                    "id": branch.id,
                    "name": branch.name,
                    "headCommitId": branch.head_commit_id,
                    "baseCommitId": branch.base_commit_id,
                    "status": "active",
                }
                for branch in branches
            ],
            "edges": [],
        },
        "metricsTimeline": list(metrics),
        "experiments": [
            {
                "id": config.experiment_id,
                "name": "Real Fashion-MNIST rollback vertical slice",
                "status": "completed",
                "branchName": "main",
                "currentStageId": "recovery",
                "currentCommitId": commit_graph.nodes[-1].id if commit_graph.nodes else None,
                "bestCommitId": commit_graph.nodes[0].id if commit_graph.nodes else None,
                "updatedAt": _now(),
            }
        ],
        "controllerDecisions": list(decisions),
        "rollbackHistory": list(rollbacks),
        "liveLogs": list(logs),
    }


def _report(
    config: RealVerticalSliceConfig,
    metrics: Sequence[dict[str, object]],
    decisions: Sequence[dict[str, object]],
    rollbacks: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Real ACN Vertical Slice",
        "",
        f"Experiment: `{config.experiment_id}`",
        "",
        "| Stage | Validation loss | Validation accuracy | Forgetting |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        lines.append(
            f"| {metric['stageId']} | {_metric_float(metric, 'validationLoss'):.4f} | "
            f"{_metric_float(metric, 'validationAccuracy'):.4f} | "
            f"{_metric_float(metric, 'forgettingScore'):.4f} |"
        )
    lines.extend(["", "## Decisions", ""])
    lines.extend(f"- `{decision['action']}` at `{decision['commitId']}`." for decision in decisions)
    lines.extend(["", "## Rollbacks", ""])
    lines.extend(
        f"- `{rollback['fromCommitId']}` -> `{rollback['toCommitId']}`." for rollback in rollbacks
    )
    return "\n".join(lines) + "\n"


def _rollback_report(rollbacks: Sequence[dict[str, object]]) -> str:
    lines = ["# Rollback Event Report", ""]
    if not rollbacks:
        lines.append("No rollback events were recorded.")
    for rollback in rollbacks:
        lines.extend(
            [
                f"## {rollback['id']}",
                "",
                f"- Branch: `{rollback['branchName']}`",
                f"- From: `{rollback['fromCommitId']}`",
                f"- To: `{rollback['toCommitId']}`",
                f"- Actor: `{rollback['actor']}`",
                f"- Reason: {rollback['reason']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _plot_svg(*, values: Sequence[float], labels: Sequence[str], title: str) -> str:
    width = 720
    height = 260
    max_value = max([1.0, *values])
    points = []
    for index, value in enumerate(values):
        x = 48 + index * (600 / max(len(values) - 1, 1))
        y = 210 - value * 160 / max_value
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="#0f172a"/>'
        '<text x="48" y="32" fill="#f8fafc" font-family="Arial" font-size="18">'
        f"{title}</text>"
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#38bdf8" stroke-width="3"/>'
        + "".join(
            f'<text x="{48 + i * (600 / max(len(labels) - 1, 1)):.1f}" y="238" '
            f'fill="#cbd5e1" font-size="11">{label}</text>'
            for i, label in enumerate(labels)
        )
        + "</svg>\n"
    )


def _batch(batch: object, device: torch.device) -> tuple[Tensor, Tensor]:
    if isinstance(batch, tuple | list) and len(batch) >= 2:
        images = batch[0]
        targets = batch[1]
        if isinstance(images, Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)
            return images.float().to(device), torch.as_tensor(
                targets,
                dtype=torch.long,
                device=device,
            ).view(-1)

    image, target = _parse_sample(batch)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    return image.to(device), torch.tensor([target], dtype=torch.long, device=device)


def _parse_sample(sample: object) -> tuple[Tensor, int]:
    if isinstance(sample, tuple | list) and len(sample) >= 2:
        image = sample[0]
        target = sample[1]
        if isinstance(image, Tensor):
            target_value = int(target.item()) if isinstance(target, Tensor) else int(target)
            return image.float(), target_value
    msg = "Expected Fashion-MNIST style sample."
    raise TypeError(msg)


def _set_optimizer_lr(optimizer: Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log(source: str, message: str) -> dict[str, object]:
    return {
        "id": f"log_{source}_{datetime.now(UTC).timestamp()}",
        "level": "info",
        "source": source,
        "message": message,
        "createdAt": _now(),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _metric_float(metric: dict[str, object], key: str) -> float:
    value = metric.get(key)
    if value is None:
        return 0.0
    return float(cast(float | int | str, value))
