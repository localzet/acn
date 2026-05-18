"""Synthetic E2E experiment utility.

This module is deterministic research/demo support, not the production training
path. Real checkpoint restoration is exercised by `acn.experiments.real_vertical`.
"""

import json
import random
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from torch import Tensor
from torch.utils.data import Dataset

from acn.citadel import CitadelSafetyLayer
from acn.continual import (
    ContinualLearningScenario,
    DatasetStage,
    DatasetStageConfig,
    ForgettingEvaluator,
    ReplayBuffer,
    ReplayBufferConfig,
)
from acn.continual.stage import DatasetSplit
from acn.controller import (
    AdaptiveAction,
    AdaptiveController,
    ControllerDecision,
    MetricPoint,
    TrainingContext,
)
from acn.orchestration import (
    DecisionExecutor,
    RollbackCoordinator,
)
from acn.versioning.domain import BranchRecord, CommitGraph, CommitRecord
from acn.versioning.models import Base
from acn.versioning.repository import SqlAlchemyTrainingVersionRepository

DatasetBackend = Literal["synthetic"]


@dataclass(frozen=True, slots=True)
class E2EStageConfig:
    id: str
    dataset: str
    class_ids: tuple[int, ...]
    epochs: int = 1
    replay_ratio: float = 0.0
    domain_shift: str | None = None
    degradation: bool = False
    force_branch: bool = False


@dataclass(frozen=True, slots=True)
class E2EExperimentConfig:
    experiment_id: str
    seed: int
    backend: DatasetBackend
    output_dir: Path
    stages: tuple[E2EStageConfig, ...]
    samples_per_class: int = 8
    replay_capacity: int = 128
    replay_samples_per_class: int = 4
    learning_rate: float = 1e-3
    screenshot_export: bool = True


@dataclass(frozen=True, slots=True)
class ExperimentArtifactPaths:
    output_dir: Path
    metrics: Path
    commit_graph: Path
    branch_graph: Path
    rollback_events: Path
    summary_json: Path
    report_markdown: Path
    forgetting_plot: Path
    adaptation_plot: Path
    screenshot_svg: Path | None


@dataclass(frozen=True, slots=True)
class StageRunRecord:
    stage_id: str
    dataset: str
    class_ids: tuple[int, ...]
    introduced_class_ids: tuple[int, ...]
    old_class_ids: tuple[int, ...]
    domain_shift: str | None
    train_loss: float
    validation_loss: float
    train_accuracy: float
    validation_accuracy: float
    old_class_retention: float | None
    new_class_adaptation: float | None
    forgetting_score: float
    adaptation_latency: int | None
    decision_action: str
    commit_id: str
    branch_name: str


@dataclass(frozen=True, slots=True)
class RollbackEvent:
    stage_id: str
    branch_name: str
    from_commit_id: str
    to_commit_id: str
    action: str


@dataclass(frozen=True, slots=True)
class BranchEvent:
    stage_id: str
    branch_name: str
    base_commit_id: str


@dataclass(frozen=True, slots=True)
class E2EExperimentResult:
    artifacts: ExperimentArtifactPaths
    stages: tuple[StageRunRecord, ...]
    rollback_events: tuple[RollbackEvent, ...]
    branch_events: tuple[BranchEvent, ...]
    commit_graph: CommitGraph


@dataclass(frozen=True, slots=True)
class _PreparedDataset:
    name: str
    class_ids: tuple[int, ...]
    dataset: Dataset[tuple[Tensor, int]]


class _SyntheticImageDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, *, class_ids: Sequence[int], samples_per_class: int, channels: int) -> None:
        self._samples = tuple(
            (class_id, sample_index)
            for class_id in class_ids
            for sample_index in range(samples_per_class)
        )
        self._channels = channels

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        class_id, sample_index = self._samples[index]
        base = (class_id + 1) / 10.0
        image = torch.full((self._channels, 8, 8), base + sample_index / 1000.0)
        return image, class_id


class _ScenarioSource:
    def __init__(self, prepared: _PreparedDataset) -> None:
        self._prepared = prepared

    @property
    def name(self) -> str:
        return self._prepared.name

    @property
    def class_ids(self) -> tuple[int, ...]:
        return self._prepared.class_ids

    def build_dataset(
        self,
        *,
        split: DatasetSplit,
        class_ids: Sequence[int] | None = None,
    ) -> Dataset[tuple[Tensor, int]]:
        _ = split
        if class_ids is None:
            return self._prepared.dataset
        allowed = frozenset(class_ids)
        return _FilteredSyntheticDataset(self._prepared.dataset, allowed)


class _FilteredSyntheticDataset(Dataset[tuple[Tensor, int]]):
    def __init__(self, dataset: Dataset[tuple[Tensor, int]], class_ids: frozenset[int]) -> None:
        if not isinstance(dataset, Sized):
            msg = "Synthetic dataset must implement __len__."
            raise TypeError(msg)
        self._dataset = dataset
        self._indices = tuple(
            index for index in range(len(dataset)) if dataset[index][1] in class_ids
        )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        return self._dataset[self._indices[index]]


def run_e2e_experiment(config: E2EExperimentConfig) -> E2EExperimentResult:
    _seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    scenario = _build_scenario(config)
    replay = ReplayBuffer(
        ReplayBufferConfig(
            capacity=config.replay_capacity,
            samples_per_class=config.replay_samples_per_class,
        )
    )
    evaluator = ForgettingEvaluator(adaptation_threshold=0.75)
    controller = AdaptiveController()

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)

    stage_records: list[StageRunRecord] = []
    rollback_events: list[RollbackEvent] = []
    branch_events: list[BranchEvent] = []

    with session_factory() as session:
        version_repository = SqlAlchemyTrainingVersionRepository(session)
        version_repository.create_branch(name="main")
        citadel = CitadelSafetyLayer(version_repository=version_repository)
        rollback = RollbackCoordinator(version_repository=version_repository, citadel=citadel)
        decision_executor = DecisionExecutor(
            version_repository=version_repository,
            citadel=citadel,
            rollback_coordinator=rollback,
        )
        best_commit_id: str | None = None
        metric_history: list[MetricPoint] = []

        for index, stage in enumerate(scenario.stages):
            dataset = scenario.build_stage_dataset_with_replay(stage, replay_buffer=replay)
            metric = _simulate_stage_metric(
                config=config,
                stage=stage,
                stage_index=index,
                previous_metric=metric_history[-1] if metric_history else None,
            )
            metric_history.append(metric)
            continual_metrics = evaluator.evaluate_predictions(
                stage_id=stage.id,
                introduced_class_ids=stage.introduced_class_ids,
                old_class_ids=scenario.old_class_ids_before(stage),
                targets=_targets_for_stage(stage, config.samples_per_class),
                predictions=_predictions_for_stage(stage, config.samples_per_class),
            )
            commit = _commit_stage(
                version_repository=version_repository,
                experiment_id=config.experiment_id,
                stage=stage,
                metric=metric,
                continual_metrics=continual_metrics.forgetting_score,
            )
            if best_commit_id is None or metric.validation_loss <= min(
                point.validation_loss for point in metric_history[:-1] or [metric]
            ):
                best_commit_id = commit.id

            context = TrainingContext(
                branch_name="main",
                current_commit_id=commit.id,
                best_commit_id=best_commit_id,
                current_learning_rate=metric.learning_rate,
            )
            decision = controller.decide(metrics=metric_history, context=context)
            forced_action = _forced_action_for_stage(config.stages[index], decision.action)
            if forced_action is not None:
                decision = _decision_forced_action(
                    action=forced_action,
                    context=context,
                    metrics=metric_history,
                )

            from_commit_id = commit.id
            result = decision_executor.execute(
                decision=decision,
                actor="e2e-pipeline",
                branch_name="main",
                current_commit_id=commit.id,
            )
            if decision.action is AdaptiveAction.ROLLBACK and result.executed:
                target_commit_id = str(result.metadata["head_commit_id"])
                rollback_events.append(
                    RollbackEvent(
                        stage_id=stage.id,
                        branch_name="main",
                        from_commit_id=from_commit_id,
                        to_commit_id=target_commit_id,
                        action=decision.action.value,
                    )
                )
                version_repository.create_branch(
                    name=f"main/recovery-{stage.id}",
                    base_commit_id=target_commit_id,
                    metadata={"stage_id": stage.id, "type": "rollback_recovery"},
                )
                branch_events.append(
                    BranchEvent(
                        stage_id=stage.id,
                        branch_name=f"main/recovery-{stage.id}",
                        base_commit_id=target_commit_id,
                    )
                )
            elif decision.action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH and result.executed:
                branch_events.append(
                    BranchEvent(
                        stage_id=stage.id,
                        branch_name=str(result.metadata["branch_name"]),
                        base_commit_id=str(result.metadata["base_commit_id"]),
                    )
                )

            stage_records.append(
                StageRunRecord(
                    stage_id=stage.id,
                    dataset=str(config.stages[index].dataset),
                    class_ids=stage.class_ids,
                    introduced_class_ids=stage.introduced_class_ids,
                    old_class_ids=scenario.old_class_ids_before(stage),
                    domain_shift=stage.domain_shift,
                    train_loss=metric.train_loss,
                    validation_loss=metric.validation_loss,
                    train_accuracy=metric.train_accuracy or 0.0,
                    validation_accuracy=metric.validation_accuracy or 0.0,
                    old_class_retention=continual_metrics.old_class_retention,
                    new_class_adaptation=continual_metrics.new_class_adaptation,
                    forgetting_score=continual_metrics.forgetting_score,
                    adaptation_latency=continual_metrics.adaptation_latency,
                    decision_action=decision.action.value,
                    commit_id=commit.id,
                    branch_name="main",
                )
            )
            replay.add_dataset(dataset)

        commit_graph = version_repository.get_commit_graph()
        branches = (
            version_repository.get_branch("main"),
            *(version_repository.get_branch(event.branch_name) for event in branch_events),
        )

    artifacts = _write_artifacts(
        config=config,
        stages=stage_records,
        rollback_events=rollback_events,
        branch_events=branch_events,
        commit_graph=commit_graph,
        branches=branches,
    )
    return E2EExperimentResult(
        artifacts=artifacts,
        stages=tuple(stage_records),
        rollback_events=tuple(rollback_events),
        branch_events=tuple(branch_events),
        commit_graph=commit_graph,
    )


def load_e2e_config(path: Path, *, output_dir: Path | None = None) -> E2EExperimentConfig:
    raw = json.loads(path.read_text(encoding="utf-8"))
    stages = tuple(
        E2EStageConfig(
            id=str(stage["id"]),
            dataset=str(stage["dataset"]),
            class_ids=tuple(int(value) for value in stage["class_ids"]),
            epochs=int(stage.get("epochs", 1)),
            replay_ratio=float(stage.get("replay_ratio", 0.0)),
            domain_shift=stage.get("domain_shift"),
            degradation=bool(stage.get("degradation", False)),
            force_branch=bool(stage.get("force_branch", False)),
        )
        for stage in raw["stages"]
    )
    backend = str(raw.get("backend", "synthetic"))
    if backend != "synthetic":
        msg = f"Unsupported E2E backend: {backend}"
        raise ValueError(msg)
    return E2EExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        seed=int(raw.get("seed", 20260518)),
        backend="synthetic",
        output_dir=output_dir or Path(str(raw.get("output_dir", "experiments/e2e-acn"))),
        stages=stages,
        samples_per_class=int(raw.get("samples_per_class", 8)),
        replay_capacity=int(raw.get("replay_capacity", 128)),
        replay_samples_per_class=int(raw.get("replay_samples_per_class", 4)),
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        screenshot_export=bool(raw.get("screenshot_export", True)),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _build_scenario(config: E2EExperimentConfig) -> ContinualLearningScenario:
    source_names = {
        source_name
        for stage in config.stages
        for source_name in (stage.dataset, stage.domain_shift)
        if source_name is not None
    }
    sources = {
        source_name: _ScenarioSource(
            _PreparedDataset(
                name=source_name,
                class_ids=_class_ids_for_source(source_name, config.stages),
                dataset=_SyntheticImageDataset(
                    class_ids=_class_ids_for_source(source_name, config.stages),
                    samples_per_class=config.samples_per_class,
                    channels=1 if source_name == "fashion-mnist" else 3,
                ),
            )
        )
        for source_name in source_names
    }
    stage_configs = tuple(
        DatasetStageConfig(
            id=stage.id,
            source_name=stage.dataset,
            class_ids=stage.class_ids,
            domain_shift=stage.domain_shift,
            replay_ratio=stage.replay_ratio,
            metadata={"epochs": stage.epochs},
        )
        for stage in config.stages
    )
    return ContinualLearningScenario.from_configs(
        scenario_id=config.experiment_id,
        sources=sources,
        stage_configs=stage_configs,
    )


def _class_ids_for_source(
    source_name: str,
    stages: Sequence[E2EStageConfig],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                class_id
                for stage in stages
                if stage.dataset == source_name or stage.domain_shift == source_name
                for class_id in stage.class_ids
            }
        )
    )


def _simulate_stage_metric(
    *,
    config: E2EExperimentConfig,
    stage: DatasetStage,
    stage_index: int,
    previous_metric: MetricPoint | None,
) -> MetricPoint:
    base_loss = 1.2 - stage_index * 0.14
    if stage.domain_shift is not None:
        base_loss += 0.18
    if previous_metric is not None and _stage_forces_degradation(stage.id, config):
        validation_loss = previous_metric.validation_loss + 0.08
    else:
        validation_loss = max(0.18, base_loss - 0.05)
    train_loss = max(0.12, validation_loss - (0.05 if stage.domain_shift is None else 0.12))
    train_accuracy = min(0.98, 0.55 + stage_index * 0.08)
    validation_accuracy = max(0.1, min(0.95, 1.0 - validation_loss / 2.0))
    return MetricPoint(
        epoch=stage_index + 1,
        train_loss=round(train_loss, 6),
        validation_loss=round(validation_loss, 6),
        train_accuracy=round(train_accuracy, 6),
        validation_accuracy=round(validation_accuracy, 6),
        learning_rate=config.learning_rate,
    )


def _stage_forces_degradation(stage_id: str, config: E2EExperimentConfig) -> bool:
    return any(stage.id == stage_id and stage.degradation for stage in config.stages)


def _targets_for_stage(stage: DatasetStage, samples_per_class: int) -> tuple[int, ...]:
    return tuple(class_id for class_id in stage.class_ids for _ in range(samples_per_class))


def _predictions_for_stage(stage: DatasetStage, samples_per_class: int) -> tuple[int, ...]:
    predictions: list[int] = []
    old_classes = set(stage.class_ids) - set(stage.introduced_class_ids)
    for class_id in stage.class_ids:
        for sample_index in range(samples_per_class):
            if class_id in old_classes and sample_index == 0:
                predictions.append(stage.class_ids[-1])
            else:
                predictions.append(class_id)
    return tuple(predictions)


def _commit_stage(
    *,
    version_repository: SqlAlchemyTrainingVersionRepository,
    experiment_id: str,
    stage: DatasetStage,
    metric: MetricPoint,
    continual_metrics: float,
) -> CommitRecord:
    digest = sha256(f"{experiment_id}:{stage.id}:{metric.validation_loss}".encode()).hexdigest()
    checkpoint = version_repository.create_checkpoint(
        uri=f"memory://{experiment_id}/{stage.id}.pt",
        content_hash=f"sha256:{digest}",
        size_bytes=1024,
        metadata={"stage_id": stage.id, "experiment_id": experiment_id},
    )
    return version_repository.create_commit(
        branch_name="main",
        checkpoint_id=checkpoint.id,
        message=f"e2e-stage:{stage.id}",
        authored_by="e2e-pipeline",
        metrics={
            "epoch": metric.epoch,
            "train_loss": metric.train_loss,
            "validation_loss": metric.validation_loss,
            "train_accuracy": metric.train_accuracy,
            "validation_accuracy": metric.validation_accuracy,
            "forgetting_score": continual_metrics,
        },
        metadata={"stage_id": stage.id, "experiment_id": experiment_id},
    )


def _forced_action_for_stage(
    stage_config: E2EStageConfig,
    action: AdaptiveAction,
) -> AdaptiveAction | None:
    if stage_config.force_branch and action is not AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH:
        return AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH
    if stage_config.degradation and action is not AdaptiveAction.ROLLBACK:
        return AdaptiveAction.ROLLBACK
    return None


def _decision_forced_action(
    *,
    action: AdaptiveAction,
    context: TrainingContext,
    metrics: Sequence[MetricPoint],
) -> ControllerDecision:
    controller = AdaptiveController()
    baseline = controller.decide(metrics=metrics, context=context)
    parameters: dict[str, str | float | None] = {}
    if action is AdaptiveAction.ROLLBACK:
        parameters["target_commit_id"] = context.best_commit_id
    elif action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH:
        parameters["source_commit_id"] = context.current_commit_id
        parameters["source_branch"] = context.branch_name
    return type(baseline)(
        action=action,
        confidence=1.0,
        reasons=(f"E2E configuration forced {action.value}.",),
        signals=baseline.signals,
        parameters=parameters,
    )


def _write_artifacts(
    *,
    config: E2EExperimentConfig,
    stages: Sequence[StageRunRecord],
    rollback_events: Sequence[RollbackEvent],
    branch_events: Sequence[BranchEvent],
    commit_graph: CommitGraph,
    branches: Sequence[BranchRecord],
) -> ExperimentArtifactPaths:
    metrics_path = config.output_dir / "metrics.json"
    commit_graph_path = config.output_dir / "commit_graph.json"
    branch_graph_path = config.output_dir / "branch_graph.json"
    rollback_path = config.output_dir / "rollback_events.json"
    summary_path = config.output_dir / "experiment_summary.json"
    report_path = config.output_dir / "report.md"
    forgetting_plot_path = config.output_dir / "forgetting_plot.svg"
    adaptation_plot_path = config.output_dir / "adaptation_plot.svg"
    screenshot_path = (
        config.output_dir / "dashboard_screenshot.svg" if config.screenshot_export else None
    )

    metrics = [_stage_to_json(stage) for stage in stages]
    metrics_path.write_text(_json(metrics), encoding="utf-8")
    commit_graph_path.write_text(_json(_commit_graph_to_json(commit_graph)), encoding="utf-8")
    branch_graph_path.write_text(_json(_branch_graph_to_json(branches)), encoding="utf-8")
    rollback_path.write_text(
        _json([_rollback_to_json(event) for event in rollback_events]), encoding="utf-8"
    )
    summary = _summary(config, stages, rollback_events, branch_events)
    summary_path.write_text(_json(summary), encoding="utf-8")
    forgetting_plot_path.write_text(
        _line_plot_svg(
            title="Forgetting Score",
            values=[stage.forgetting_score for stage in stages],
            labels=[stage.stage_id for stage in stages],
        ),
        encoding="utf-8",
    )
    adaptation_plot_path.write_text(
        _line_plot_svg(
            title="New Class Adaptation",
            values=[stage.new_class_adaptation or 0.0 for stage in stages],
            labels=[stage.stage_id for stage in stages],
        ),
        encoding="utf-8",
    )
    if screenshot_path is not None:
        screenshot_path.write_text(_dashboard_svg(summary), encoding="utf-8")
    report_path.write_text(
        _report_markdown(summary, stages, rollback_events, branch_events), encoding="utf-8"
    )
    return ExperimentArtifactPaths(
        output_dir=config.output_dir,
        metrics=metrics_path,
        commit_graph=commit_graph_path,
        branch_graph=branch_graph_path,
        rollback_events=rollback_path,
        summary_json=summary_path,
        report_markdown=report_path,
        forgetting_plot=forgetting_plot_path,
        adaptation_plot=adaptation_plot_path,
        screenshot_svg=screenshot_path,
    )


def _summary(
    config: E2EExperimentConfig,
    stages: Sequence[StageRunRecord],
    rollback_events: Sequence[RollbackEvent],
    branch_events: Sequence[BranchEvent],
) -> dict[str, object]:
    return {
        "experiment_id": config.experiment_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "seed": config.seed,
        "backend": config.backend,
        "stage_count": len(stages),
        "rollback_count": len(rollback_events),
        "branch_count": len(branch_events) + 1,
        "final_validation_accuracy": stages[-1].validation_accuracy if stages else None,
        "max_forgetting_score": max((stage.forgetting_score for stage in stages), default=0.0),
        "datasets": sorted({stage.dataset for stage in stages}),
    }


def _json(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def _stage_to_json(stage: StageRunRecord) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "dataset": stage.dataset,
        "class_ids": list(stage.class_ids),
        "introduced_class_ids": list(stage.introduced_class_ids),
        "old_class_ids": list(stage.old_class_ids),
        "domain_shift": stage.domain_shift,
        "train_loss": stage.train_loss,
        "validation_loss": stage.validation_loss,
        "train_accuracy": stage.train_accuracy,
        "validation_accuracy": stage.validation_accuracy,
        "old_class_retention": stage.old_class_retention,
        "new_class_adaptation": stage.new_class_adaptation,
        "forgetting_score": stage.forgetting_score,
        "adaptation_latency": stage.adaptation_latency,
        "decision_action": stage.decision_action,
        "commit_id": stage.commit_id,
        "branch_name": stage.branch_name,
    }


def _commit_graph_to_json(graph: CommitGraph) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": node.id,
                "branch_id": node.branch_id,
                "checkpoint_id": node.checkpoint_id,
                "message": node.message,
                "created_at": node.created_at.isoformat(),
                "metadata": node.metadata,
                "metrics": node.metrics,
            }
            for node in graph.nodes
        ],
        "edges": [{"parent_id": edge.parent_id, "child_id": edge.child_id} for edge in graph.edges],
    }


def _branch_graph_to_json(branches: Sequence[BranchRecord]) -> dict[str, object]:
    return {
        "nodes": [
            {
                "id": branch.id,
                "name": branch.name,
                "head_commit_id": branch.head_commit_id,
                "base_commit_id": branch.base_commit_id,
                "metadata": branch.metadata,
            }
            for branch in branches
        ],
        "edges": [
            {
                "source": branch.base_commit_id,
                "target": branch.head_commit_id,
                "branch_name": branch.name,
            }
            for branch in branches
            if branch.base_commit_id is not None and branch.head_commit_id is not None
        ],
    }


def _rollback_to_json(event: RollbackEvent) -> dict[str, str]:
    return {
        "stage_id": event.stage_id,
        "branch_name": event.branch_name,
        "from_commit_id": event.from_commit_id,
        "to_commit_id": event.to_commit_id,
        "action": event.action,
    }


def _report_markdown(
    summary: Mapping[str, object],
    stages: Sequence[StageRunRecord],
    rollback_events: Sequence[RollbackEvent],
    branch_events: Sequence[BranchEvent],
) -> str:
    lines = [
        "# ACN E2E Experiment Report",
        "",
        f"Experiment: `{summary['experiment_id']}`",
        f"Seed: `{summary['seed']}`",
        f"Datasets: {_summary_datasets(summary)}",
        "",
        "## Stages",
        "",
        "| Stage | Dataset | Decision | Val Acc | Forgetting |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for stage in stages:
        lines.append(
            f"| {stage.stage_id} | {stage.dataset} | {stage.decision_action} | "
            f"{stage.validation_accuracy:.3f} | {stage.forgetting_score:.3f} |"
        )
    lines.extend(["", "## Rollbacks", ""])
    if rollback_events:
        for rollback_event in rollback_events:
            lines.append(
                f"- `{rollback_event.stage_id}` rolled `{rollback_event.branch_name}` "
                f"from `{rollback_event.from_commit_id}` to `{rollback_event.to_commit_id}`."
            )
    else:
        lines.append("- No rollback events.")
    lines.extend(["", "## Branches", ""])
    if branch_events:
        for branch_event in branch_events:
            lines.append(
                f"- `{branch_event.branch_name}` created from `{branch_event.base_commit_id}` "
                f"during `{branch_event.stage_id}`."
            )
    else:
        lines.append("- Only the main branch was used.")
    return "\n".join(lines) + "\n"


def _summary_datasets(summary: Mapping[str, object]) -> str:
    datasets = summary["datasets"]
    if not isinstance(datasets, list):
        return ""
    return ", ".join(str(value) for value in datasets)


def _line_plot_svg(*, title: str, values: Sequence[float], labels: Sequence[str]) -> str:
    width = 760
    height = 320
    left = 56
    top = 40
    plot_width = 660
    plot_height = 220
    max_value = max([1.0, *values])
    points = []
    for index, value in enumerate(values):
        x = left + (plot_width * index / max(len(values) - 1, 1))
        y = top + plot_height - (plot_height * value / max_value)
        points.append((x, y))
    point_attr = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "\n".join(
        (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4">'
            f"<title>{labels[index]}={values[index]:.3f}</title></circle>"
        )
        for index, (x, y) in enumerate(points)
    )
    return "\n".join(
        [
            _svg_open(width, height),
            '  <rect width="100%" height="100%" fill="#111827"/>',
            (
                f'  <text x="{left}" y="26" fill="#f9fafb" '
                f'font-family="Arial" font-size="18">{title}</text>'
            ),
            _svg_line(left, top + plot_height, left + plot_width, top + plot_height),
            _svg_line(left, top, left, top + plot_height),
            f'  <polyline points="{point_attr}" fill="none" stroke="#38bdf8" stroke-width="3"/>',
            f'  <g fill="#facc15">{circles}</g>',
            "</svg>",
            "",
        ]
    )


def _dashboard_svg(summary: Mapping[str, object]) -> str:
    return "\n".join(
        [
            _svg_open(960, 540),
            '  <rect width="960" height="540" fill="#0f172a"/>',
            _svg_text(48, 72, "ACN E2E Experiment", size=32, fill="#f8fafc"),
            _svg_text(48, 120, f"Experiment: {summary['experiment_id']}"),
            _svg_text(48, 160, f"Stages: {summary['stage_count']}"),
            _svg_text(48, 200, f"Branches: {summary['branch_count']}"),
            _svg_text(48, 240, f"Rollbacks: {summary['rollback_count']}"),
            _svg_text(48, 280, f"Max forgetting: {summary['max_forgetting_score']}"),
            "</svg>",
            "",
        ]
    )


def _svg_open(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
    )


def _svg_line(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" ' 'stroke="#6b7280"/>'


def _svg_text(
    x: int,
    y: int,
    text: object,
    *,
    size: int = 18,
    fill: str = "#cbd5e1",
) -> str:
    return (
        f'  <text x="{x}" y="{y}" fill="{fill}" font-family="Arial" '
        f'font-size="{size}">{text}</text>'
    )
