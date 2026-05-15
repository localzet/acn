from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

OptimizerName = Literal["sgd", "adam", "adamw"]
SchedulerName = Literal["none", "step", "cosine", "exponential"]


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    name: OptimizerName = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    name: SchedulerName = "none"
    step_size: int = 10
    gamma: float = 0.1
    t_max: int = 10


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    epochs: int = 1
    device: str | None = None
    mixed_precision: bool = True
    max_grad_norm: float | None = None
    checkpoint_dir: Path | None = None
    checkpoint_every_n_epochs: int = 1
    log_every_n_steps: int = 50


@dataclass(slots=True)
class CheckpointState:
    epoch: int = 0
    global_step: int = 0
    best_validation_loss: float | None = None


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    accuracy: float | None = None


@dataclass(slots=True)
class TrainingHistory:
    train: list[EpochMetrics] = field(default_factory=list)
    validation: list[EpochMetrics] = field(default_factory=list)
