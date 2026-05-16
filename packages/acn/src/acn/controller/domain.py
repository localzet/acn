from dataclasses import dataclass, field
from enum import StrEnum

type DecisionParameter = str | int | float | bool | None
type DecisionParameters = dict[str, DecisionParameter]


class AdaptiveAction(StrEnum):
    ROLLBACK = "rollback"
    DECREASE_LEARNING_RATE = "decrease_learning_rate"
    INCREASE_LEARNING_RATE = "increase_learning_rate"
    FREEZE_LAYERS = "freeze_layers"
    UNFREEZE_LAYERS = "unfreeze_layers"
    CREATE_EXPERIMENTAL_BRANCH = "create_experimental_branch"
    CONTINUE_TRAINING = "continue_training"


@dataclass(frozen=True, slots=True)
class MetricPoint:
    epoch: int
    train_loss: float
    validation_loss: float
    train_accuracy: float | None = None
    validation_accuracy: float | None = None
    learning_rate: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingContext:
    branch_name: str
    current_commit_id: str | None = None
    best_commit_id: str | None = None
    frozen_layers: bool = False
    current_learning_rate: float | None = None


@dataclass(frozen=True, slots=True)
class ControllerSignals:
    degradation: bool = False
    plateau: bool = False
    overfitting: bool = False
    underfitting: bool = False
    stable_improvement: bool = False
    latest_validation_delta: float | None = None
    generalization_gap: float | None = None


@dataclass(frozen=True, slots=True)
class ControllerDecision:
    action: AdaptiveAction
    confidence: float
    reasons: tuple[str, ...]
    signals: ControllerSignals
    parameters: DecisionParameters = field(default_factory=dict)
