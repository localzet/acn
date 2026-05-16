from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from acn.controller.domain import (
    AdaptiveAction,
    ControllerDecision,
    ControllerSignals,
    MetricPoint,
    TrainingContext,
)


@dataclass(frozen=True, slots=True)
class RuleBasedPolicyConfig:
    degradation_patience: int = 2
    degradation_min_delta: float = 0.02
    plateau_window: int = 4
    plateau_min_delta: float = 0.005
    overfitting_gap: float = 0.12
    overfitting_validation_delta: float = 0.01
    underfitting_max_train_accuracy: float = 0.45
    underfitting_min_epochs: int = 3
    stable_improvement_window: int = 3
    stable_improvement_min_delta: float = 0.01
    learning_rate_decrease_factor: float = 0.5
    learning_rate_increase_factor: float = 1.25
    minimum_learning_rate: float = 1e-6
    maximum_learning_rate: float = 1.0
    plateau_action: AdaptiveAction = AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH
    overfitting_action: AdaptiveAction = AdaptiveAction.FREEZE_LAYERS
    degradation_action: AdaptiveAction = AdaptiveAction.ROLLBACK


class RuleBasedAdaptivePolicy:
    def __init__(self, config: RuleBasedPolicyConfig | None = None) -> None:
        self._config = config or RuleBasedPolicyConfig()

    @property
    def config(self) -> RuleBasedPolicyConfig:
        return self._config

    def analyze(self, metrics: Sequence[MetricPoint]) -> ControllerSignals:
        if not metrics:
            return ControllerSignals()

        return ControllerSignals(
            degradation=self._detect_degradation(metrics),
            plateau=self._detect_plateau(metrics),
            overfitting=self._detect_overfitting(metrics),
            underfitting=self._detect_underfitting(metrics),
            stable_improvement=self._detect_stable_improvement(metrics),
            latest_validation_delta=_latest_validation_delta(metrics),
            generalization_gap=_generalization_gap(metrics[-1]),
        )

    def decide(
        self,
        *,
        metrics: Sequence[MetricPoint],
        context: TrainingContext,
    ) -> ControllerDecision:
        signals = self.analyze(metrics)
        latest = metrics[-1] if metrics else None

        if signals.degradation:
            return self._degradation_decision(signals, context)
        if signals.overfitting:
            return self._overfitting_decision(signals, context)
        if signals.stable_improvement and context.frozen_layers:
            return self._unfreeze_decision(signals)
        if signals.underfitting:
            return self._increase_learning_rate_decision(signals, context)
        if signals.plateau:
            return self._plateau_decision(signals, context)

        reason = (
            "No degradation, plateau, overfitting or underfitting signal exceeded "
            "policy thresholds."
        )
        if latest is not None:
            reason = f"{reason} Latest validation loss is {latest.validation_loss:.6f}."
        return ControllerDecision(
            action=AdaptiveAction.CONTINUE_TRAINING,
            confidence=0.6,
            reasons=(reason,),
            signals=signals,
        )

    def _degradation_decision(
        self,
        signals: ControllerSignals,
        context: TrainingContext,
    ) -> ControllerDecision:
        if self._config.degradation_action == AdaptiveAction.ROLLBACK:
            target_commit_id = context.best_commit_id
            return ControllerDecision(
                action=AdaptiveAction.ROLLBACK,
                confidence=0.9,
                reasons=(
                    "Validation loss degraded beyond configured patience and minimum delta.",
                    "Rollback targets the best known commit and preserves current branch history.",
                ),
                signals=signals,
                parameters={"target_commit_id": target_commit_id},
            )

        return self._decrease_learning_rate_decision(signals, context)

    def _overfitting_decision(
        self,
        signals: ControllerSignals,
        context: TrainingContext,
    ) -> ControllerDecision:
        if (
            context.frozen_layers
            or self._config.overfitting_action == AdaptiveAction.DECREASE_LEARNING_RATE
        ):
            return self._decrease_learning_rate_decision(signals, context)

        return ControllerDecision(
            action=AdaptiveAction.FREEZE_LAYERS,
            confidence=0.82,
            reasons=(
                "Training and validation curves show an overfitting gap above policy threshold.",
                "Freezing feature layers can reduce destructive updates before further adaptation.",
            ),
            signals=signals,
            parameters={"layer_selector": "features"},
        )

    def _unfreeze_decision(self, signals: ControllerSignals) -> ControllerDecision:
        return ControllerDecision(
            action=AdaptiveAction.UNFREEZE_LAYERS,
            confidence=0.74,
            reasons=(
                "Validation loss improved consistently while layers are frozen.",
                "Unfreezing allows controlled fine-tuning after stabilization.",
            ),
            signals=signals,
            parameters={"layer_selector": "all"},
        )

    def _increase_learning_rate_decision(
        self,
        signals: ControllerSignals,
        context: TrainingContext,
    ) -> ControllerDecision:
        current_lr = _resolve_learning_rate(context)
        next_lr = min(
            current_lr * self._config.learning_rate_increase_factor,
            self._config.maximum_learning_rate,
        )
        return ControllerDecision(
            action=AdaptiveAction.INCREASE_LEARNING_RATE,
            confidence=0.68,
            reasons=(
                "Training accuracy remains low after the configured minimum number of epochs.",
                "Increasing learning rate can help escape underfitting caused by "
                "overly small updates.",
            ),
            signals=signals,
            parameters={"learning_rate": next_lr},
        )

    def _plateau_decision(
        self,
        signals: ControllerSignals,
        context: TrainingContext,
    ) -> ControllerDecision:
        if self._config.plateau_action == AdaptiveAction.DECREASE_LEARNING_RATE:
            return self._decrease_learning_rate_decision(signals, context)

        return ControllerDecision(
            action=AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
            confidence=0.72,
            reasons=(
                "Validation loss is within the plateau threshold across the configured window.",
                "Creating an experimental branch preserves the stable line while "
                "exploring alternatives.",
            ),
            signals=signals,
            parameters={
                "source_commit_id": context.current_commit_id,
                "source_branch": context.branch_name,
            },
        )

    def _decrease_learning_rate_decision(
        self,
        signals: ControllerSignals,
        context: TrainingContext,
    ) -> ControllerDecision:
        current_lr = _resolve_learning_rate(context)
        next_lr = max(
            current_lr * self._config.learning_rate_decrease_factor,
            self._config.minimum_learning_rate,
        )
        return ControllerDecision(
            action=AdaptiveAction.DECREASE_LEARNING_RATE,
            confidence=0.78,
            reasons=(
                "Policy selected a conservative learning-rate reduction.",
                "This action keeps training on the current branch without mutating checkpoints.",
            ),
            signals=signals,
            parameters={"learning_rate": next_lr},
        )

    def _detect_degradation(self, metrics: Sequence[MetricPoint]) -> bool:
        window_size = self._config.degradation_patience + 1
        if len(metrics) < window_size:
            return False

        previous_best = min(
            point.validation_loss for point in metrics[: -self._config.degradation_patience]
        )
        recent = metrics[-self._config.degradation_patience :]
        return all(
            point.validation_loss - previous_best >= self._config.degradation_min_delta
            for point in recent
        )

    def _detect_plateau(self, metrics: Sequence[MetricPoint]) -> bool:
        if len(metrics) < self._config.plateau_window:
            return False

        recent = metrics[-self._config.plateau_window :]
        losses = [point.validation_loss for point in recent]
        return max(losses) - min(losses) <= self._config.plateau_min_delta

    def _detect_overfitting(self, metrics: Sequence[MetricPoint]) -> bool:
        if len(metrics) < 2:
            return False

        latest = metrics[-1]
        previous = metrics[-2]
        gap = _generalization_gap(latest)
        validation_worsened = latest.validation_loss - previous.validation_loss
        return (
            gap is not None
            and gap >= self._config.overfitting_gap
            and validation_worsened >= self._config.overfitting_validation_delta
        )

    def _detect_underfitting(self, metrics: Sequence[MetricPoint]) -> bool:
        if len(metrics) < self._config.underfitting_min_epochs:
            return False

        latest = metrics[-1]
        return (
            latest.train_accuracy is not None
            and latest.train_accuracy <= self._config.underfitting_max_train_accuracy
        )

    def _detect_stable_improvement(self, metrics: Sequence[MetricPoint]) -> bool:
        if len(metrics) < self._config.stable_improvement_window:
            return False

        recent = metrics[-self._config.stable_improvement_window :]
        return all(
            previous.validation_loss - current.validation_loss
            >= self._config.stable_improvement_min_delta
            for previous, current in pairwise(recent)
        )


def _latest_validation_delta(metrics: Sequence[MetricPoint]) -> float | None:
    if len(metrics) < 2:
        return None
    return metrics[-1].validation_loss - metrics[-2].validation_loss


def _generalization_gap(metric: MetricPoint) -> float | None:
    if metric.train_loss <= 0:
        return None
    return metric.validation_loss - metric.train_loss


def _resolve_learning_rate(context: TrainingContext) -> float:
    return context.current_learning_rate if context.current_learning_rate is not None else 1e-3
