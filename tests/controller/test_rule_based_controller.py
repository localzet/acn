import logging

from acn.controller import AdaptiveAction, AdaptiveController, MetricPoint, TrainingContext
from acn.controller.policies import RuleBasedAdaptivePolicy, RuleBasedPolicyConfig


def _context(*, frozen_layers: bool = False) -> TrainingContext:
    return TrainingContext(
        branch_name="main",
        current_commit_id="cmt_latest",
        best_commit_id="cmt_best",
        frozen_layers=frozen_layers,
        current_learning_rate=1e-3,
    )


def test_degradation_selects_rollback() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.9, validation_loss=0.80),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.76),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.83),
        MetricPoint(epoch=4, train_loss=0.5, validation_loss=0.86),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.ROLLBACK
    assert decision.parameters["target_commit_id"] == "cmt_best"
    assert decision.signals.degradation
    assert decision.reasons


def test_plateau_can_create_experimental_branch() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.8, validation_loss=0.701),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.700),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.702),
        MetricPoint(epoch=4, train_loss=0.5, validation_loss=0.699),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH
    assert decision.parameters["source_commit_id"] == "cmt_latest"
    assert decision.signals.plateau


def test_plateau_policy_can_decrease_learning_rate() -> None:
    policy = RuleBasedAdaptivePolicy(
        RuleBasedPolicyConfig(plateau_action=AdaptiveAction.DECREASE_LEARNING_RATE)
    )
    controller = AdaptiveController(policy)
    metrics = [
        MetricPoint(epoch=1, train_loss=0.8, validation_loss=0.701),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.700),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.702),
        MetricPoint(epoch=4, train_loss=0.5, validation_loss=0.699),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.DECREASE_LEARNING_RATE
    assert decision.parameters["learning_rate"] == 5e-4


def test_overfitting_freezes_layers() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.55, validation_loss=0.64),
        MetricPoint(epoch=2, train_loss=0.42, validation_loss=0.68),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.FREEZE_LAYERS
    assert decision.signals.overfitting
    assert decision.parameters["layer_selector"] == "features"


def test_overfitting_with_frozen_layers_decreases_learning_rate() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.55, validation_loss=0.64),
        MetricPoint(epoch=2, train_loss=0.42, validation_loss=0.68),
    ]

    decision = controller.decide(metrics=metrics, context=_context(frozen_layers=True))

    assert decision.action is AdaptiveAction.DECREASE_LEARNING_RATE
    assert decision.parameters["learning_rate"] == 5e-4


def test_underfitting_increases_learning_rate() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=1.8, validation_loss=1.9, train_accuracy=0.30),
        MetricPoint(epoch=2, train_loss=1.7, validation_loss=1.8, train_accuracy=0.34),
        MetricPoint(epoch=3, train_loss=1.6, validation_loss=1.7, train_accuracy=0.35),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.INCREASE_LEARNING_RATE
    assert decision.parameters["learning_rate"] == 0.00125
    assert decision.signals.underfitting


def test_stable_improvement_unfreezes_layers() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.9, validation_loss=0.90),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.87),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.84),
    ]

    decision = controller.decide(metrics=metrics, context=_context(frozen_layers=True))

    assert decision.action is AdaptiveAction.UNFREEZE_LAYERS
    assert decision.signals.stable_improvement


def test_no_signal_continues_training() -> None:
    controller = AdaptiveController()
    metrics = [
        MetricPoint(epoch=1, train_loss=0.9, validation_loss=0.95, train_accuracy=0.50),
        MetricPoint(epoch=2, train_loss=0.8, validation_loss=0.90, train_accuracy=0.55),
    ]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.CONTINUE_TRAINING


def test_controller_logs_decision(caplog: object) -> None:
    caplog.set_level(logging.INFO, logger="acn.controller.controller")
    controller = AdaptiveController()
    metrics = [MetricPoint(epoch=1, train_loss=0.9, validation_loss=0.95)]

    decision = controller.decide(metrics=metrics, context=_context())

    assert decision.action is AdaptiveAction.CONTINUE_TRAINING
    assert "controller.decision" in caplog.text
