from acn.controller import AdaptiveAction, MetricPoint, RuleBasedAdaptivePolicy, TrainingContext


def test_controller_selects_rollback_on_repeated_degradation() -> None:
    policy = RuleBasedAdaptivePolicy()
    metrics = (
        MetricPoint(epoch=1, train_loss=0.6, validation_loss=0.5),
        MetricPoint(epoch=2, train_loss=0.6, validation_loss=0.55),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.58),
    )

    decision = policy.decide(
        metrics=metrics,
        context=TrainingContext(branch_name="main", best_commit_id="cmt_best"),
    )

    assert decision.action is AdaptiveAction.ROLLBACK
    assert decision.parameters["target_commit_id"] == "cmt_best"


def test_controller_unfreezes_after_stable_improvement() -> None:
    policy = RuleBasedAdaptivePolicy()
    metrics = (
        MetricPoint(epoch=1, train_loss=0.7, validation_loss=0.8),
        MetricPoint(epoch=2, train_loss=0.65, validation_loss=0.76),
        MetricPoint(epoch=3, train_loss=0.62, validation_loss=0.72),
    )

    decision = policy.decide(
        metrics=metrics,
        context=TrainingContext(branch_name="main", frozen_layers=True),
    )

    assert decision.action is AdaptiveAction.UNFREEZE_LAYERS
    assert decision.parameters["layer_selector"] == "all"
