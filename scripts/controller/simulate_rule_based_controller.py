from acn.controller import AdaptiveController, MetricPoint, RuleBasedAdaptivePolicy, TrainingContext
from acn.controller.example_policies import exploratory_policy


def main() -> None:
    controller = AdaptiveController(policy=RuleBasedAdaptivePolicy(exploratory_policy()))
    metrics = [
        MetricPoint(epoch=1, train_loss=1.20, validation_loss=1.25, train_accuracy=0.42),
        MetricPoint(epoch=2, train_loss=0.92, validation_loss=1.03, train_accuracy=0.58),
        MetricPoint(epoch=3, train_loss=0.73, validation_loss=0.94, train_accuracy=0.67),
        MetricPoint(epoch=4, train_loss=0.62, validation_loss=0.94, train_accuracy=0.71),
        MetricPoint(epoch=5, train_loss=0.58, validation_loss=0.945, train_accuracy=0.73),
        MetricPoint(epoch=6, train_loss=0.55, validation_loss=0.942, train_accuracy=0.74),
    ]
    context = TrainingContext(
        branch_name="main",
        current_commit_id="cmt_latest",
        best_commit_id="cmt_best",
        current_learning_rate=1e-3,
    )

    decision = controller.decide(metrics=metrics, context=context)
    print(f"action={decision.action.value}")
    print(f"confidence={decision.confidence:.2f}")
    for reason in decision.reasons:
        print(f"- {reason}")
    if decision.parameters:
        print(f"parameters={decision.parameters}")


if __name__ == "__main__":
    main()
