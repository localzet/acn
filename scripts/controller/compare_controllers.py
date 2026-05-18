from acn.controller import (
    AdaptiveController,
    MetricPoint,
    NeuralAdaptivePolicy,
    NeuralControllerState,
    TrainingContext,
)


def main() -> None:
    metrics = (
        MetricPoint(epoch=1, train_loss=0.8, validation_loss=0.72, learning_rate=1e-3),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.78, learning_rate=1e-3),
        MetricPoint(epoch=3, train_loss=0.6, validation_loss=0.83, learning_rate=1e-3),
    )
    context = TrainingContext(
        branch_name="main",
        current_commit_id="cmt_current",
        best_commit_id="cmt_best",
        current_learning_rate=1e-3,
    )
    rule_based = AdaptiveController().decide(metrics=metrics, context=context)
    neural = NeuralAdaptivePolicy().decide(
        metrics=metrics,
        context=context,
        state=NeuralControllerState(forgetting_score=0.2, rollback_count=1),
    )

    print(f"rule_based={rule_based.action.value} confidence={rule_based.confidence:.3f}")
    print(f"neural={neural.action.value} confidence={neural.confidence:.3f}")
    print("neural_reasons:")
    for reason in neural.reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
