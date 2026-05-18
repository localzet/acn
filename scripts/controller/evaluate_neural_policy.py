from acn.controller import (
    AdaptiveAction,
    MetricPoint,
    NeuralAdaptivePolicy,
    NeuralControllerState,
    PolicyTrainingExample,
    TrainingContext,
    evaluate_policy_examples,
)


def main() -> None:
    context = TrainingContext(branch_name="main", current_learning_rate=1e-3)
    examples = (
        PolicyTrainingExample(
            metrics=(MetricPoint(epoch=1, train_loss=0.8, validation_loss=0.82),),
            context=context,
            state=NeuralControllerState(),
            action=AdaptiveAction.CONTINUE_TRAINING,
        ),
        PolicyTrainingExample(
            metrics=(MetricPoint(epoch=1, train_loss=0.5, validation_loss=0.9),),
            context=context,
            state=NeuralControllerState(forgetting_score=0.5, rollback_count=1),
            action=AdaptiveAction.ROLLBACK,
        ),
    )
    result = evaluate_policy_examples(policy=NeuralAdaptivePolicy(), examples=examples)
    print(f"accuracy={result.accuracy:.3f}")
    print(f"confusion={dict(result.confusion)}")


if __name__ == "__main__":
    main()
