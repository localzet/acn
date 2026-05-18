from pathlib import Path

import pytest
import torch

from acn.controller import (
    ACTION_ORDER,
    AdaptiveAction,
    MetricPoint,
    NeuralAdaptivePolicy,
    NeuralControllerState,
    NeuralPolicyConfig,
    OfflineTrainingConfig,
    PolicyNetwork,
    PolicyTrainingExample,
    TrainingContext,
    build_policy_features,
    evaluate_policy_examples,
    train_policy_offline,
)


def _metrics() -> tuple[MetricPoint, ...]:
    return (
        MetricPoint(epoch=1, train_loss=0.8, validation_loss=0.82, learning_rate=1e-3),
        MetricPoint(epoch=2, train_loss=0.7, validation_loss=0.78, learning_rate=1e-3),
    )


def _context() -> TrainingContext:
    return TrainingContext(
        branch_name="main",
        current_commit_id="cmt_current",
        best_commit_id="cmt_best",
        current_learning_rate=1e-3,
    )


def _examples() -> tuple[PolicyTrainingExample, ...]:
    return (
        PolicyTrainingExample(
            metrics=_metrics(),
            context=_context(),
            state=NeuralControllerState(branch_history_length=2),
            action=AdaptiveAction.CONTINUE_TRAINING,
        ),
        PolicyTrainingExample(
            metrics=(MetricPoint(epoch=2, train_loss=0.6, validation_loss=0.9),),
            context=_context(),
            state=NeuralControllerState(forgetting_score=0.4, rollback_count=1),
            action=AdaptiveAction.ROLLBACK,
        ),
        PolicyTrainingExample(
            metrics=(MetricPoint(epoch=2, train_loss=0.5, validation_loss=0.7),),
            context=_context(),
            state=NeuralControllerState(gradient_norm=10.0),
            action=AdaptiveAction.DECREASE_LEARNING_RATE,
        ),
        PolicyTrainingExample(
            metrics=(MetricPoint(epoch=2, train_loss=1.8, validation_loss=1.9),),
            context=_context(),
            state=NeuralControllerState(adaptation_latency=5),
            action=AdaptiveAction.INCREASE_LEARNING_RATE,
        ),
    )


def test_policy_features_include_required_inputs() -> None:
    features = build_policy_features(
        metrics=_metrics(),
        context=_context(),
        state=NeuralControllerState(
            forgetting_score=0.2,
            rollback_count=3,
            adaptation_latency=4,
            gradient_norm=1.5,
            branch_history_length=7,
            branch_divergence_depth=2,
        ),
    )

    assert features.shape == (10,)
    assert features[2].item() == pytest.approx(0.2)
    assert features[3].item() == 3.0
    assert features[7].item() == 7.0


def test_policy_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        NeuralPolicyConfig(hidden_size=0)

    with pytest.raises(ValueError, match="confidence_threshold"):
        NeuralPolicyConfig(confidence_threshold=1.5)


def test_offline_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="epochs"):
        OfflineTrainingConfig(epochs=0)

    with pytest.raises(ValueError, match="batch_size"):
        OfflineTrainingConfig(batch_size=0)


def test_neural_policy_can_decide_with_confident_network() -> None:
    network = PolicyNetwork(hidden_size=8, dropout=0.0)
    for parameter in network.parameters():
        parameter.data.zero_()
    final_layer = network.layers[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    final_layer.bias.data[ACTION_ORDER.index(AdaptiveAction.ROLLBACK)] = 10.0
    policy = NeuralAdaptivePolicy(
        network=network,
        config=NeuralPolicyConfig(confidence_threshold=0.5),
    )

    decision = policy.decide(metrics=_metrics(), context=_context())

    assert decision.action is AdaptiveAction.ROLLBACK
    assert decision.parameters["target_commit_id"] == "cmt_best"
    assert "Top action probabilities" in decision.reasons[1]


def test_neural_policy_falls_back_when_confidence_is_low() -> None:
    network = PolicyNetwork(hidden_size=8, dropout=0.0)
    for parameter in network.parameters():
        parameter.data.zero_()
    policy = NeuralAdaptivePolicy(
        network=network,
        config=NeuralPolicyConfig(confidence_threshold=0.95),
    )

    decision = policy.decide(metrics=_metrics(), context=_context())

    assert decision.action is AdaptiveAction.CONTINUE_TRAINING
    assert "below threshold" in decision.reasons[0]


def test_offline_training_updates_policy_and_evaluates() -> None:
    policy = NeuralAdaptivePolicy(config=NeuralPolicyConfig(hidden_size=16, dropout=0.0))

    result = train_policy_offline(
        policy=policy,
        examples=_examples(),
        config=OfflineTrainingConfig(epochs=3, batch_size=2, learning_rate=1e-2),
    )
    evaluation = evaluate_policy_examples(policy=policy, examples=_examples())

    assert result.loss >= 0.0
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.confusion


def test_policy_save_and_load_round_trip(tmp_path: Path) -> None:
    source = NeuralAdaptivePolicy(config=NeuralPolicyConfig(hidden_size=8, dropout=0.0))
    path = tmp_path / "policy.pt"

    source.save(path)
    restored = NeuralAdaptivePolicy(config=NeuralPolicyConfig(hidden_size=8, dropout=0.0))
    restored.load(path)

    for source_parameter, restored_parameter in zip(
        source.network.parameters(),
        restored.network.parameters(),
        strict=True,
    ):
        assert torch.equal(source_parameter, restored_parameter)
