from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from acn.controller.domain import (
    AdaptiveAction,
    ControllerDecision,
    MetricPoint,
    TrainingContext,
)
from acn.controller.policies import RuleBasedAdaptivePolicy

ACTION_ORDER: tuple[AdaptiveAction, ...] = (
    AdaptiveAction.CONTINUE_TRAINING,
    AdaptiveAction.ROLLBACK,
    AdaptiveAction.DECREASE_LEARNING_RATE,
    AdaptiveAction.INCREASE_LEARNING_RATE,
    AdaptiveAction.FREEZE_LAYERS,
    AdaptiveAction.UNFREEZE_LAYERS,
    AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH,
)


@dataclass(frozen=True, slots=True)
class NeuralControllerState:
    forgetting_score: float = 0.0
    rollback_count: int = 0
    adaptation_latency: int = 0
    gradient_norm: float = 0.0
    branch_history_length: int = 0
    branch_divergence_depth: int = 0


@dataclass(frozen=True, slots=True)
class NeuralPolicyConfig:
    hidden_size: int = 32
    dropout: float = 0.1
    confidence_threshold: float = 0.55
    learning_rate_decrease_factor: float = 0.5
    learning_rate_increase_factor: float = 1.25

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            msg = "hidden_size must be positive."
            raise ValueError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = "dropout must be in the [0.0, 1.0) range."
            raise ValueError(msg)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            msg = "confidence_threshold must be in the [0.0, 1.0] range."
            raise ValueError(msg)
        if self.learning_rate_decrease_factor <= 0.0:
            msg = "learning_rate_decrease_factor must be positive."
            raise ValueError(msg)
        if self.learning_rate_increase_factor <= 0.0:
            msg = "learning_rate_increase_factor must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PolicyTrainingExample:
    metrics: tuple[MetricPoint, ...]
    context: TrainingContext
    state: NeuralControllerState
    action: AdaptiveAction


@dataclass(frozen=True, slots=True)
class OfflineTrainingConfig:
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 1e-3
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            msg = "epochs must be positive."
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if self.learning_rate <= 0.0:
            msg = "learning_rate must be positive."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PolicyTrainingResult:
    loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class PolicyEvaluationResult:
    accuracy: float
    confusion: Mapping[str, Mapping[str, int]]


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_size: int = 10,
        hidden_size: int = 32,
        output_size: int = len(ACTION_ORDER),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.layers(features))


class NeuralAdaptivePolicy:
    def __init__(
        self,
        *,
        network: PolicyNetwork | None = None,
        config: NeuralPolicyConfig | None = None,
        fallback_policy: RuleBasedAdaptivePolicy | None = None,
    ) -> None:
        self._config = config or NeuralPolicyConfig()
        self._network = network or PolicyNetwork(
            hidden_size=self._config.hidden_size,
            dropout=self._config.dropout,
        )
        self._fallback_policy = fallback_policy or RuleBasedAdaptivePolicy()

    @property
    def network(self) -> PolicyNetwork:
        return self._network

    def decide(
        self,
        *,
        metrics: Sequence[MetricPoint],
        context: TrainingContext,
        state: NeuralControllerState | None = None,
    ) -> ControllerDecision:
        if not metrics:
            fallback = self._fallback_policy.decide(metrics=metrics, context=context)
            return _fallback_decision(fallback, "Neural policy requires at least one metric point.")

        self._network.eval()
        features = build_policy_features(metrics=metrics, context=context, state=state).to(
            _network_device(self._network)
        )
        with torch.inference_mode():
            logits = self._network(features.unsqueeze(0))
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        confidence, action_index = torch.max(probabilities, dim=0)
        action = ACTION_ORDER[int(action_index.item())]
        confidence_value = float(confidence.item())
        if confidence_value < self._config.confidence_threshold:
            fallback = self._fallback_policy.decide(metrics=metrics, context=context)
            return _fallback_decision(
                fallback,
                f"Neural confidence {confidence_value:.3f} is below threshold.",
            )

        reasons = _explain_prediction(action, probabilities)
        return ControllerDecision(
            action=action,
            confidence=confidence_value,
            reasons=reasons,
            signals=self._fallback_policy.analyze(metrics),
            parameters=_action_parameters(action, context, self._config),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._network.state_dict(), path)

    def load(self, path: Path, *, map_location: str | torch.device = "cpu") -> None:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self._network.load_state_dict(state_dict)


class PolicyExampleDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, examples: Sequence[PolicyTrainingExample]) -> None:
        self._samples = tuple(
            (
                build_policy_features(
                    metrics=example.metrics,
                    context=example.context,
                    state=example.state,
                ),
                torch.tensor(ACTION_ORDER.index(example.action), dtype=torch.long),
            )
            for example in examples
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._samples[index]


def train_policy_offline(
    *,
    policy: NeuralAdaptivePolicy,
    examples: Sequence[PolicyTrainingExample],
    config: OfflineTrainingConfig | None = None,
) -> PolicyTrainingResult:
    resolved_config = config or OfflineTrainingConfig()
    if not examples:
        msg = "Offline policy training requires at least one example."
        raise ValueError(msg)

    device = torch.device(resolved_config.device)
    policy.network.to(device)
    policy.network.train()
    dataset = PolicyExampleDataset(examples)
    dataloader = DataLoader(dataset, batch_size=resolved_config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(policy.network.parameters(), lr=resolved_config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    last_loss = 0.0
    correct = 0
    total = 0

    for _epoch in range(resolved_config.epochs):
        correct = 0
        total = 0
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = policy.network(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().item())
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

    return PolicyTrainingResult(loss=last_loss, accuracy=correct / total)


def evaluate_policy_examples(
    *,
    policy: NeuralAdaptivePolicy,
    examples: Sequence[PolicyTrainingExample],
) -> PolicyEvaluationResult:
    if not examples:
        return PolicyEvaluationResult(accuracy=0.0, confusion={})

    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    for example in examples:
        decision = policy.decide(
            metrics=example.metrics,
            context=example.context,
            state=example.state,
        )
        expected = example.action.value
        predicted = decision.action.value
        confusion.setdefault(expected, {})
        confusion[expected][predicted] = confusion[expected].get(predicted, 0) + 1
        if decision.action is example.action:
            correct += 1

    return PolicyEvaluationResult(accuracy=correct / len(examples), confusion=confusion)


def build_policy_features(
    *,
    metrics: Sequence[MetricPoint],
    context: TrainingContext,
    state: NeuralControllerState | None = None,
) -> Tensor:
    latest = metrics[-1] if metrics else MetricPoint(epoch=0, train_loss=0.0, validation_loss=0.0)
    resolved_state = state or NeuralControllerState()
    learning_rate = latest.learning_rate or context.current_learning_rate or 0.0
    features = [
        latest.train_loss,
        latest.validation_loss,
        resolved_state.forgetting_score,
        float(resolved_state.rollback_count),
        float(resolved_state.adaptation_latency),
        resolved_state.gradient_norm,
        learning_rate,
        float(resolved_state.branch_history_length),
        float(resolved_state.branch_divergence_depth),
        1.0 if context.frozen_layers else 0.0,
    ]
    return torch.tensor(features, dtype=torch.float32)


def _fallback_decision(decision: ControllerDecision, reason: str) -> ControllerDecision:
    return ControllerDecision(
        action=decision.action,
        confidence=decision.confidence,
        reasons=(reason, *decision.reasons),
        signals=decision.signals,
        parameters=decision.parameters,
    )


def _network_device(network: nn.Module) -> torch.device:
    try:
        return next(network.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _explain_prediction(action: AdaptiveAction, probabilities: Tensor) -> tuple[str, ...]:
    top_probabilities, top_indices = torch.topk(probabilities, k=min(3, len(ACTION_ORDER)))
    alternatives = ", ".join(
        f"{ACTION_ORDER[int(index.item())].value}={float(probability.item()):.3f}"
        for probability, index in zip(top_probabilities, top_indices, strict=True)
    )
    return (
        f"Neural policy selected {action.value}.",
        f"Top action probabilities: {alternatives}.",
    )


def _action_parameters(
    action: AdaptiveAction,
    context: TrainingContext,
    config: NeuralPolicyConfig,
) -> dict[str, str | float | None]:
    if action is AdaptiveAction.ROLLBACK:
        return {"target_commit_id": context.best_commit_id}
    if action is AdaptiveAction.DECREASE_LEARNING_RATE:
        current_lr = context.current_learning_rate or 1e-3
        return {"learning_rate": current_lr * config.learning_rate_decrease_factor}
    if action is AdaptiveAction.INCREASE_LEARNING_RATE:
        current_lr = context.current_learning_rate or 1e-3
        return {"learning_rate": current_lr * config.learning_rate_increase_factor}
    if action is AdaptiveAction.FREEZE_LAYERS:
        return {"layer_selector": "features"}
    if action is AdaptiveAction.UNFREEZE_LAYERS:
        return {"layer_selector": "all"}
    if action is AdaptiveAction.CREATE_EXPERIMENTAL_BRANCH:
        return {
            "source_commit_id": context.current_commit_id,
            "source_branch": context.branch_name,
        }
    return {}
