from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from acn.continual.dataset import parse_image_sample


@dataclass(frozen=True, slots=True)
class ContinualMetrics:
    stage_id: str
    old_class_retention: float | None
    new_class_adaptation: float | None
    forgetting_score: float
    adaptation_latency: int | None
    per_class_accuracy: Mapping[int, float]


class ForgettingEvaluator:
    def __init__(self, *, adaptation_threshold: float = 0.8) -> None:
        self._adaptation_threshold = adaptation_threshold
        self._best_accuracy_by_class: dict[int, float] = {}
        self._introduced_at_stage: dict[int, int] = {}
        self._adapted_at_stage: dict[int, int] = {}
        self._stage_index = 0

    def evaluate_predictions(
        self,
        *,
        stage_id: str,
        introduced_class_ids: Sequence[int],
        old_class_ids: Sequence[int],
        targets: Sequence[int],
        predictions: Sequence[int],
    ) -> ContinualMetrics:
        self._stage_index += 1
        per_class_accuracy = _per_class_accuracy(targets, predictions)
        for class_id in introduced_class_ids:
            self._introduced_at_stage.setdefault(class_id, self._stage_index)

        old_class_retention = _mean_accuracy(per_class_accuracy, old_class_ids)
        new_class_adaptation = _mean_accuracy(per_class_accuracy, introduced_class_ids)
        forgetting_score = self._forgetting_score(per_class_accuracy, old_class_ids)
        adaptation_latency = self._adaptation_latency(per_class_accuracy, introduced_class_ids)
        self._update_best_accuracy(per_class_accuracy)

        return ContinualMetrics(
            stage_id=stage_id,
            old_class_retention=old_class_retention,
            new_class_adaptation=new_class_adaptation,
            forgetting_score=forgetting_score,
            adaptation_latency=adaptation_latency,
            per_class_accuracy=per_class_accuracy,
        )

    def _forgetting_score(
        self,
        per_class_accuracy: Mapping[int, float],
        old_class_ids: Sequence[int],
    ) -> float:
        deltas = []
        for class_id in old_class_ids:
            if class_id not in per_class_accuracy:
                continue
            previous_best = self._best_accuracy_by_class.get(class_id, per_class_accuracy[class_id])
            deltas.append(max(0.0, previous_best - per_class_accuracy[class_id]))
        return sum(deltas) / len(deltas) if deltas else 0.0

    def _adaptation_latency(
        self,
        per_class_accuracy: Mapping[int, float],
        introduced_class_ids: Sequence[int],
    ) -> int | None:
        latencies: list[int] = []
        for class_id in introduced_class_ids:
            accuracy = per_class_accuracy.get(class_id)
            if accuracy is None or accuracy < self._adaptation_threshold:
                continue
            self._adapted_at_stage.setdefault(class_id, self._stage_index)
            introduced_at = self._introduced_at_stage[class_id]
            latencies.append(self._adapted_at_stage[class_id] - introduced_at)
        if not introduced_class_ids:
            return None
        return max(latencies) if latencies else None

    def _update_best_accuracy(self, per_class_accuracy: Mapping[int, float]) -> None:
        for class_id, accuracy in per_class_accuracy.items():
            previous = self._best_accuracy_by_class.get(class_id, 0.0)
            self._best_accuracy_by_class[class_id] = max(previous, accuracy)


class ContinualEvaluationPipeline:
    def __init__(self, evaluator: ForgettingEvaluator | None = None) -> None:
        self._evaluator = evaluator or ForgettingEvaluator()

    @torch.inference_mode()
    def evaluate_model(
        self,
        *,
        model: nn.Module,
        dataloader: DataLoader[Any],
        stage_id: str,
        introduced_class_ids: Sequence[int],
        old_class_ids: Sequence[int],
        device: torch.device | str = "cpu",
    ) -> ContinualMetrics:
        resolved_device = torch.device(device)
        model.to(resolved_device)
        model.eval()
        targets: list[int] = []
        predictions: list[int] = []

        for batch in dataloader:
            inputs, batch_targets = _parse_batch(batch)
            logits = model(inputs.to(resolved_device))
            predictions.extend(int(value) for value in logits.argmax(dim=1).cpu().tolist())
            targets.extend(int(value) for value in batch_targets.cpu().tolist())

        return self._evaluator.evaluate_predictions(
            stage_id=stage_id,
            introduced_class_ids=introduced_class_ids,
            old_class_ids=old_class_ids,
            targets=targets,
            predictions=predictions,
        )


def _parse_batch(batch: object) -> tuple[Tensor, Tensor]:
    if isinstance(batch, dict):
        inputs = batch.get("inputs")
        targets = batch.get("targets")
        if isinstance(inputs, Tensor) and isinstance(targets, Tensor):
            return inputs, targets
    if isinstance(batch, tuple | list) and len(batch) >= 2:
        inputs = batch[0]
        targets = batch[1]
        if isinstance(inputs, Tensor) and isinstance(targets, Tensor):
            return inputs, targets

    image, target = parse_image_sample(batch)
    return image.unsqueeze(0), torch.tensor([target], dtype=torch.long)


def _per_class_accuracy(targets: Sequence[int], predictions: Sequence[int]) -> dict[int, float]:
    if len(targets) != len(predictions):
        msg = "Targets and predictions must have the same length."
        raise ValueError(msg)

    totals: dict[int, int] = defaultdict(int)
    correct: dict[int, int] = defaultdict(int)
    for target, prediction in zip(targets, predictions, strict=True):
        totals[target] += 1
        if target == prediction:
            correct[target] += 1

    return {class_id: correct[class_id] / total for class_id, total in totals.items()}


def _mean_accuracy(
    per_class_accuracy: Mapping[int, float],
    class_ids: Sequence[int],
) -> float | None:
    values = [
        per_class_accuracy[class_id] for class_id in class_ids if class_id in per_class_accuracy
    ]
    if not values:
        return None
    return sum(values) / len(values)
