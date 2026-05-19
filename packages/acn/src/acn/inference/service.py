from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Protocol

import torch
from torch import Tensor, nn

from acn.inference.domain import InferenceComparisonResult, InferenceResult


class ClassifierModel(Protocol):
    def eval(self) -> object: ...

    def load_state_dict(self, state_dict: Mapping[str, object]) -> object: ...

    def __call__(self, inputs: Tensor) -> Tensor: ...


class InferenceService:
    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        class_names: tuple[str, ...],
        device: torch.device,
    ) -> None:
        self._model_factory = model_factory
        self._class_names = class_names
        self._device = device

    @torch.inference_mode()
    def predict(
        self,
        *,
        image: Tensor,
        model_state: Mapping[str, object],
        checkpoint_id: str,
        model_version: str,
    ) -> InferenceResult:
        started = perf_counter()
        model = self._model_factory().to(self._device)
        model.load_state_dict(model_state)
        model.eval()
        logits = model(image.to(self._device).unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())
        latency_ms = (perf_counter() - started) * 1000.0
        return InferenceResult(
            predicted_class=self._class_names[index],
            confidence=float(probabilities[index].item()),
            checkpoint_id=checkpoint_id,
            model_version=model_version,
            latency_ms=latency_ms,
        )

    def compare(
        self,
        *,
        image: Tensor,
        baseline_state: Mapping[str, object],
        baseline_checkpoint_id: str,
        candidate_state: Mapping[str, object],
        candidate_checkpoint_id: str,
    ) -> InferenceComparisonResult:
        return InferenceComparisonResult(
            baseline=self.predict(
                image=image,
                model_state=baseline_state,
                checkpoint_id=baseline_checkpoint_id,
                model_version="early",
            ),
            candidate=self.predict(
                image=image,
                model_state=candidate_state,
                checkpoint_id=candidate_checkpoint_id,
                model_version="selected",
            ),
        )
