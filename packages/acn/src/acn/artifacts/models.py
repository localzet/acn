from typing import Any, TypedDict


class CheckpointArtifactPayload(TypedDict):
    epoch: int
    global_step: int
    best_validation_loss: float | None
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None
