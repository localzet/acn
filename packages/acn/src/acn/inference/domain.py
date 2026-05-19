from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    id: str
    epoch: int
    accuracy: float
    validation_loss: float
    stable: bool
    artifact_uri: str | None = None
    mlflow_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    predicted_class: str
    confidence: float
    checkpoint_id: str
    model_version: str
    latency_ms: float


@dataclass(frozen=True, slots=True)
class InferenceComparisonResult:
    baseline: InferenceResult
    candidate: InferenceResult
