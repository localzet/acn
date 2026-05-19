"""Live visual adaptive-training demo.

This module is a presentation-oriented local demo. It trains a small CNN on a
lightweight visual airplane-vs-ship dataset and exposes state snapshots for the
FastAPI demo endpoints.
"""

import base64
import copy
import io
import json
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torch import Tensor, nn

from acn.artifacts import CheckpointArtifactPayload
from acn.config.settings import Settings
from acn.inference import InferenceComparisonResult, InferenceResult, InferenceService
from acn.inference.preprocessing import ImagePreprocessor
from acn.runtime import RuntimeStack, RuntimeStatus

ClassName = Literal["airplane", "ship"]
DemoStatus = Literal["idle", "running", "paused", "awaiting_approval", "completed", "failed"]


class VisualDemoError(RuntimeError):
    """Raised when the visual demo cannot execute the requested operation."""


class VisualDemoMetric(TypedDict):
    timestamp: str
    epoch: int
    trainLoss: float
    validationLoss: float
    accuracy: float
    learningRate: float
    stage: str


class VisualDemoCheckpoint(TypedDict):
    id: str
    epoch: int
    validationLoss: float
    accuracy: float
    stable: bool
    createdAt: str
    artifactUri: str | None
    sizeBytes: int | None
    mlflowRunId: str | None
    storage: str | None


class VisualDemoEvent(TypedDict):
    id: str
    level: str
    message: str
    createdAt: str


class VisualDemoPrediction(TypedDict):
    id: str
    image: str
    actualClass: str
    predictedClass: str
    confidence: float
    correct: bool


class VisualDemoDecision(TypedDict):
    id: str
    action: str
    status: str
    reason: str
    createdAt: str


class VisualDemoInference(TypedDict):
    predictedClass: str
    confidence: float
    checkpointId: str
    modelVersion: str
    latencyMs: float


class VisualDemoComparison(TypedDict):
    early: VisualDemoInference
    selected: VisualDemoInference


class VisualDemoSnapshot(TypedDict):
    status: DemoStatus
    autoMode: bool
    epoch: int
    stage: str
    controllerState: str
    currentBranch: str
    activeCheckpointId: str | None
    rollbackCount: int
    gpuUsage: dict[str, str | float | None]
    metrics: list[VisualDemoMetric]
    checkpoints: list[VisualDemoCheckpoint]
    predictions: list[VisualDemoPrediction]
    events: list[VisualDemoEvent]
    decisions: list[VisualDemoDecision]
    runtimeStatus: dict[str, dict[str, str | bool]]
    mlflowRunId: str | None
    artifacts: list[dict[str, str | int | float | bool | None]]
    inferenceHistory: list[VisualDemoInference]


@dataclass(slots=True)
class _CheckpointState:
    id: str
    epoch: int
    validation_loss: float
    accuracy: float
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, object]
    stable: bool
    created_at: str
    artifact_uri: str | None = None
    checksum: str | None = None
    size_bytes: int | None = None
    mlflow_run_id: str | None = None
    storage: str | None = None


@dataclass(slots=True)
class _VisualDemoState:
    status: DemoStatus = "idle"
    auto_mode: bool = True
    epoch: int = 0
    stage: str = "ready"
    controller_state: str = "idle"
    current_branch: str = "main"
    active_checkpoint_id: str | None = None
    rollback_count: int = 0
    metrics: list[VisualDemoMetric] = field(default_factory=list)
    checkpoints: list[_CheckpointState] = field(default_factory=list)
    predictions: list[VisualDemoPrediction] = field(default_factory=list)
    events: list[VisualDemoEvent] = field(default_factory=list)
    decisions: list[VisualDemoDecision] = field(default_factory=list)


class TinyVisualClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.classifier(self.features(inputs)))


class VisualDemoSession:
    def __init__(self, *, seed: int = 20260519) -> None:
        self._seed = seed
        self._lock = threading.RLock()
        self._state = _VisualDemoState()
        self._thread: threading.Thread | None = None
        self._pause_requested = False
        self._stop_requested = False
        self._approve_event = threading.Event()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = TinyVisualClassifier().to(self._device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=2e-3)
        self._validation = _build_dataset(samples=72, seed=seed + 1)
        self._runtime: RuntimeStack | None = None
        self._runtime_status: RuntimeStatus | None = None
        self._mlflow_run_id: str | None = None
        self._experiment_id = "exp_visual_demo"
        self._preprocessor = ImagePreprocessor(size=32)
        self._inference = InferenceService(
            model_factory=TinyVisualClassifier,
            class_names=("airplane", "ship"),
            device=self._device,
        )
        self._inference_history: list[VisualDemoInference] = []

    def configure_runtime(self, settings: Settings) -> None:
        if not settings.runtime_stack_enabled:
            return
        self._runtime = RuntimeStack(settings)
        self._runtime_status = self._runtime.initialize()

    def snapshot(self) -> VisualDemoSnapshot:
        with self._lock:
            return {
                "status": self._state.status,
                "autoMode": self._state.auto_mode,
                "epoch": self._state.epoch,
                "stage": self._state.stage,
                "controllerState": self._state.controller_state,
                "currentBranch": self._state.current_branch,
                "activeCheckpointId": self._state.active_checkpoint_id,
                "rollbackCount": self._state.rollback_count,
                "gpuUsage": _gpu_usage(self._device),
                "metrics": list(self._state.metrics),
                "checkpoints": [
                    {
                        "id": checkpoint.id,
                        "epoch": checkpoint.epoch,
                        "validationLoss": checkpoint.validation_loss,
                        "accuracy": checkpoint.accuracy,
                        "stable": checkpoint.stable,
                        "createdAt": checkpoint.created_at,
                        "artifactUri": checkpoint.artifact_uri,
                        "sizeBytes": checkpoint.size_bytes,
                        "mlflowRunId": checkpoint.mlflow_run_id,
                        "storage": checkpoint.storage,
                    }
                    for checkpoint in self._state.checkpoints
                ],
                "predictions": list(self._state.predictions),
                "events": list(self._state.events[-80:]),
                "decisions": list(self._state.decisions[-20:]),
                "runtimeStatus": _runtime_status_dict(self._runtime_status),
                "mlflowRunId": self._mlflow_run_id,
                "artifacts": self._runtime.artifact_browser() if self._runtime is not None else [],
                "inferenceHistory": list(self._inference_history[-20:]),
            }

    def start(self, *, auto_mode: bool = True) -> VisualDemoSnapshot:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                self._state.auto_mode = auto_mode
                return self.snapshot()
            self._reset(auto_mode=auto_mode)
            self._thread = threading.Thread(target=self._run, name="acn-visual-demo", daemon=True)
            self._thread.start()
            return self.snapshot()

    def pause(self) -> VisualDemoSnapshot:
        with self._lock:
            self._pause_requested = True
            if self._state.status == "running":
                self._state.status = "paused"
                self._event("info", "Training paused by operator.")
            return self.snapshot()

    def resume(self) -> VisualDemoSnapshot:
        with self._lock:
            self._pause_requested = False
            if self._state.status == "paused":
                self._state.status = "running"
                self._event("info", "Training resumed by operator.")
            return self.snapshot()

    def set_auto_mode(self, enabled: bool) -> VisualDemoSnapshot:
        with self._lock:
            self._state.auto_mode = enabled
            self._event("info", f"Controller mode changed to {'AUTO' if enabled else 'MANUAL'}.")
            if enabled and self._state.status == "awaiting_approval":
                self._approve_event.set()
            return self.snapshot()

    def approve(self, decision_id: str) -> VisualDemoSnapshot:
        with self._lock:
            for decision in self._state.decisions:
                if decision["id"] == decision_id and decision["status"] == "pending":
                    decision["status"] = "approved"
                    self._event("info", "Operator approved rollback.")
                    self._approve_event.set()
                    break
            return self.snapshot()

    def reject(self, decision_id: str) -> VisualDemoSnapshot:
        with self._lock:
            for decision in self._state.decisions:
                if decision["id"] == decision_id and decision["status"] == "pending":
                    decision["status"] = "denied"
                    self._state.status = "running"
                    self._event("warning", "Operator rejected rollback; training continues.")
                    self._approve_event.set()
                    break
            return self.snapshot()

    def rollback(self) -> VisualDemoSnapshot:
        with self._lock:
            self._restore_best_checkpoint()
            return self.snapshot()

    def predict_data_url(
        self,
        image_data_url: str,
        *,
        checkpoint_id: str | None = None,
    ) -> VisualDemoInference:
        checkpoint = self._resolve_checkpoint(checkpoint_id)
        result = self._inference.predict(
            image=self._preprocessor.from_data_url(image_data_url),
            model_state=checkpoint.model_state,
            checkpoint_id=checkpoint.id,
            model_version="selected" if checkpoint_id is not None else "active",
        )
        response = _inference_response(result)
        with self._lock:
            self._inference_history.append(response)
        return response

    def compare_data_url(
        self,
        image_data_url: str,
        *,
        baseline_checkpoint_id: str | None = None,
        candidate_checkpoint_id: str | None = None,
    ) -> VisualDemoComparison:
        baseline = self._resolve_checkpoint(baseline_checkpoint_id or "earliest")
        candidate = self._resolve_checkpoint(candidate_checkpoint_id or "latest")
        comparison = self._inference.compare(
            image=self._preprocessor.from_data_url(image_data_url),
            baseline_state=baseline.model_state,
            baseline_checkpoint_id=baseline.id,
            candidate_state=candidate.model_state,
            candidate_checkpoint_id=candidate.id,
        )
        return _comparison_response(comparison)

    def export_report(self) -> dict[str, str]:
        snapshot = self.snapshot()
        output_dir = Path("experiments/acn-guided-demo")
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report": output_dir / "guided_demo_summary.md",
            "metrics": output_dir / "metrics.json",
            "timeline": output_dir / "timeline.json",
            "finalModelInfo": output_dir / "final_model_info.json",
            "predictionResults": output_dir / "prediction_results.json",
            "modelMetadata": output_dir / "model_metadata.json",
            "screenshot": output_dir / "guided_demo_screenshot.svg",
        }
        paths["metrics"].write_text(json.dumps(snapshot["metrics"], indent=2), encoding="utf-8")
        paths["timeline"].write_text(json.dumps(snapshot["events"], indent=2), encoding="utf-8")
        paths["predictionResults"].write_text(
            json.dumps(snapshot["inferenceHistory"], indent=2),
            encoding="utf-8",
        )
        paths["modelMetadata"].write_text(
            json.dumps(snapshot["checkpoints"], indent=2),
            encoding="utf-8",
        )
        paths["finalModelInfo"].write_text(
            json.dumps(
                {
                    "bestModelVersion": snapshot["activeCheckpointId"],
                    "bestAccuracy": _best_accuracy(snapshot),
                    "rollbackCount": snapshot["rollbackCount"],
                    "totalCheckpoints": len(snapshot["checkpoints"]),
                    "mlflowRunId": snapshot["mlflowRunId"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["report"].write_text(_guided_report(snapshot), encoding="utf-8")
        paths["screenshot"].write_text(_screenshot_svg(snapshot), encoding="utf-8")
        return {key: str(path) for key, path in paths.items()}

    def _run(self) -> None:
        try:
            random.seed(self._seed)
            torch.manual_seed(self._seed)
            self._start_runtime_run()
            self._event("info", "Visual demo started: airplane vs ship classifier.")
            best_loss = float("inf")

            for epoch in range(1, 11):
                self._wait_if_paused()
                with self._lock:
                    self._state.status = "running"
                    self._state.epoch = epoch
                    self._state.stage = "degraded-lr-spike" if epoch == 5 else "adaptive-training"
                    self._state.controller_state = "observing"

                degraded = epoch == 5
                learning_rate = 0.35 if degraded else 2e-3
                _set_lr(self._optimizer, learning_rate)
                train_loss = self._train_epoch(epoch=epoch, degraded=degraded)
                validation_loss, accuracy = self._evaluate()
                self._record_metric(epoch, train_loss, validation_loss, accuracy, learning_rate)
                self._save_checkpoint(
                    epoch,
                    validation_loss,
                    accuracy,
                    stable=validation_loss <= best_loss,
                )
                self._refresh_predictions()

                if validation_loss < best_loss:
                    best_loss = validation_loss

                if degraded or validation_loss > best_loss + 0.08:
                    self._handle_degradation()

                time.sleep(0.65)

            with self._lock:
                self._state.status = "completed"
                self._state.stage = "trained-model-ready"
                self._state.controller_state = "completed"
                self._event("info", "Training completed. Final model is ready for inference.")
                if self._runtime is not None:
                    rollback_events = [
                        event for event in self._state.events if "Rollback" in event["message"]
                    ]
                    self._runtime.log_mlflow_text(
                        "rollback_events.json",
                        str(rollback_events),
                    )
                    self._runtime.end_mlflow_run()
        except Exception as exc:  # pragma: no cover - surfaced through UI state.
            with self._lock:
                self._state.status = "failed"
                self._event("error", f"Visual demo failed: {exc}")

    def _train_epoch(self, *, epoch: int, degraded: bool) -> float:
        self._model.train()
        total_loss = 0.0
        total = 0
        train = _build_dataset(samples=160, seed=self._seed + epoch)
        for start in range(0, len(train), 32):
            batch = train[start : start + 32]
            inputs = torch.stack([item[0] for item in batch]).to(self._device)
            targets = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.long,
                device=self._device,
            )
            if degraded:
                targets = 1 - targets
                inputs = torch.clamp(inputs + torch.randn_like(inputs) * 0.22, 0.0, 1.0)

            self._optimizer.zero_grad(set_to_none=True)
            loss = functional.cross_entropy(self._model(inputs), targets)
            loss.backward()  # type: ignore[no-untyped-call]
            self._optimizer.step()

            total_loss += float(loss.item()) * int(targets.numel())
            total += int(targets.numel())
            time.sleep(0.035)

        return total_loss / max(total, 1)

    @torch.inference_mode()
    def _evaluate(self) -> tuple[float, float]:
        self._model.eval()
        inputs = torch.stack([item[0] for item in self._validation]).to(self._device)
        targets = torch.tensor(
            [item[1] for item in self._validation],
            dtype=torch.long,
            device=self._device,
        )
        logits = self._model(inputs)
        loss = functional.cross_entropy(logits, targets)
        accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
        return float(loss.item()), accuracy

    def _handle_degradation(self) -> None:
        decision_id = f"dec_{int(time.time() * 1000)}"
        with self._lock:
            self._state.controller_state = "degradation_detected"
            self._state.decisions.append(
                {
                    "id": decision_id,
                    "action": "rollback",
                    "status": "executed" if self._state.auto_mode else "pending",
                    "reason": "Validation loss degradation detected after LR spike.",
                    "createdAt": _now(),
                }
            )
            self._persist_decision(decision_id)
            self._event("warning", "Validation loss degradation detected.")
            if not self._state.auto_mode:
                self._state.status = "awaiting_approval"
                self._event("warning", "Rollback is waiting for operator approval.")

        if not self.snapshot()["autoMode"]:
            self._approve_event.clear()
            self._approve_event.wait(timeout=120)
            if self._latest_decision_status(decision_id) == "denied":
                return

        self._restore_best_checkpoint()
        _set_lr(self._optimizer, 8e-4)
        self._event("info", "Learning rate reduced for recovery stage.")

    def _restore_best_checkpoint(self) -> None:
        stable = [checkpoint for checkpoint in self._state.checkpoints if checkpoint.stable]
        if not stable:
            return
        best = min(stable, key=lambda checkpoint: checkpoint.validation_loss)
        previous_checkpoint_id = self._state.active_checkpoint_id
        self._model.load_state_dict(best.model_state)
        self._optimizer.load_state_dict(best.optimizer_state)
        self._state.active_checkpoint_id = best.id
        self._state.rollback_count += 1
        self._state.status = "running"
        self._state.stage = "rollback-recovery"
        self._state.controller_state = "checkpoint_restored"
        self._event("warning", "Rollback initiated.")
        self._event("info", f"Checkpoint restored: {best.id}.")
        if self._runtime is not None:
            self._runtime.persist_rollback(
                experiment_id=self._experiment_id,
                from_commit_id=previous_checkpoint_id,
                to_commit_id=best.id,
                artifact_uri=best.artifact_uri,
                reason="Validation degradation rollback",
                mlflow_run_id=self._mlflow_run_id,
            )
            self._runtime.log_mlflow_text("rollback.txt", f"Restored checkpoint {best.id}")

    def _record_metric(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
        accuracy: float,
        learning_rate: float,
    ) -> None:
        with self._lock:
            self._state.metrics.append(
                {
                    "timestamp": _now(),
                    "epoch": epoch,
                    "trainLoss": train_loss,
                    "validationLoss": validation_loss,
                    "accuracy": accuracy,
                    "learningRate": learning_rate,
                    "stage": self._state.stage,
                }
            )
        if self._runtime is not None:
            self._runtime.log_mlflow_metric("train_loss", train_loss, step=epoch)
            self._runtime.log_mlflow_metric("validation_loss", validation_loss, step=epoch)
            self._runtime.log_mlflow_metric("accuracy", accuracy, step=epoch)
            self._runtime.log_mlflow_metric("learning_rate", learning_rate, step=epoch)

    def _save_checkpoint(
        self,
        epoch: int,
        validation_loss: float,
        accuracy: float,
        *,
        stable: bool,
    ) -> None:
        checkpoint_id = f"cmt_epoch_{epoch:02d}"
        checkpoint = _CheckpointState(
            id=checkpoint_id,
            epoch=epoch,
            validation_loss=validation_loss,
            accuracy=accuracy,
            model_state=copy.deepcopy(self._model.state_dict()),
            optimizer_state=copy.deepcopy(self._optimizer.state_dict()),
            stable=stable,
            created_at=_now(),
        )
        payload: CheckpointArtifactPayload = {
            "epoch": epoch,
            "global_step": epoch * 1000 + int(time.time()) % 1000,
            "best_validation_loss": validation_loss if stable else None,
            "model_state": copy.deepcopy(self._model.state_dict()),
            "optimizer_state": copy.deepcopy(self._optimizer.state_dict()),
            "scheduler_state": None,
            "scaler_state": None,
        }
        if self._runtime is not None:
            artifact = self._runtime.save_checkpoint(
                name=f"visual-demo/{checkpoint_id}.pt",
                payload=payload,
            )
            if artifact is not None:
                checkpoint.artifact_uri = artifact.uri
                checkpoint.checksum = artifact.checksum
                checkpoint.size_bytes = artifact.size_bytes
                checkpoint.mlflow_run_id = self._mlflow_run_id
                checkpoint.storage = "minio"
                commit = self._runtime.persist_checkpoint_commit(
                    experiment_id=self._experiment_id,
                    checkpoint_id=f"{checkpoint_id}_{int(time.time() * 1000)}",
                    artifact=artifact,
                    metrics={
                        "epoch": epoch,
                        "validation_loss": validation_loss,
                        "accuracy": accuracy,
                        "stable": stable,
                        "stage": self._state.stage,
                    },
                    mlflow_run_id=self._mlflow_run_id,
                )
                if commit is not None:
                    checkpoint.id = commit.id
        with self._lock:
            self._state.checkpoints.append(checkpoint)
            self._state.active_checkpoint_id = checkpoint.id
            self._event("info", f"Checkpoint committed: {checkpoint.id}.")

    @torch.inference_mode()
    def _refresh_predictions(self) -> None:
        samples = self._validation[:8]
        predictions: list[VisualDemoPrediction] = []
        for index, (image, target) in enumerate(samples):
            predicted, confidence = self._predict_tensor(image.to(self._device))
            actual = _class_name(target)
            predictions.append(
                {
                    "id": f"sample_{index}",
                    "image": _image_data_url(image),
                    "actualClass": actual,
                    "predictedClass": predicted,
                    "confidence": confidence,
                    "correct": predicted == actual,
                }
            )
        with self._lock:
            self._state.predictions = predictions

    @torch.inference_mode()
    def _predict_tensor(self, image: Tensor) -> tuple[ClassName, float]:
        self._model.eval()
        logits = self._model(image.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        index = int(probabilities.argmax().item())
        return _class_name(index), float(probabilities[index].item())

    def _wait_if_paused(self) -> None:
        while True:
            with self._lock:
                paused = self._pause_requested or self._state.status == "paused"
            if not paused:
                return
            time.sleep(0.2)

    def _latest_decision_status(self, decision_id: str) -> str:
        with self._lock:
            for decision in self._state.decisions:
                if decision["id"] == decision_id:
                    return decision["status"]
        return "pending"

    def _reset(self, *, auto_mode: bool) -> None:
        random.seed(self._seed)
        torch.manual_seed(self._seed)
        self._state = _VisualDemoState(auto_mode=auto_mode)
        self._pause_requested = False
        self._stop_requested = False
        self._approve_event.clear()
        self._model = TinyVisualClassifier().to(self._device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=2e-3)
        self._mlflow_run_id = None
        self._inference_history = []
        self._refresh_predictions()
        self._event("info", "Untrained baseline predictions are visible before learning starts.")

    def _event(self, level: str, message: str) -> None:
        self._state.events.append(
            {
                "id": f"evt_{int(time.time() * 1000)}_{len(self._state.events)}",
                "level": level,
                "message": message,
                "createdAt": _now(),
            }
        )

    def _start_runtime_run(self) -> None:
        if self._runtime is None:
            self._event("warning", "Runtime stack is offline; using in-memory demo mode.")
            return
        self._mlflow_run_id = self._runtime.start_mlflow_run(
            run_name=f"visual-demo-{int(time.time())}",
            params={
                "dataset": "generated-airplane-vs-ship",
                "model": "TinyVisualClassifier",
                "controller_mode": "auto" if self._state.auto_mode else "manual",
            },
        )
        self._runtime.ensure_visual_experiment(
            experiment_id=self._experiment_id,
            run_id=self._mlflow_run_id,
        )
        self._event("info", "Runtime stack connected: PostgreSQL, MLflow and MinIO are active.")

    def _persist_decision(self, decision_id: str) -> None:
        if self._runtime is None:
            return
        self._runtime.persist_decision(
            experiment_id=self._experiment_id,
            decision_id=decision_id,
            action="rollback",
            status="executed" if self._state.auto_mode else "pending",
            reason="Validation loss degradation detected after LR spike.",
            commit_id=self._state.active_checkpoint_id,
            mlflow_run_id=self._mlflow_run_id,
            metadata={"controller_state": self._state.controller_state},
        )

    def _resolve_checkpoint(self, checkpoint_id: str | None) -> _CheckpointState:
        if not self._state.checkpoints:
            return _CheckpointState(
                id="untrained",
                epoch=0,
                validation_loss=0.0,
                accuracy=0.0,
                model_state=copy.deepcopy(self._model.state_dict()),
                optimizer_state=copy.deepcopy(self._optimizer.state_dict()),
                stable=False,
                created_at=_now(),
            )
        if checkpoint_id == "earliest":
            return self._state.checkpoints[0]
        if checkpoint_id is None or checkpoint_id == "latest":
            return self._state.checkpoints[-1]
        for checkpoint in self._state.checkpoints:
            if checkpoint.id == checkpoint_id:
                return checkpoint
        return self._state.checkpoints[-1]


def _build_dataset(*, samples: int, seed: int) -> list[tuple[Tensor, int]]:
    rng = random.Random(seed)  # noqa: S311 - deterministic visual demo data generation.
    return [_render_sample(label=index % 2, rng=rng) for index in range(samples)]


def _render_sample(*, label: int, rng: random.Random) -> tuple[Tensor, int]:
    image = Image.new("RGB", (32, 32), color=(128, 190, 235) if label == 0 else (82, 170, 220))
    draw = ImageDraw.Draw(image)
    if label == 0:
        offset = rng.randint(-3, 3)
        draw.polygon([(7, 18 + offset), (25, 12 + offset), (21, 18 + offset)], fill=(238, 242, 247))
        draw.line([(12, 18 + offset), (7, 23 + offset)], fill=(238, 242, 247), width=2)
        draw.line([(18, 17 + offset), (25, 22 + offset)], fill=(238, 242, 247), width=2)
    else:
        draw.rectangle([(0, 18), (32, 32)], fill=(24, 92, 170))
        offset = rng.randint(-2, 2)
        draw.polygon(
            [(7, 19 + offset), (25, 19 + offset), (21, 24 + offset), (10, 24 + offset)],
            fill=(111, 78, 55),
        )
        draw.line([(16, 18 + offset), (16, 9 + offset)], fill=(245, 245, 235), width=1)
        draw.polygon(
            [(17, 10 + offset), (24, 16 + offset), (17, 16 + offset)],
            fill=(245, 245, 235),
        )

    return _pil_to_tensor(image), label


def _pil_to_tensor(image: object) -> Tensor:
    pixels = list(cast(Image.Image, image).getdata())
    raw = torch.tensor(pixels, dtype=torch.uint8)
    return raw.view(32, 32, 3).permute(2, 0, 1).float() / 255.0


def _image_data_url(tensor: Tensor) -> str:
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="RGB").resize((96, 96))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _class_name(index: int) -> ClassName:
    return "airplane" if index == 0 else "ship"


def _set_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _gpu_usage(device: torch.device) -> dict[str, str | float | None]:
    if device.type != "cuda":
        return {"device": "cpu", "memoryAllocatedMb": None, "memoryReservedMb": None}
    return {
        "device": torch.cuda.get_device_name(device),
        "memoryAllocatedMb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
        "memoryReservedMb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
    }


def _runtime_status_dict(status: RuntimeStatus | None) -> dict[str, dict[str, str | bool]]:
    if status is None:
        return {
            "postgres": {"connected": False, "message": "not configured"},
            "mlflow": {"connected": False, "message": "not configured"},
            "minio": {"connected": False, "message": "not configured"},
            "artifactStorage": {"connected": False, "message": "not configured"},
        }
    return {
        "postgres": {
            "connected": status.postgres.connected,
            "message": status.postgres.message,
        },
        "mlflow": {
            "connected": status.mlflow.connected,
            "message": status.mlflow.message,
        },
        "minio": {
            "connected": status.minio.connected,
            "message": status.minio.message,
        },
        "artifactStorage": {
            "connected": status.artifact_storage.connected,
            "message": status.artifact_storage.message,
        },
    }


def _inference_response(result: InferenceResult) -> VisualDemoInference:
    return {
        "predictedClass": result.predicted_class,
        "confidence": result.confidence,
        "checkpointId": result.checkpoint_id,
        "modelVersion": result.model_version,
        "latencyMs": result.latency_ms,
    }


def _comparison_response(result: InferenceComparisonResult) -> VisualDemoComparison:
    return {
        "early": _inference_response(result.baseline),
        "selected": _inference_response(result.candidate),
    }


def _best_accuracy(snapshot: VisualDemoSnapshot) -> float:
    if not snapshot["metrics"]:
        return 0.0
    return max(metric["accuracy"] for metric in snapshot["metrics"])


def _guided_report(snapshot: VisualDemoSnapshot) -> str:
    lines = [
        "# ACN Guided Demo Summary",
        "",
        "ACN is an adaptive training control system. It watches model learning, saves model",
        "checkpoints, detects degradation, restores a stable version and keeps the experiment",
        "traceable.",
        "",
        "## Final Showcase",
        "",
        f"- Best model version: `{snapshot['activeCheckpointId'] or 'untrained'}`",
        f"- Best accuracy: `{_best_accuracy(snapshot):.2%}`",
        f"- Rollbacks: `{snapshot['rollbackCount']}`",
        f"- Checkpoints: `{len(snapshot['checkpoints'])}`",
        f"- MLflow run: `{snapshot['mlflowRunId'] or 'offline'}`",
        "",
        "## Why This Matters",
        "",
        "- Less manual experiment babysitting.",
        "- Safer training because unstable model states can be rolled back.",
        "- Better traceability through checkpoints, commits and event history.",
        "- A usable trained model can be tested immediately after training.",
        "",
    ]
    return "\n".join(lines)


def _screenshot_svg(snapshot: VisualDemoSnapshot) -> str:
    accuracy = _best_accuracy(snapshot) * 100
    rollback_count = snapshot["rollbackCount"]
    checkpoint_count = len(snapshot["checkpoints"])
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720">',
            '<rect width="1280" height="720" fill="#020617"/>',
            '<text x="72" y="96" fill="#67e8f9" font-size="28" font-family="Arial">',
            "Adaptive Core Network Demo</text>",
            '<text x="72" y="156" fill="#f8fafc" font-size="48" font-family="Arial">',
            "Learn. Fail. Recover. Improve.</text>",
            (
                '<text x="72" y="250" fill="#22c55e" font-size="32" '
                f'font-family="Arial">Best accuracy: {accuracy:.1f}%</text>'
            ),
            (
                '<text x="72" y="305" fill="#f59e0b" font-size="32" '
                f'font-family="Arial">Rollbacks: {rollback_count}</text>'
            ),
            (
                '<text x="72" y="360" fill="#38bdf8" font-size="32" '
                f'font-family="Arial">Checkpoints: {checkpoint_count}</text>'
            ),
            "</svg>",
        ]
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()


visual_demo_session = VisualDemoSession()
