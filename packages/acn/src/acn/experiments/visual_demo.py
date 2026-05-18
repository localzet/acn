"""Live visual adaptive-training demo.

This module is a presentation-oriented local demo. It trains a small CNN on a
lightweight visual airplane-vs-ship dataset and exposes state snapshots for the
FastAPI demo endpoints.
"""

import base64
import copy
import io
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, TypedDict, cast

import torch
import torch.nn.functional as functional
from PIL import Image, ImageDraw
from torch import Tensor, nn

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
                    }
                    for checkpoint in self._state.checkpoints
                ],
                "predictions": list(self._state.predictions),
                "events": list(self._state.events[-80:]),
                "decisions": list(self._state.decisions[-20:]),
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

    def predict_data_url(self, image_data_url: str) -> dict[str, str | float]:
        tensor = _tensor_from_data_url(image_data_url).to(self._device)
        predicted, confidence = self._predict_tensor(tensor)
        return {
            "predictedClass": predicted,
            "confidence": confidence,
            "modelCheckpoint": self.snapshot()["activeCheckpointId"] or "uncommitted",
        }

    def _run(self) -> None:
        try:
            random.seed(self._seed)
            torch.manual_seed(self._seed)
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
        self._model.load_state_dict(best.model_state)
        self._optimizer.load_state_dict(best.optimizer_state)
        self._state.active_checkpoint_id = best.id
        self._state.rollback_count += 1
        self._state.status = "running"
        self._state.stage = "rollback-recovery"
        self._state.controller_state = "checkpoint_restored"
        self._event("warning", "Rollback initiated.")
        self._event("info", f"Checkpoint restored: {best.id}.")

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
        with self._lock:
            self._state.checkpoints.append(checkpoint)
            self._state.active_checkpoint_id = checkpoint_id
            self._event("info", f"Checkpoint committed: {checkpoint_id}.")

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

    def _event(self, level: str, message: str) -> None:
        self._state.events.append(
            {
                "id": f"evt_{int(time.time() * 1000)}_{len(self._state.events)}",
                "level": level,
                "message": message,
                "createdAt": _now(),
            }
        )


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


def _tensor_from_data_url(data_url: str) -> Tensor:
    _, _, payload = data_url.partition(",")
    image = Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB").resize((32, 32))
    return _pil_to_tensor(image)


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


def _now() -> str:
    return datetime.now(UTC).isoformat()


visual_demo_session = VisualDemoSession()
