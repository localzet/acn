import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from acn.training.checkpointing import CheckpointManager
from acn.training.config import CheckpointState, EpochMetrics, TrainerConfig, TrainingHistory

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Batch:
    inputs: Tensor
    targets: Tensor


class Trainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        config: TrainerConfig,
        scheduler: LRScheduler | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = _resolve_device(config.device)
        self.model.to(self.device)
        self.criterion.to(self.device)

        self._amp_enabled = config.mixed_precision and self.device.type == "cuda"
        self._scaler = torch.GradScaler(device=self.device.type, enabled=self._amp_enabled)
        self._checkpoint_manager = checkpoint_manager or _build_checkpoint_manager(
            config.checkpoint_dir
        )
        self._state = CheckpointState()

    @property
    def state(self) -> CheckpointState:
        return self._state

    def fit(
        self,
        *,
        train_loader: DataLoader[object],
        validation_loader: DataLoader[object] | None = None,
    ) -> TrainingHistory:
        history = TrainingHistory()
        for epoch in range(self._state.epoch + 1, self.config.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            history.train.append(train_metrics)

            validation_metrics = None
            if validation_loader is not None:
                validation_metrics = self.validate(validation_loader)
                history.validation.append(validation_metrics)
                self._update_best_validation_loss(validation_metrics.loss)

            if self.scheduler is not None:
                self.scheduler.step()

            self._state.epoch = epoch
            self._log_epoch(epoch, train_metrics, validation_metrics)
            self._save_epoch_checkpoint()

        return history

    def train_epoch(self, train_loader: DataLoader[object]) -> EpochMetrics:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        has_accuracy = False
        total_examples = 0

        for step, raw_batch in enumerate(train_loader, start=1):
            batch = _move_batch(_parse_batch(raw_batch), self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type,
                enabled=self._amp_enabled,
            ):
                outputs = self.model(batch.inputs)
                loss = self.criterion(outputs, batch.targets)

            self._scaler.scale(loss).backward()
            if self.config.max_grad_norm is not None:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            batch_size = _batch_size(batch.targets)
            total_loss += float(loss.detach().item()) * batch_size
            correct = _count_correct(outputs.detach(), batch.targets)
            if correct is not None:
                total_correct += correct
                has_accuracy = True
            total_examples += batch_size
            self._state.global_step += 1
            self._log_step(step, total_loss, total_examples)

        return _build_metrics(total_loss, total_correct, total_examples, has_accuracy=has_accuracy)

    @torch.inference_mode()
    def validate(self, validation_loader: DataLoader[object]) -> EpochMetrics:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        has_accuracy = False
        total_examples = 0

        for raw_batch in validation_loader:
            batch = _move_batch(_parse_batch(raw_batch), self.device)
            outputs = self.model(batch.inputs)
            loss = self.criterion(outputs, batch.targets)

            batch_size = _batch_size(batch.targets)
            total_loss += float(loss.item()) * batch_size
            correct = _count_correct(outputs, batch.targets)
            if correct is not None:
                total_correct += correct
                has_accuracy = True
            total_examples += batch_size

        return _build_metrics(total_loss, total_correct, total_examples, has_accuracy=has_accuracy)

    def load_checkpoint(self, path: Path) -> None:
        if self._checkpoint_manager is None:
            msg = "Checkpoint manager is not configured."
            raise RuntimeError(msg)

        self._state = self._checkpoint_manager.load(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self._scaler,
            map_location=self.device,
        )
        logger.info("checkpoint.loaded", extra={"path": str(path), "epoch": self._state.epoch})

    def _update_best_validation_loss(self, validation_loss: float) -> None:
        if (
            self._state.best_validation_loss is None
            or validation_loss < self._state.best_validation_loss
        ):
            self._state.best_validation_loss = validation_loss

    def _save_epoch_checkpoint(self) -> None:
        if self._checkpoint_manager is None:
            return
        if self.config.checkpoint_every_n_epochs <= 0:
            return
        if self._state.epoch % self.config.checkpoint_every_n_epochs != 0:
            return

        path = self._checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self._scaler,
            state=self._state,
        )
        logger.info("checkpoint.saved", extra={"path": str(path), "epoch": self._state.epoch})

    def _log_step(self, step: int, total_loss: float, total_examples: int) -> None:
        if self.config.log_every_n_steps <= 0:
            return
        if step % self.config.log_every_n_steps != 0:
            return

        logger.info(
            "train.step",
            extra={
                "epoch": self._state.epoch + 1,
                "step": step,
                "loss": total_loss / max(total_examples, 1),
            },
        )

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: EpochMetrics,
        validation_metrics: EpochMetrics | None,
    ) -> None:
        extra: dict[str, float | int | None] = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_accuracy": train_metrics.accuracy,
        }
        if validation_metrics is not None:
            extra["validation_loss"] = validation_metrics.loss
            extra["validation_accuracy"] = validation_metrics.accuracy
        logger.info("train.epoch", extra=extra)


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_checkpoint_manager(checkpoint_dir: Path | None) -> CheckpointManager | None:
    if checkpoint_dir is None:
        return None
    return CheckpointManager(checkpoint_dir)


def _parse_batch(raw_batch: object) -> Batch:
    if isinstance(raw_batch, Mapping):
        return Batch(
            inputs=_require_tensor(raw_batch["inputs"]),
            targets=_require_tensor(raw_batch["targets"]),
        )
    if isinstance(raw_batch, (tuple, list)) and len(raw_batch) >= 2:
        return Batch(inputs=_require_tensor(raw_batch[0]), targets=_require_tensor(raw_batch[1]))

    msg = "Batch must be a mapping with inputs/targets or a sequence of at least two tensors."
    raise TypeError(msg)


def _require_tensor(value: object) -> Tensor:
    if not isinstance(value, Tensor):
        msg = f"Expected torch.Tensor, got {type(value).__name__}."
        raise TypeError(msg)
    return value


def _move_batch(batch: Batch, device: torch.device) -> Batch:
    return Batch(
        inputs=batch.inputs.to(device, non_blocking=True),
        targets=batch.targets.to(device, non_blocking=True),
    )


def _batch_size(targets: Tensor) -> int:
    if targets.ndim == 0:
        return 1
    return int(targets.size(0))


def _count_correct(outputs: Tensor, targets: Tensor) -> int | None:
    if outputs.ndim < 2 or targets.ndim != 1:
        return None
    predictions = outputs.argmax(dim=1)
    return int((predictions == targets).sum().item())


def _build_metrics(
    total_loss: float,
    total_correct: int,
    total_examples: int,
    *,
    has_accuracy: bool,
) -> EpochMetrics:
    if total_examples <= 0:
        msg = "DataLoader produced no examples."
        raise ValueError(msg)

    accuracy = total_correct / total_examples if has_accuracy else None
    return EpochMetrics(loss=total_loss / total_examples, accuracy=accuracy)
