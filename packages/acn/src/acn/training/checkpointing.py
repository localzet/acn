from pathlib import Path
from typing import Any, TypedDict, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from acn.training.config import CheckpointState


class CheckpointPayload(TypedDict):
    epoch: int
    global_step: int
    best_validation_loss: float | None
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any] | None
    scaler_state: dict[str, Any] | None


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        scaler: torch.GradScaler | None,
        state: CheckpointState,
        name: str | None = None,
    ) -> Path:
        checkpoint_name = name or f"epoch-{state.epoch:04d}.pt"
        path = self._checkpoint_dir / checkpoint_name
        payload: CheckpointPayload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_validation_loss": state.best_validation_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(payload, path)
        return path

    def load(
        self,
        path: Path,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: torch.GradScaler | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointState:
        payload = cast(CheckpointPayload, torch.load(path, map_location=map_location))
        model.load_state_dict(payload["model_state"])

        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload["scheduler_state"] is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload["scaler_state"] is not None:
            scaler.load_state_dict(payload["scaler_state"])

        return CheckpointState(
            epoch=payload["epoch"],
            global_step=payload["global_step"],
            best_validation_loss=payload["best_validation_loss"],
        )
