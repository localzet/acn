from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from acn.artifacts import ArtifactReference, ArtifactStore, LocalArtifactStore
from acn.artifacts.models import CheckpointArtifactPayload
from acn.training.config import CheckpointState

CheckpointPayload = CheckpointArtifactPayload


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, artifact_store: ArtifactStore | None = None) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._artifact_store = artifact_store or LocalArtifactStore(
            checkpoint_dir,
            checkpoint_subdir="",
        )

    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._artifact_store

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        scaler: torch.GradScaler | None,
        state: CheckpointState,
        name: str | None = None,
    ) -> ArtifactReference:
        checkpoint_name = name or f"epoch-{state.epoch:04d}.pt"
        payload: CheckpointPayload = {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "best_validation_loss": state.best_validation_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        return self._artifact_store.save_checkpoint(name=checkpoint_name, payload=payload)

    def load(
        self,
        path: Path | str | ArtifactReference,
        *,
        model: nn.Module,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        scaler: torch.GradScaler | None = None,
        map_location: str | torch.device = "cpu",
        expected_checksum: str | None = None,
    ) -> CheckpointState:
        uri = path.uri if isinstance(path, ArtifactReference) else path
        checksum = path.checksum if isinstance(path, ArtifactReference) else expected_checksum
        payload = self._artifact_store.load_checkpoint(
            uri,
            expected_checksum=checksum,
            map_location=map_location,
        )
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
