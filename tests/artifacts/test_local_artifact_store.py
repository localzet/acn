from pathlib import Path

import pytest
import torch

from acn.artifacts import (
    ArtifactChecksumMismatchError,
    ArtifactCorruptedError,
    ArtifactNotFoundError,
    LocalArtifactStore,
)
from acn.artifacts.models import CheckpointArtifactPayload


def _payload(epoch: int = 1) -> CheckpointArtifactPayload:
    return {
        "epoch": epoch,
        "global_step": epoch * 10,
        "best_validation_loss": 0.25,
        "model_state": {"weight": torch.tensor([float(epoch)])},
        "optimizer_state": {"state": {}, "param_groups": []},
        "scheduler_state": None,
        "scaler_state": None,
    }


def test_local_artifact_store_saves_and_loads_checkpoint(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    reference = store.save_checkpoint(name="run-1/epoch-0001.pt", payload=_payload())
    loaded = store.load_checkpoint(reference.uri, expected_checksum=reference.checksum)

    assert reference.uri.startswith("file://")
    assert reference.checksum.startswith("sha256:")
    assert reference.size_bytes > 0
    assert store.exists(reference.uri)
    assert loaded["epoch"] == 1
    assert torch.equal(loaded["model_state"]["weight"], torch.tensor([1.0]))


def test_local_artifact_store_detects_checksum_mismatch(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    reference = store.save_checkpoint(name="epoch-0001.pt", payload=_payload())

    with pytest.raises(ArtifactChecksumMismatchError):
        store.load_checkpoint(reference.uri, expected_checksum="sha256:bad")


def test_local_artifact_store_fails_on_missing_artifact(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactNotFoundError):
        store.load_checkpoint(tmp_path / "checkpoints" / "missing.pt")


def test_local_artifact_store_detects_corrupted_checkpoint_bytes(tmp_path: Path) -> None:
    corrupted = tmp_path / "checkpoints" / "corrupted.pt"
    corrupted.parent.mkdir(parents=True)
    corrupted.write_bytes(b"not a torch checkpoint")
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactCorruptedError):
        store.load_checkpoint(corrupted)


def test_local_artifact_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        store.save_checkpoint(name="../escape.pt", payload=_payload())
