from pathlib import Path
from typing import Protocol

import torch

from acn.artifacts.domain import ArtifactReference
from acn.artifacts.models import CheckpointArtifactPayload


class ArtifactStore(Protocol):
    def save_checkpoint(
        self,
        *,
        name: str,
        payload: CheckpointArtifactPayload,
    ) -> ArtifactReference: ...

    def load_checkpoint(
        self,
        uri: str | Path,
        *,
        expected_checksum: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointArtifactPayload: ...

    def delete_checkpoint(self, uri: str | Path) -> None: ...

    def exists(self, uri: str | Path) -> bool: ...

    def checksum(self, uri: str | Path) -> str: ...
