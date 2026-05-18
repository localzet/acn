import os
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

import torch

from acn.artifacts.domain import (
    ArtifactChecksumMismatchError,
    ArtifactCorruptedError,
    ArtifactNotFoundError,
    ArtifactReference,
    UnsupportedArtifactURIError,
)
from acn.artifacts.models import CheckpointArtifactPayload

CHECKSUM_PREFIX = "sha256:"


class LocalArtifactStore:
    def __init__(self, root: Path, *, checkpoint_subdir: str = "checkpoints") -> None:
        self._root = root
        self._checkpoint_root = self._root / checkpoint_subdir if checkpoint_subdir else self._root
        self._checkpoint_root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def save_checkpoint(
        self,
        *,
        name: str,
        payload: CheckpointArtifactPayload,
    ) -> ArtifactReference:
        path = self._checkpoint_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

        return ArtifactReference(
            uri=_file_uri(path),
            checksum=self.checksum(path),
            size_bytes=path.stat().st_size,
        )

    def load_checkpoint(
        self,
        uri: str | Path,
        *,
        expected_checksum: str | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointArtifactPayload:
        path = self._path_from_uri(uri)
        self._require_exists(path)
        if expected_checksum is not None:
            actual_checksum = self.checksum(path)
            if _normalize_checksum(actual_checksum) != _normalize_checksum(expected_checksum):
                msg = (
                    f"Artifact checksum mismatch for {path}: "
                    f"expected {expected_checksum}, got {actual_checksum}."
                )
                raise ArtifactChecksumMismatchError(msg)
        try:
            payload = torch.load(path, map_location=map_location)
        except Exception as exc:
            msg = f"Artifact exists but could not be loaded as a checkpoint: {path}"
            raise ArtifactCorruptedError(msg) from exc
        return cast(CheckpointArtifactPayload, payload)

    def delete_checkpoint(self, uri: str | Path) -> None:
        path = self._path_from_uri(uri)
        path.unlink(missing_ok=True)

    def exists(self, uri: str | Path) -> bool:
        return self._path_from_uri(uri).exists()

    def checksum(self, uri: str | Path) -> str:
        path = self._path_from_uri(uri)
        self._require_exists(path)
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{CHECKSUM_PREFIX}{digest.hexdigest()}"

    def _checkpoint_path(self, name: str) -> Path:
        relative_name = Path(name)
        if relative_name.is_absolute() or ".." in relative_name.parts:
            msg = f"Checkpoint name must be relative to the artifact root: {name}"
            raise ValueError(msg)
        return self._checkpoint_root / relative_name

    def _path_from_uri(self, uri: str | Path) -> Path:
        if isinstance(uri, Path):
            return uri
        parsed = urlparse(uri)
        if parsed.scheme == "":
            return Path(uri)
        if parsed.scheme != "file":
            msg = f"LocalArtifactStore only supports file:// URIs, got: {uri}"
            raise UnsupportedArtifactURIError(msg)
        return Path(unquote(parsed.path))

    def _require_exists(self, path: Path) -> None:
        if not path.exists():
            msg = f"Artifact not found: {path}"
            raise ArtifactNotFoundError(msg)


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _normalize_checksum(value: str) -> str:
    return value.removeprefix(CHECKSUM_PREFIX)
