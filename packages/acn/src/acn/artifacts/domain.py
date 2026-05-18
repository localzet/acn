from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    uri: str
    checksum: str
    size_bytes: int


class ArtifactError(RuntimeError):
    """Base error for artifact lifecycle operations."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when an artifact cannot be found."""


class ArtifactChecksumMismatchError(ArtifactError):
    """Raised when artifact bytes do not match the expected checksum."""


class ArtifactCorruptedError(ArtifactError):
    """Raised when artifact bytes exist but cannot be loaded as a checkpoint."""


class UnsupportedArtifactURIError(ArtifactError):
    """Raised when a store cannot handle the artifact URI."""
