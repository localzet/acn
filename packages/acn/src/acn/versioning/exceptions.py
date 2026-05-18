class VersioningError(RuntimeError):
    """Base error for training version store operations."""


class BranchAlreadyExistsError(VersioningError):
    """Raised when a branch name is already registered."""


class BranchNotFoundError(VersioningError):
    """Raised when a branch cannot be found."""


class CommitNotFoundError(VersioningError):
    """Raised when a commit cannot be found."""


class CheckpointNotFoundError(VersioningError):
    """Raised when a stable checkpoint cannot be found."""


class InvalidRollbackTargetError(VersioningError):
    """Raised when rollback target is not reachable from the current branch head."""


class BranchHeadConflictError(VersioningError):
    """Raised when a branch head changed before a guarded update."""


class ImmutableCheckpointError(VersioningError):
    """Raised when immutable checkpoint metadata is modified or deleted."""
