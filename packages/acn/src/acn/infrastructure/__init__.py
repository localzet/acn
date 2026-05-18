"""Infrastructure adapters with explicit ownership.

This package is intentionally small. It exists because ACN currently has concrete
transaction boundary adapters (`UnitOfWork`) shared by orchestration and repository
coordination. Do not add generic service abstractions here without a real external
system boundary.
"""

from acn.infrastructure.uow import (
    SqlAlchemySessionUnitOfWork,
    SqlAlchemyUnitOfWork,
    UnitOfWork,
)

__all__ = [
    "SqlAlchemySessionUnitOfWork",
    "SqlAlchemyUnitOfWork",
    "UnitOfWork",
]
