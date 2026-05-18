from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol, Self

from sqlalchemy.orm import Session


class UnitOfWork(Protocol):
    def transaction(self) -> AbstractContextManager[object]: ...


class SqlAlchemySessionUnitOfWork:
    def __init__(self, session: Session) -> None:
        self._session = session

    def transaction(self) -> AbstractContextManager[object]:
        if self._session.in_transaction():
            return self._session.begin_nested()
        return self._session.begin()


class SqlAlchemyUnitOfWork:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory
        self.session: Session | None = None

    def __enter__(self) -> Self:
        self.session = self._session_factory()
        self.session.begin()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.session is None:
            return
        try:
            if exc_type is None:
                self.session.commit()
            else:
                self.session.rollback()
        finally:
            self.session.close()

    def require_session(self) -> Session:
        if self.session is None:
            msg = "Unit of work session is not active."
            raise RuntimeError(msg)
        return self.session

    def transaction(self) -> AbstractContextManager[object]:
        return self
